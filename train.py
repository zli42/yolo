import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision import transforms
from torchvision.datasets import VOCDetection
from tqdm import tqdm

import utils
from model import YOLO


if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = torch.device("cuda")
else:
    torch.set_default_tensor_type(torch.FloatTensor)
    device = torch.device("cpu")

phases = ("train", "val")


class SquarePad:
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [
            max_wh - (s + p) for s, p in zip(image.size, (p_left, p_top))
        ]
        padding = (p_left, p_top, p_right, p_bottom)
        return transforms.functional.pad(image, padding, 0, "constant")


img_height = 416
img_width = 416


def get_transform(phase):
    transform_list = []
    if phase == "train":
        transform_list.append(transforms.ColorJitter(0.1, 0.1, 0.1, 0.1))
    transform_list.append(SquarePad())
    transform_list.append(transforms.Resize((img_height, img_width)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    )
    return transforms.Compose(transform_list)


transform = {phase: get_transform(phase) for phase in phases}

dataset = {
    phase: VOCDetection(
        "./data",
        image_set=phase,
        download=True,
        transform=transform[phase],
    )
    for phase in phases
}


def get_classes(class_path):
    with open(class_path, "rt") as f:
        cls = {line.strip() for line in f}
    return dict(zip(cls, range(len(cls))))


cls_idx = get_classes("./data/classes.txt")
num_classes = len(cls_idx)
idx_cls = {v: k for k, v in cls_idx.items()}


def collate_fn(sample):
    images = []
    max_num_box = max([len(each[1]["annotation"]["object"]) for each in sample])
    boxes = torch.full((len(sample), max_num_box, 5), -1)
    filenames = []
    for i, (image, target) in enumerate(sample):
        images.append(image)
        width = int(target["annotation"]["size"]["width"])
        height = int(target["annotation"]["size"]["height"])
        max_wh = max(width, height)
        x_offset = (max_wh - width) // 2
        y_offset = (max_wh - height) // 2
        for j, each in enumerate(target["annotation"]["object"]):
            cls = cls_idx[each["name"]]
            x_1 = (int(each["bndbox"]["xmin"]) + x_offset) * img_width / max_wh
            y_1 = (int(each["bndbox"]["ymin"]) + y_offset) * img_height / max_wh
            x_2 = (int(each["bndbox"]["xmax"]) + x_offset) * img_width / max_wh
            y_2 = (int(each["bndbox"]["ymax"]) + y_offset) * img_height / max_wh
            boxes[i][j] = torch.tensor((x_1, y_1, x_2, y_2, cls))
        filenames.append(target["annotation"]["filename"])
    return torch.stack(images, 0), boxes, filenames


dataloader = {
    phase: DataLoader(
        dataset[phase],
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn,
        generator=torch.Generator(device),
    )
    for phase in phases
}


def get_anchors(anchor_path):
    anchors = []
    with open(anchor_path, "rt") as f:
        for line in f:
            anchors.extend([float(x) for x in line.strip().split(",")])
    return torch.tensor(anchors).view(3, 3, 2)


anchors = get_anchors("./data/anchors.txt")
num_anchors = anchors.size(1)

checkpoint_pt = "./data/checkpoint.pt"
checkpoint = torch.load(checkpoint_pt, map_location=device)

model = YOLO(num_anchors, num_classes).to(device)
model.load_state_dict(checkpoint["model"])

metric = MeanAveragePrecision()
preds = []
targets = []
model.eval()
with torch.no_grad():
    for X, y, _ in tqdm(dataloader["val"]):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        result, loss = utils.decode_predictions(anchors, pred, y)
        outputs = utils.postprocess_result(result)
        for i, (b, s, l) in enumerate(outputs):
            preds.append({"boxes": b, "scores": s, "labels": l})
            targets.append({"boxes": y[i][:, :4], "labels": y[i][:, -1]})
metric.update(preds, targets)
for k, v in metric.compute():
    print(f"{k}: {v.item()}")

lr = 1e-2
epochs = 100
swa_start = epochs - 30

for param in model.backbone.parameters():
    param.requires_grad = False
params = [param for param in model.parameters() if param.requires_grad]

optimizer = torch.optim.AdamW(params, lr=lr, amsgrad=False)
optimizer.load_state_dict(checkpoint["optimizer"])

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=lr,
    epochs=epochs,
    steps_per_epoch=len(dataloader["train"]),
    three_phase=True,
)
scheduler.load_state_dict(checkpoint["scheduler"])

ema_avg = (
    lambda averaged_model_parameter, model_parameter, num_averaged: 0.1
    * averaged_model_parameter
    + 0.9 * model_parameter
)
swa_model = torch.optim.swa_utils.AveragedModel(model, device=device, avg_fn=ema_avg)
swa_model.load_state_dict(checkpoint["swa_model"])

swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=lr)
swa_scheduler.load_state_dict(checkpoint["swa_scheduler"])

loss_hist = {phase: [] for phase in phases}
for epoch in range(checkpoint["epoch"] + 1, epochs):
    for phase in phases:
        if phase == "train":
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        with torch.set_grad_enabled(phase == "train"):
            for images, boxes, _ in tqdm(
                dataloader[phase], desc=f"{phase} [{epoch}/{epochs}]"
            ):
                images = images.to(device)
                boxes = boxes.to(device)
                prdictions = model(images)

                _, loss = utils.decode_predictions(anchors, prdictions, boxes)
                running_loss += loss.item()

                if phase == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if epoch < swa_start:
                        scheduler.step()

            if phase == "train" and epoch >= swa_start:
                swa_model.update_parameters(model)
                swa_scheduler.step()

        running_loss /= len(dataloader[phase])
        loss_hist[phase].append(running_loss)

        if phase == "val":
            if epoch == epochs - 1:
                torch.optim.swa_utils.update_bn(dataloader["train"], swa_model)

            x = range(len(loss_hist["train"]))
            plt.plot(x, loss_hist["train"], label="train")
            plt.plot(x, loss_hist["val"], label="val")
            plt.legend()
            plt.savefig("./data/loss_hist.png")

            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "swa_model": swa_model.state_dict(),
                    "swa_scheduler": swa_scheduler.state_dict(),
                    "loss": loss_hist,
                },
                checkpoint_pt,
            )
