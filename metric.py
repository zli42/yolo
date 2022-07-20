import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision import transforms
from torchvision.datasets import VOCDetection
from tqdm import tqdm

import utils
from model import YOLO

device = torch.device("cpu" if torch.cuda.is_available else "cpu")


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

transform = transforms.Compose(
    (
        SquarePad(),
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    )
)

dataset = VOCDetection("./data", image_set="val", download=True, transform=transform)


with open("./data/classes.txt", "rt") as f:
    cls = {line.strip() for line in f}
cls_idx = dict(zip(cls, range(len(cls))))
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


dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)


anchors = []
with open("./data/anchors.txt", "rt") as f:
    for line in f:
        anchors.extend([float(x) for x in line.strip().split(",")])
anchors = torch.tensor(anchors).view(3, 3, 2)
num_anchors = anchors.size(1)

checkpoint = torch.load("./data/checkpoint.pt", map_location=device)

model = YOLO(num_anchors, num_classes).to(device)
model.load_state_dict(checkpoint["model"])

metric = MeanAveragePrecision()
preds = []
targets = []
model.eval()
with torch.no_grad():
    for j, (X, y, _) in enumerate(dataloader):
        if j >= 4:
            break
        X, y = X.to(device), y.to(device)
        pred = model(X)
        result, loss = utils.decode_predictions(anchors, pred, y)
        outputs = utils.postprocess_result(result)
        for i, (b, s, l) in enumerate(outputs):
            preds.append({"boxes": b, "scores": s, "labels": l})
            targets.append({"boxes": y[i][:, :4], "labels": y[i][:, -1]})
            metric.update(preds, targets)

print(len(preds), len(targets))
for k, v in preds[0].items():
    print(k, v.shape)
for k, v in targets[0].items():
    print(k, v.shape)
for k, v in metric.compute():
    print(f"{k}: {v.item()}")
