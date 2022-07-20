import torch
from torch.nn.functional import binary_cross_entropy_with_logits
from torchvision import ops


def decode_predictions(
    anchors, predictions, targets=torch.tensor(()), weights=(1.0, 1.0, 1.0)
):
    scalers = (32.0, 16.0, 8.0)
    result = []
    total_loss = 0.0
    for i, prediction in enumerate(predictions):
        batch_size, channels, height, width = prediction.shape
        scaler = scalers[i]
        anchor = anchors[i]
        num_anchors = anchor.size(0)
        num_preds = channels // num_anchors
        num_classes = num_preds - 5

        t_x, t_y, t_w, t_h, obj, probs = (
            prediction.permute(0, 2, 3, 1)
            .view(batch_size, height, width, num_anchors, num_preds)
            .split((1, 1, 1, 1, 1, num_classes), -1)
        )

        c_x = (
            torch.arange(width)
            .tile(batch_size, height, num_anchors, 1)
            .permute(0, 1, 3, 2)
            .unsqueeze_(-1)
        )
        c_y = (
            torch.arange(height)
            .tile(batch_size, width, num_anchors, 1)
            .permute(0, 3, 1, 2)
            .unsqueeze_(-1)
        )
        p_w = anchor[:, 0].tile(batch_size, height, width, 1).unsqueeze_(-1)
        p_h = anchor[:, 1].tile(batch_size, height, width, 1).unsqueeze_(-1)

        b_x = (1.1 * torch.sigmoid(t_x) - 0.05 + c_x) * scaler
        b_y = (1.1 * torch.sigmoid(t_y) - 0.05 + c_y) * scaler
        b_w = 4.0 * p_w * torch.sigmoid(t_w)
        b_h = 4.0 * p_h * torch.sigmoid(t_h)

        x_1 = b_x - b_w / 2.0
        y_1 = b_y - b_h / 2.0
        x_2 = b_x + b_w / 2.0
        y_2 = b_y + b_h / 2.0

        pred_boxes = torch.cat((x_1, y_1, x_2, y_2), -1)
        pred_result = torch.cat((pred_boxes, obj, probs), -1)
        result.append(pred_result.view(batch_size, -1, num_preds))

        if torch.equal(targets.float(), torch.tensor(())):
            continue
        target_result = transform_targets(
            targets, pred_boxes, anchor, scaler, num_classes
        )
        loss = get_loss(pred_result, target_result)
        total_loss += loss * weights[i]
    return torch.cat(result, 1), total_loss


def transform_targets(
    targets, pred_boxes, anchor, scaler, num_classes, iou_threshold=0.5
):
    anchor_boxes = torch.cat((torch.zeros_like(anchor), anchor), -1)
    result = []
    for target, pred_box in zip(targets, pred_boxes):
        b_w = target[:, 2] - target[:, 0]
        b_h = target[:, 3] - target[:, 1]
        filter = (b_w > 0.0) * (b_h > 0.0)
        target = target[filter]
        if not target.size(0):
            continue

        boxes, label = target.split((4, 1), -1)
        b_x, b_y, b_w, b_h = ops.box_convert(boxes, "xyxy", "cxcywh").split(1, -1)
        target_boxes = torch.cat((torch.zeros((b_w.size(0), 2)), b_w, b_h), -1)
        bst_anchor_idx = torch.argmax(
            ops.box_iou(target_boxes, anchor_boxes), 1
        ).tolist()
        h = (b_y / scaler).long().squeeze_().tolist()
        w = (b_x / scaler).long().squeeze_().tolist()
        idx = (h, w, bst_anchor_idx)

        iou = ops.box_iou(boxes, pred_box.view(-1, 4))
        obj = torch.where(iou > iou_threshold, -1, 0).amin(0)
        pos_obj = torch.zeros(pred_box.shape[:3])
        pos_obj[idx] = 1
        pos_obj_idx = pos_obj.view(-1).nonzero()
        obj[pos_obj_idx] = 1

        cls = torch.zeros((label.size(0), num_classes)).scatter_(1, label.long(), 1)

        result.append((boxes, obj.float(), cls.float(), idx))
    return result


def get_loss(
    pred_result,
    target_result,
    loc_weight=0.4,
    obj_weight=0.2,
    cls_weight=0.4,
):
    loss = torch.tensor(0.0)
    for target, pred in zip(target_result, pred_result):
        target_boxes, target_obj, target_cls, target_idx = target
        pred_boxes = pred[target_idx][:, :4]
        pred_cls = pred[target_idx][:, 5:]
        pred_obj = pred[..., 4].view(-1)
        loss_loc = ops.complete_box_iou_loss(pred_boxes, target_boxes, reduction="mean")
        mask = target_obj >= 0
        loss_obj = binary_cross_entropy_with_logits(
            pred_obj[mask], target_obj[mask], reduction="mean"
        )
        loss_cls = binary_cross_entropy_with_logits(
            pred_cls, target_cls, reduction="mean"
        )
        loss += loss_loc * loc_weight + loss_obj * obj_weight + loss_cls * cls_weight
    return loss / len(target_result)


def postprocess_result(predictions, conf_thres=0.5, iou_thres=0.5):
    result = []
    for prediction in predictions:
        loc, obj, probs = prediction.split((4, 1, prediction.size(-1) - 5), -1)
        prob, cls = torch.max(probs, -1)
        conf = obj.squeeze() * prob
        filter = conf > conf_thres
        loc = loc[filter]
        if not loc.size(0):
            continue

        conf = conf[filter]
        cls = cls[filter]
        keep = ops.batched_nms(loc, conf, cls, iou_thres)
        result.append((loc[keep], conf[keep], cls[keep]))
    return result


def test():
    anchors = torch.tensor(
        (373, 326, 156, 198, 116, 90, 59, 119, 62, 45, 30, 61, 33, 23, 16, 30, 10, 13),
        dtype=torch.float,
    ).view((3, 3, 2))
    x0 = torch.rand((2, 75, 13, 13))
    x1 = torch.rand((2, 75, 26, 26))
    x2 = torch.rand((2, 75, 52, 52))
    targets = torch.tensor(
        (
            ((10, 10, 60, 60, 0), (30, 20, 180, 190, 1)),
            ((20, 20, 70, 90, 1), (-1, -1, -1, -1, -1)),
        ),
        dtype=torch.float,
    )
    pred, loss = decode_predictions(anchors, (x0, x1, x2), targets)
    output = postprocess_result(pred)
    print(loss)
    for i, batch in enumerate(output):
        for j, each in enumerate(batch):
            print(i, j, each.shape)


if __name__ == "__main__":
    test()
