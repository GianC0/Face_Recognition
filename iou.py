import torch


def intersection_over_union(boxes_preds, boxes_labels):
    box1_x1 = boxes_preds[0]
    box1_y1 = boxes_preds[1]
    box1_x2 = boxes_preds[2]
    box1_y2 = boxes_preds[3]
    box2_x1 = boxes_labels[0]
    box2_y1 = boxes_labels[1]
    box2_x2 = boxes_labels[2]
    box2_y2 = boxes_labels[3]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # Need clamp(0) in case they do not intersect, then we want intersection to be 0
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0) + 1e-6
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    # box1_proportion = box1_area/intersection
    # box2_proportion = box2_area/intersection
    # if box1_proportion > 0.9 or box2_proportion > 0.9:
    #     return 1.0
    box1_proportion = intersection / box1_area
    box2_proportion = intersection / box2_area
    if (box1_proportion > 0.9 or box2_proportion > 0.9) and box1_area > box2_area:
        return 1.0

    return intersection / (box1_area + box2_area - intersection + 1e-6)