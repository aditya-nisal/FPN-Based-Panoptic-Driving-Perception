import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from bdd_dataset import BDDDataset
from yolo_p import YOLOPHead

# Define the YOLOP loss function
class YOLOPLoss(nn.Module):
    def __init__(self, num_classes, anchors):
        super(YOLOPLoss, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, output, target):
        # output: (batch_size, num_anchors * (5 + num_classes), grid_size, grid_size)
        # target: (batch_size, num_boxes, 6)
        nB = output.size(0)
        nGh = output.size(2)
        nGw = output.size(3)
        nA = self.num_anchors
        nC = self.num_classes

        # reshape the output tensor
        output = output.view(nB, nA, 5 + nC, nGh, nGw)
        output = output.permute(0, 1, 3, 4, 2).contiguous()

        # get the x, y, w, h, and objectness predictions
        pred_txtytwth = output[:, :, :, :, :4]
        pred_obj = output[:, :, :, :, 4].unsqueeze(-1)
        pred_cls = output[:, :, :, :, 5:]

        # get the target values
        target_mask = target[:, :, 0].bool().unsqueeze(-1)
        target_txtytwth = target[:, :, 1:5]
        target_obj = target[:, :, 0].unsqueeze(-1)
        target_cls = target[:, :, 5:]

        # calculate the matching between anchors and ground truth boxes
        iou_scores = []
        for i in range(nB):
            iou_anchor = []
            for j in range(nA):
                iou_box = bbox_iou(pred_txtytwth[i,j,:,:,:].view(-1,4), target_txtytwth[i,:,:])
                iou_anchor.append(iou_box)
            iou_anchor = torch.stack(iou_anchor)
            iou_scores.append(iou_anchor)
        iou_scores = torch.stack(iou_scores)
        best_ious, best_n = iou_scores.max(1)
        best_ious = best_ious.unsqueeze(-1)
        best_n = best_n.unsqueeze(-1)

                # calculate the loss for objectness
        obj_mask = (best_ious > 0.5) & target_mask
        noobj_mask = (best_ious < 0.4) & ~obj_mask
        obj_loss = self.bce_loss(pred_obj[obj_mask], target_obj[obj_mask])
        noobj_loss = self.bce_loss(pred_obj[noobj_mask], target_obj[noobj_mask])
        obj_loss = obj_loss.mean()
        noobj_loss = noobj_loss.mean()
        obj_weight = obj_mask.float().mean()
        noobj_weight = noobj_mask.float().mean()

        # calculate the loss for classification
        cls_mask = obj_mask.expand(-1, -1, -1, nC)
        cls_loss = self.bce_loss(pred_cls[cls_mask], target_cls[cls_mask])
        cls_loss = cls_loss.mean()
        cls_weight = obj_mask.float().mean()

        # calculate the loss for bounding box regression
        txty_loss = self.bce_loss(pred_txty[bbox_mask], target_txty[bbox_mask])
        txty_loss = txty_loss.mean()
        txty_weight = obj_mask.float().mean()

        twth_loss = self.mse_loss(pred_twth[bbox_mask], target_twth[bbox_mask])
        twth_loss = twth_loss.mean()
        twth_weight = obj_mask.float().mean()

        # compute the final loss as a weighted sum of the individual losses
        loss = (obj_weight * obj_loss +
                noobj_weight * noobj_loss +
                cls_weight * cls_loss +
                txty_weight * txty_loss +
                twth_weight * twth_loss)

        return loss

