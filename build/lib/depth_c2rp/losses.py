import torch
from torch import nn, Tensor
from torch.nn import functional as F
from depth_c2rp.utils.utils import batch_quaternion_matrix, compute_concat_loss
from depth_c2rp.DifferentiableRenderer.Kaolin.Renderer import DiffPFDepthRenderer

class CrossEntropy(nn.Module):
    def __init__(self, weight: Tensor = None, aux_weights: list = [1, 0.4, 0.4]) -> None:
        super().__init__()
        self.aux_weights = aux_weights
        self.criterion = nn.CrossEntropyLoss(weight=weight)

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        return self.criterion(preds, labels)

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label: int = 255, weight: Tensor = None, thresh: float = 0.7, aux_weights: list = [1, 1]) -> None:
        super().__init__()
        self.ignore_label = ignore_label
        self.aux_weights = aux_weights
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction='none')

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        n_min = labels[labels != self.ignore_label].numel() // 16
        loss = self.criterion(preds, labels).view(-1)
        loss_hard = loss[loss > self.thresh]

        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)

        return torch.mean(loss_hard)

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)


class Dice(nn.Module):
    def __init__(self, delta: float = 0.5, aux_weights: list = [1, 0.4, 0.4]):
        """
        delta: Controls weight given to FP and FN. This equals to dice score when delta=0.5
        """
        super().__init__()
        self.delta = delta
        self.aux_weights = aux_weights

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        num_classes = preds.shape[1]
        labels = F.one_hot(labels, num_classes).permute(0, 3, 1, 2)
        tp = torch.sum(labels*preds, dim=(2, 3))
        fn = torch.sum(labels*(1-preds), dim=(2, 3))
        fp = torch.sum((1-labels)*preds, dim=(2, 3))

        dice_score = (tp + 1e-6) / (tp + self.delta * fn + (1 - self.delta) * fp + 1e-6)
        dice_score = torch.sum(1 - dice_score, dim=-1)

        dice_score = dice_score / num_classes
        return dice_score.mean()

    def forward(self, preds, targets: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, targets) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, targets)


class Calculate_Loss(nn.Module):
    def __init__(self, loss_fn_names, weights=[1, 1, 1, 1],is_res=False):
        """
        Expect Loss Function of Mask, RT head, Joints 3D, Joints Pos
        """
        super().__init__()
        self.mask_loss = eval(loss_fn_names["masks"])()
        self.joints_3d_loss = nn.L1Loss()
        self.joints_pos_loss = nn.L1Loss()
        self.rt_loss = nn.MSELoss()
        self.weights = weights
        self.check_loss = nn.MSELoss()
        self.is_res = is_res
    
    def forward(self, batch_dt_trans, batch_dt_quaternion, batch_dt_joints_pos, batch_dt_masks, \
                batch_gt_masks, batch_gt_joints_pos, batch_gt_joints_wrt_cam, batch_gt_joints_wrt_rob, batch_xyz_rp=0.0):
        # batch_dt_trans (B, 3) batch_dt_quaternion (B, 4), batch_dt_joints_pos (B, N, 1), batch_dt_joints_wrt_cam (B, N, 3)
        # batch_dt_masks (B, num_classes, H, W), batch_gt_masks (B, H, W), batch_gt_joints_pos (B, N, 1)
        # batch_gt_joints_wrt_cam (B, N, 3), batch_gt_joints_wrt_rob (B, N, 3)
        # batch_xyz : (B, 1, 3) , both can be directly added to tensors
        
        mask_coeff, joints_3d_coeff, joints_pos_coeff, rt_pos_coeff = self.weights
        batch_joint_pos_loss = self.joints_pos_loss(batch_dt_joints_pos, batch_gt_joints_pos)
        
        # print("batch_dt_masks.shape", batch_dt_masks.shape)
        # print("batch_gt_masks.shape", batch_gt_masks.shape)
        
        batch_mask_loss = self.mask_loss(batch_dt_masks, batch_gt_masks) # (B, H, W)
        
        
        #print("norm between 3Ds", torch.mean(torch.norm(batch_dt_joints_wrt_cam - batch_gt_joints_wrt_cam, dim=-1)))
        #print("L1 loss", self.check_loss(batch_dt_joints_wrt_cam, batch_gt_joints_wrt_cam))
        # 对batch_dt_quaternion先归一化
        
        batch_dt_quaternion_norm = batch_dt_quaternion / torch.norm(batch_dt_quaternion, dim=-1,keepdim=True)
#        batch_gt_quaternion_norm = batch_gt_quaternion
#        batch_gt_rot = batch_quaternion_matrix(batch_gt_quaternion_norm.T)
        
        batch_rot = batch_quaternion_matrix(batch_dt_quaternion_norm.T) # (B, 3, 3)
        batch_res = torch.bmm(batch_rot, batch_gt_joints_wrt_rob.permute(0, 2, 1).contiguous()) # (B, 3, N)
        batch_res = batch_res + batch_dt_trans[:, :, None]
        batch_res = batch_res.permute(0, 2, 1).contiguous() # (B, N, 3)
        # batch_rt_loss = self.rt_loss(batch_res, batch_dt_joints_wrt_cam)
        batch_rt_loss = self.rt_loss(batch_res+batch_xyz_rp, batch_gt_joints_wrt_cam)
        # batch_rt_loss = self.rt_loss(batch_res, batch_gt_joints_wrt_cam)
        #print("batch_rot.device", batch_rot.device)
        batch_compute_joints_wrt_cam = compute_concat_loss(batch_rot, batch_dt_trans[:, :, None], batch_dt_joints_pos, batch_rot.device)
        #batch_compute_joints_wrt_cam = compute_concat_loss(batch_gt_rot, batch_gt_trans[:, :, None], batch_gt_joints_pos, batch_gt_rot.device)
        #print("batch_compute_joints_wrt_cam", batch_compute_joints_wrt_cam)
        #print("batch_gt_joints_wrt_cam", batch_gt_joints_wrt_cam)
        
        if not self.is_res:
            # batch_joint_3d_loss = self.joints_3d_loss(batch_dt_joints_wrt_cam+batch_xyz_rp, batch_gt_joints_wrt_cam)
            batch_joint_3d_loss = self.joints_3d_loss(batch_compute_joints_wrt_cam, batch_gt_joints_wrt_cam)
#        else:
#            d = batch_dt_trans[:, None, :][:,:, 2] + batch_xyz_rp[:, :, 2]
#            batch_dt_joints_wrt_cam[:, :, 2] += d
##            print("shape1", batch_xyz_rp[:, :, :2].shape)
##            print("shape2", batch_xyz_rp[:, :, 2].shape)
##            print("shape3", batch_dt_joints_wrt_cam[:, :, :2].shape)
##            print("d.shape", d.shape)
#            batch_dt_joints_wrt_cam[:, :, :2] = (batch_dt_joints_wrt_cam[:, :, :2] + batch_xyz_rp[:, :, :2] / batch_xyz_rp[:, :, 2][:, :, None]) * d[:, :, None]
#            batch_joint_3d_loss = self.joints_3d_loss(batch_dt_joints_wrt_cam, batch_gt_joints_wrt_cam)  
        
        
        
        losses = {}
        losses["mask_loss"] = mask_coeff * batch_mask_loss
        losses["joint_3d_loss"] = joints_3d_coeff * batch_joint_3d_loss
        losses["joint_pos_loss"] = joints_pos_coeff * batch_joint_pos_loss
        losses["rt_loss"] = rt_pos_coeff * batch_rt_loss
        losses["total_loss"] = mask_coeff * batch_mask_loss + joints_3d_coeff * batch_joint_3d_loss + \
                               joints_pos_coeff * batch_joint_pos_loss + rt_pos_coeff * batch_rt_loss
        return losses


__all__ = ['CrossEntropy', 'OhemCrossEntropy', 'Dice']


def get_loss(loss_fn_name: str = 'CrossEntropy', ignore_label: int = 255, cls_weights: Tensor = None):
    assert loss_fn_name in __all__, f"Unavailable loss function name >> {loss_fn_name}.\nAvailable loss functions: {__all__}"
    if loss_fn_name == 'Dice':
        return Dice()
    return eval(loss_fn_name)(ignore_label, cls_weights)


if __name__ == '__main__':
    pred = torch.randint(0, 19, (2, 19, 480, 640), dtype=torch.float)
    label = torch.randint(0, 19, (2, 480, 640), dtype=torch.long)
    loss_fn = Dice()
    y = loss_fn(pred, label)
    print(y)