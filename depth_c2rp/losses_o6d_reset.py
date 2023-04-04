import torch
from torch import nn, Tensor
from torch.nn import functional as F
from depth_c2rp.utils.utils import batch_quaternion_matrix, compute_concat_loss, compute_rotation_matrix_from_ortho6d, get_K_crop_resize, update_translation, compute_distengle_loss, get_gt_vxvyvz, compute_2nplus2_loss
from depth_c2rp.DifferentiableRenderer.Kaolin.Renderer import DiffPFDepthRenderer 
from depth_c2rp.utils.utils import get_meshes_bounding_boxes, get_meshes_center, compute_DX_loss
import time
import numpy as np

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
    def __init__(self, loss_fn_names, weights=[1, 1, 1, 1],is_res=False,device=None, cfg=None, rt_loss_type="entangle"):
        """
        Expect Loss Function of Mask, RT head, Joints 3D, Joints Pos
        """
        super().__init__()
        self.mask_loss = eval(loss_fn_names["masks"])()
        self.joints_3d_loss = nn.L1Loss()
        self.joints_pos_loss = nn.L1Loss()
        self.rt_loss = nn.L1Loss()
        self.weights = weights
        self.check_loss = nn.MSELoss()
        self.is_res = is_res
        self.device = device
        #self.t_init = torch.tensor([[0.0, 0.0, 1.0]]).to(self.device)
        self.cfg = cfg
        self.three_d_loss_type=self.cfg["LOSS"]["THREE_D_LOSS_TYPE"]
        self.rt_loss_type=rt_loss_type
        if cfg is not None and device is not None:
            self.dr_engine = self.cfg["DR"]["ENGINE"]
            if self.dr_engine == "Kaolin":
                self.DPRenderer = DiffPFDepthRenderer(self.cfg, self.device)
                self.DPRenderer.load_mesh()
                self.num_pts = self.cfg["DR"]["NUM_OF_SAMPLE_PTS"]
                
    
    def forward(self, batch_dt_poses, batch_dt_joints_pos, batch_dt_masks, \
                batch_gt_masks, batch_gt_joints_pos, batch_gt_trans, batch_gt_quaternion, batch_gt_joints_wrt_cam,\
                batch_gt_K, batch_gt_boxes, batch_gt_2nplus2_wrt_cam=None):
        # batch_dt_poses (B, 9), batch_dt_joints_pos (B, N, 1), batch_dt_joints_wrt_cam (B, N, 3)
        # batch_dt_masks (B, num_classes, H, W), batch_gt_masks (B, H, W), batch_gt_joints_pos (B, N, 1)
        # batch_gt_joints_wrt_cam (B, N, 3), batch_gt_joints_wrt_rob (B, N, 3)
        # batch_xyz : (B, 1, 3) , both can be directly added to tensors
        
        mask_coeff, joints_3d_coeff, joints_pos_coeff, rt_pos_coeff = self.weights
        B, _ = batch_dt_poses.shape
        
        # joints angles loss
        batch_joint_pos_loss = self.joints_pos_loss(batch_dt_joints_pos, batch_gt_joints_pos)
        
        # mask loss
        batch_mask_loss = self.mask_loss(batch_dt_masks, batch_gt_masks) # (B, H, W)

        assert batch_dt_poses.shape[-1] == 9
        batch_dt_rot = compute_rotation_matrix_from_ortho6d(batch_dt_poses[..., :6]) # (B, 3, 3)
        
        # batch_gt_rot
        batch_gt_rot = batch_quaternion_matrix(batch_gt_quaternion.T)
        
        # sample batch_sample_gt_wrt_rob
        if self.device is not None and self.cfg is not None and self.dr_engine == "Kaolin":
            #self.DPRenderer.load_joints(batch_dt_joints_pos) big sb!!!
            index = self.DPRenderer.get_sample_index(self.num_pts)
            batch_sample_gt_wrt_rob = self.DPRenderer.get_sample_meshes(batch_gt_joints_pos, index) # B x num_pts x 3
            batch_sample_dt_wrt_rob = self.DPRenderer.get_sample_meshes(batch_dt_joints_pos, index) # B x num_pts x 3
        
        # batch_dt_trans
        #t1 = time.time()
        batch_gt_crop_resize = self.cfg["DATASET"]["INPUT_RESOLUTION"]
        batch_new_gt_K = get_K_crop_resize(batch_gt_K, batch_gt_boxes, batch_gt_crop_resize)
        batch_t_init = get_meshes_center(batch_sample_gt_wrt_rob) # B x 3
        batch_t_init = (torch.bmm(batch_gt_rot, batch_t_init[:, :, None]) + batch_gt_trans[:, :, None]).squeeze(2)
        batch_dt_trans = update_translation(batch_dt_poses[..., 6:], batch_new_gt_K, batch_t_init)

        if self.rt_loss_type == "entangle":
            batch_dt_res = torch.bmm(batch_dt_rot, batch_sample_gt_wrt_rob.permute(0, 2, 1).contiguous()) + batch_dt_trans[:, :, None] # (B, 3, num_pts) 
            batch_gt_res = torch.bmm(batch_gt_rot, batch_sample_gt_wrt_rob.permute(0, 2, 1).contiguous()) + batch_gt_trans[:, :, None] # (B, 3, num_pts) 
            batch_rt_loss = self.rt_loss(batch_dt_res, batch_gt_res)
        elif self.rt_loss_type == "disentangle":
            #print("!!!!!!!!!disentangle!!!!")
            
            batch_vxvyvz_gt = get_gt_vxvyvz(batch_t_init, batch_gt_trans, batch_new_gt_K)     
            batch_gt_res = torch.bmm(batch_gt_rot, batch_sample_gt_wrt_rob.permute(0, 2, 1).contiguous()) + batch_gt_trans[:, :, None] # (B, 3, num_pts) 
            batch_rt_loss = compute_distengle_loss(batch_dt_poses[..., 6:], batch_vxvyvz_gt, batch_new_gt_K, batch_t_init, 
                            batch_dt_rot, batch_gt_rot, batch_sample_gt_wrt_rob, batch_gt_res, l1_or_l2="l1")
        else:
            raise ValueError

        
        if self.three_d_loss_type == "few":
            batch_dt_2nplus2_wrt_cam = compute_2nplus2_loss(batch_dt_rot, batch_dt_trans[:, :, None], batch_dt_joints_pos, batch_dt_rot.device)
            batch_joint_3d_loss = self.joints_3d_loss(batch_dt_2nplus2_wrt_cam, batch_gt_2nplus2_wrt_cam)
        elif self.three_d_loss_type == "many":
            batch_joint_3d_loss = self.joints_3d_loss(batch_sample_gt_wrt_rob, batch_sample_dt_wrt_rob)
        elif self.three_d_loss_type == "edm":
            #print("edm!!")
            batch_dt_2nplus2_wrt_cam = compute_2nplus2_loss(batch_gt_rot, batch_gt_trans[:, :, None], batch_dt_joints_pos, batch_gt_rot.device)
            batch_joint_3d_loss = compute_DX_loss(batch_dt_2nplus2_wrt_cam, batch_gt_2nplus2_wrt_cam)
        elif self.three_d_loss_type == "many_dt":
            batch_dt_res = torch.bmm(batch_dt_rot, batch_sample_dt_wrt_rob.permute(0, 2, 1).contiguous()) + batch_dt_trans[:, :, None] # (B, 3, num_pts) 
            batch_joint_3d_loss = self.joints_3d_loss(batch_gt_res, batch_dt_res)
        else:
            raise ValueError


        
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