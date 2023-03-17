from torch import nn
from torch.optim import AdamW, SGD


def get_optimizer(model: nn.Module, optimizer: str, lr: float, weight_decay: float = 0.01):
    wd_params, nwd_params = [], []
    for p in model.parameters():
        if p.dim() == 1:
            nwd_params.append(p)
        else:
            wd_params.append(p)
    
    params = [
        {"params": wd_params},
        {"params": nwd_params, "weight_decay": 0}
    ]

    if optimizer == 'adamw':
        return AdamW(params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    else:
        return SGD(params, lr, momentum=0.9, weight_decay=weight_decay)

def adapt_lr(optimizer, epoch_num, batch_idx, iter_per_epoch, base_lr, max_iters): 
    cur_iters = (epoch_num - 1) * iter_per_epoch + batch_idx
    warmup_iters = 3000
    warmup_ratio = 1e-06
    # print("self.max_iters", self.max_iters)
    # print('base_lr', self.base_lr)
    # print('all_epochs', self.total_epoch_nums)
    if cur_iters <= warmup_iters:
        k = (1 - cur_iters / warmup_iters) * (1 - warmup_ratio)
        lr_ = base_lr * (1 - k)
    else:
        lr_ = base_lr * (1.0 - (cur_iters - 1) / max_iters) ** 1.0
    
    for param_group in optimizer.param_groups:
        # print("past learning rate", param_group["lr"])
        param_group['lr'] = lr_
    return lr_
        # print("current learning rate", param_group["lr"])
 