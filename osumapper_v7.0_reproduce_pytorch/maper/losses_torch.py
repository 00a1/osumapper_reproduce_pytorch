# Part 7 Loss Functions

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

def inblock_loss(vg, border, value):
    wall_var_l = torch.where(vg < border, (value - vg)**2, torch.zeros_like(vg, device=device))
    wall_var_r = torch.where(vg > 1 - border, (vg - (1 - value))**2, torch.zeros_like(vg, device=device))
    return torch.mean(wall_var_l + wall_var_r)

def GenerativeCustomLoss(y_pred):
    loss1 = 1 - torch.mean(y_pred)
    return loss1

def BoxCustomLoss(loss_border, loss_value, map_part):
    return inblock_loss(map_part[:, :, 0:2], loss_border, loss_value) + inblock_loss(map_part[:, :, 4:6], loss_border, loss_value)

def AlwaysZeroCustomLoss():
    return torch.tensor(0.0, dtype=torch.float32, device=device)