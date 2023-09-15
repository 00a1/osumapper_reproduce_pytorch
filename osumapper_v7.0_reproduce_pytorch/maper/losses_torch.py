# Part 7 Loss Functions

import torch
# import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

# def inblock_loss(vg, border, value):
#     wall_var_l = torch.where(vg < border, (value - vg)**2, torch.zeros_like(vg))
#     wall_var_r = torch.where(vg > 1 - border, (vg - (1 - value))**2, torch.zeros_like(vg))
#     return torch.mean(torch.mean(wall_var_l + wall_var_r, dim=2), dim=1)

# def inblock_loss(vg, border, value):
#     wall_var_l = torch.where(vg < border, (value - vg)**2, torch.zeros_like(vg))
#     wall_var_r = torch.where(vg > 1 - border, (vg - (1 - value))**2, torch.zeros_like(vg))
#     return torch.mean(torch.mean(wall_var_l + wall_var_r, dim=1), dim=1)  # Reduce along dimension 1

def inblock_loss(vg, border, value):
    wall_var_l = torch.where(vg < border, (value - vg)**2, torch.zeros_like(vg, device=device))
    wall_var_r = torch.where(vg > 1 - border, (vg - (1 - value))**2, torch.zeros_like(vg, device=device))
    return torch.mean(wall_var_l + wall_var_r)


# class GenerativeCustomLoss(nn.Module):
#     def __init__(self):
#         super(GenerativeCustomLoss, self).__init__()

#     # def forward(self, y_true, y_pred):
#     #     classification = y_pred
#     #     loss1 = 1 - torch.mean(classification, dim=1)
#     #     return loss1

#     def forward(self, y_true, y_pred):
#         classification = y_pred
#         if classification.dim() == 1:
#             classification = classification.unsqueeze(0)  # Convert to a 2D tensor if it's 1D
#         loss1 = 1 - torch.mean(classification, dim=1)
#         return loss1

def GenerativeCustomLoss(y_pred):
    classification = y_pred
    if classification.dim() == 1:
        classification = classification.unsqueeze(0)  # Convert to a 2D tensor if it's 1D
    loss1 = 1 - torch.mean(classification, dim=1)
    return loss1

# class BoxCustomLoss(nn.Module):
#     def __init__(self, border, value):
#         super(BoxCustomLoss, self).__init__()
#         self.loss_border = border
#         self.loss_value = value

#     def forward(self, y_true, y_pred):
#         map_part = y_pred
#         #print("Shape of map_part:", map_part.shape) # test if shape is 50
#         return inblock_loss(map_part[0:2], self.loss_border, self.loss_value) + inblock_loss(map_part[4:6], self.loss_border, self.loss_value)
#         #return inblock_loss(map_part[:, :, 0:2], self.loss_border, self.loss_value) + inblock_loss(map_part[:, :, 4:6], self.loss_border, self.loss_value)

def BoxCustomLoss(loss_border, loss_value, y_pred):
    map_part = y_pred
    return inblock_loss(map_part[0:2], loss_border, loss_value) + inblock_loss(map_part[4:6], loss_border, loss_value)

# class AlwaysZeroCustomLoss(nn.Module):
#     def __init__(self):
#         super(AlwaysZeroCustomLoss, self).__init__()

#     def forward(self, y_true, y_pred):
#         return torch.tensor(0.0, dtype=torch.float32, device=device)


def AlwaysZeroCustomLoss():
    return torch.tensor(0.0, dtype=torch.float32, device=device)