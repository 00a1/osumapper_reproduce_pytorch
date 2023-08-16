import numpy as np
import torch
g_input_size = 50
g_batch = 50
note_group_size = 10
ginput_noise = np.random.random((g_batch, g_input_size))
glabel = [np.zeros((g_batch, note_group_size * 4)), np.ones((g_batch,)), np.ones((g_batch,))]
print(ginput_noise)
print(glabel)
print("--------------------------------")
tgnoise = torch.rand(g_batch, g_input_size)
tglabel = [torch.zeros((g_batch, note_group_size * 4)), torch.ones((g_batch,)), torch.ones((g_batch,))]
print(tgnoise)
print(tglabel)