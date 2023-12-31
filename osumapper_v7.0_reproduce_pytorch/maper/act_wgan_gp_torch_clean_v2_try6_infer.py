# Part 7 action script

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from rhythm_loader import read_map_predictions
from losses_torch import BoxCustomLoss
from plot_tools import MyLine
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

import warnings
warnings.filterwarnings("ignore")

GAN_PARAMS = {
    "verbose" : False,
    "divisor" : 4,
    "good_epoch" : 12,
    "max_epoch" : 30,
    "note_group_size" : 10,
    "g_epochs" : 1,
    "c_epochs" : 1,
    "g_batch" : 50,
    "g_input_size" : 50,
    "c_true_batch" : 140,
    "c_false_batch" : 5,
    "c_randfalse_batch" : 5,
    "note_distance_basis" : 200,
    "next_from_slider_end" : False,
    "max_ticks_for_ds" : 1,
    "box_loss_border" : 0.1,
    "box_loss_value" : 0.4,
    "box_loss_weight" : 1
}

def step6_set_gan_params_v2(params):
    """
    Basically Object.assign(GAN_PARAMS, params)
    See stackoverflow 38987
    """
    global GAN_PARAMS
    GAN_PARAMS = {**GAN_PARAMS, **params}

def inblock_trueness(vg):
    """
    Despite the weird name, it checks if all notes and slider tails are within the map boundaries.
    """
    wall_var_l = torch.tensor(vg < 0, dtype=torch.float32, device=device).clone().detach()
    wall_var_r = torch.tensor(vg > 1, dtype=torch.float32, device=device).clone().detach()
    return torch.mean(torch.mean(wall_var_l + wall_var_r, dim=2), dim=1)

def construct_map_with_sliders(var_tensor, extvar=[]):
    """
    The biggest function here. It takes a tensor with random number as input, with extra variables in extvar
    (for extvar see the KerasCustomMappingLayer class)
    var_tensor shape is (batch_size(None), 4 * note_group_size)
        the first dimension is "None", or "?" if you print the shape. It is filled with batch_size in training time.
    output shape is (batch_size(None), note_group_size, 6)
        where each last dimension is (x_start, y_start, x_vector, y_vector, x_end, y_end), all mapped to [-1,1] range
        the vector in the middle is supposed to be the direction of cursor after hitting the note

    The reason this function is this big is that TensorFlow rewrites functions used in the training loop,
    which includes this one as a "mapping layer". It was amazingly done, but still I have run into troubles
    with the rewriting many times. That was the reason I didn't dare to reduce it into smaller functions.

    You might notice I didn't use np calls in this function at all. Yes, it will cause problems.
    Everything needs to be converted to tf calls instead. Take it in mind if you're editing it.
    """
    var_tensor = var_tensor.to(torch.float32)
    var_shape = var_tensor.shape
    wall_l = 0.15
    wall_r = 0.85
    x_max = 512
    y_max = 384

    # Should be equal to note_group_size
    half_tensor = var_shape[1]//4

    # length multiplier
    if "length_multiplier" in extvar:
        length_multiplier = extvar["length_multiplier"]
    else:
        length_multiplier = 1

    # notedists
    if "begin" in extvar:
        begin_offset = extvar["begin"]
    else:
        begin_offset = 0

    # Load external arrays as tensors
    relevant_tensors = extvar["relevant_tensors"]
    relevant_is_slider = relevant_tensors["is_slider"]
    relevant_slider_lengths = relevant_tensors["slider_lengths"]
    relevant_slider_types = relevant_tensors["slider_types"]
    relevant_slider_cos = relevant_tensors["slider_cos_each"]
    relevant_slider_sin = relevant_tensors["slider_sin_each"]
    relevant_note_distances = relevant_tensors["note_distances"]
    note_distances_now = length_multiplier * torch.unsqueeze(relevant_note_distances, dim=0)

    # init
    # l = torch.tensor(note_distances_now, dtype=torch.float32, device=device)
    l = note_distances_now.clone().detach().to(device, dtype=torch.float32)

    cos_list = var_tensor[:, 0:half_tensor * 2]
    sin_list = var_tensor[:, half_tensor * 2:]
    len_list = torch.sqrt(torch.square(cos_list) + torch.square(sin_list))
    cos_list = cos_list / len_list
    sin_list = sin_list / len_list

    wall_l = 0.05 * x_max + l * 0.5
    wall_r = 0.95 * x_max - l * 0.5
    wall_t = 0.05 * y_max + l * 0.5
    wall_b = 0.95 * y_max - l * 0.5

    tick_diff = extvar["tick_diff"]

    # max_ticks_for_ds is an int variable, converted to float to avoid potential type error
    # use_ds = torch.unsqueeze(torch.tensor(tick_diff <= extvar["max_ticks_for_ds"], dtype=torch.float32, device=device), dim=0)
    use_ds = torch.unsqueeze(torch.tensor(tick_diff <= extvar["max_ticks_for_ds"], dtype=torch.float32, device=device).clone().detach(), dim=0)

    # rerand = not use distance snap
    rerand = 1 - use_ds
    not_rerand = use_ds

    next_from_slider_end = extvar["next_from_slider_end"]

    if "start_pos" in extvar:
        _pre_px = extvar["start_pos"][0]
        _pre_py = extvar["start_pos"][1]
        _px = torch.tensor(_pre_px, dtype=torch.float32, device=device).clone().detach()
        _py = torch.tensor(_pre_py, dtype=torch.float32, device=device).clone().detach()
    else:
        _px = torch.tensor(256, dtype=torch.float32, device=device)
        _py = torch.tensor(192, dtype=torch.float32, device=device)

    # this is not important since the first position starts at _ppos + Δpos
    _x = torch.tensor(256, dtype=torch.float32, device=device)
    _y = torch.tensor(192, dtype=torch.float32, device=device)

    # Use a buffer to save output
    outputs = []

    for k in range(half_tensor):
        # r_max = 192, r = 192 * k, theta = k * 10
        rerand_x = 256 + 256 * var_tensor[:, k]
        rerand_y = 192 + 192 * var_tensor[:, k + half_tensor*2]

        # Distance snap start
        # If the starting point is close to the wall, use abs() to make sure it doesn't go outside the boundaries
        delta_value_x = l[:, k] * cos_list[:, k]
        delta_value_y = l[:, k] * sin_list[:, k]

        # It is tensor calculation batched 8~32 each call, so if/else do not work here.
        wall_value_l = torch.tensor(_px < wall_l[:, k], dtype=torch.float32, device=device)
        wall_value_r = torch.tensor(_px > wall_r[:, k], dtype=torch.float32, device=device)
        wall_value_xmid = (torch.tensor(_px > wall_l[:, k], dtype=torch.float32, device=device) * torch.tensor(_px < wall_r[:, k], dtype=torch.float32, device=device))
        wall_value_t = torch.tensor(_py < wall_t[:, k], dtype=torch.float32, device=device)
        wall_value_b = torch.tensor(_py > wall_b[:, k], dtype=torch.float32, device=device)
        wall_value_ymid = (torch.tensor(_py > wall_t[:, k], dtype=torch.float32, device=device) * torch.tensor(_py < wall_b[:, k], dtype=torch.float32, device=device))

        x_delta = torch.abs(delta_value_x) * wall_value_l - torch.abs(delta_value_x) * wall_value_r + delta_value_x * wall_value_xmid
        y_delta = torch.abs(delta_value_y) * wall_value_t - torch.abs(delta_value_y) * wall_value_b + delta_value_y * wall_value_ymid

        # rerand_* if not using distance snap, (_p* + *_delta) if using distance snap
        _x = rerand[:, k] * rerand_x + not_rerand[:, k] * (_px + x_delta)
        _y = rerand[:, k] * rerand_y + not_rerand[:, k] * (_py + y_delta)
        # Distance snap end

        # calculate output vector

        # slider part
        sln = relevant_slider_lengths[k]
        slider_type = relevant_slider_types[k]
        scos = relevant_slider_cos[k]
        ssin = relevant_slider_sin[k]
        _a = cos_list[:, k + half_tensor]
        _b = sin_list[:, k + half_tensor]

        # cos(a+θ) = cosa cosθ - sina sinθ
        # sin(a+θ) = cosa sinθ + sina cosθ
        _oa = _a * scos - _b * ssin
        _ob = _a * ssin + _b * scos

        cp_slider = torch.stack([_x / x_max, _y / y_max, _oa, _ob, (_x + _a * sln) / x_max, (_y + _b * sln) / y_max]).T
        _px_slider = torch.where(next_from_slider_end, _x + _a * sln, _x)
        _py_slider = torch.where(next_from_slider_end, _y + _b * sln, _y)

        # circle part
        _a = rerand[:, k] * cos_list[:, k + half_tensor] + not_rerand[:, k] * cos_list[:, k]
        _b = rerand[:, k] * sin_list[:, k + half_tensor] + not_rerand[:, k] * sin_list[:, k]

        cp_circle = torch.stack([_x / x_max, _y / y_max, _a, _b, _x / x_max, _y / y_max]).T
        _px_circle = _x
        _py_circle = _y

        # Outputs are scaled to [0,1] region
        output_value = torch.where(relevant_is_slider[k], cp_slider, cp_circle)
        outputs.append(output_value)

        # Set starting point for the next circle/slider
        _px = torch.where(relevant_is_slider[k], _px_slider, _px_circle)
        _py = torch.where(relevant_is_slider[k], _py_slider, _py_circle)

    return torch.stack(outputs).permute(1, 0, 2)

class PyTorchCustomMappingLayer(nn.Module):
    def __init__(self, extvar, note_group_size=10):
        super(PyTorchCustomMappingLayer, self).__init__()
        self.extvar = extvar
        self.extvar_begin = nn.Parameter(torch.tensor(extvar["begin"], dtype=torch.int32, device=device), requires_grad=False)
        self.extvar_lmul = nn.Parameter(torch.tensor([extvar["length_multiplier"]], dtype=torch.float32, device=device), requires_grad=False)
        self.extvar_nfse = nn.Parameter(torch.tensor(extvar["next_from_slider_end"], dtype=torch.bool, device=device), requires_grad=False)
        self.extvar_mtfd = nn.Parameter(torch.tensor(GAN_PARAMS["max_ticks_for_ds"], dtype=torch.float32, device=device), requires_grad=False)
        self.note_group_size = GAN_PARAMS["note_group_size"]

        self.extvar_spos = nn.Parameter(torch.zeros(2, dtype=torch.float32, device=device), requires_grad=False)
        self.extvar_rel = nn.Parameter(torch.zeros(6, note_group_size, dtype=torch.float32, device=device), requires_grad=False)
        self.extvar_tickdiff = nn.Parameter(torch.zeros(note_group_size, dtype=torch.float32, device=device), requires_grad=False)

    def set_extvar(self, extvar):
        self.extvar = extvar

        begin_offset = extvar["begin"]
        relevant_tensors = {
            "is_slider": torch.tensor(is_slider[begin_offset: begin_offset + self.note_group_size], dtype=torch.bool, device=device),
            "slider_lengths": torch.tensor(slider_lengths[begin_offset: begin_offset + self.note_group_size], dtype=torch.float32, device=device),
            "slider_types": torch.tensor(slider_types[begin_offset: begin_offset + self.note_group_size], dtype=torch.float32, device=device),
            "slider_cos_each": torch.tensor(slider_cos_each[begin_offset: begin_offset + self.note_group_size], dtype=torch.float32, device=device),
            "slider_sin_each": torch.tensor(slider_sin_each[begin_offset: begin_offset + self.note_group_size], dtype=torch.float32, device=device),
            "note_distances": torch.tensor(note_distances[begin_offset: begin_offset + self.note_group_size], dtype=torch.float32, device=device),
        }
        self.extvar["relevant_tensors"] = relevant_tensors

        self.extvar_begin.data = torch.tensor(extvar["begin"], dtype=torch.int32, device=device)
        self.extvar_spos.data = torch.tensor(extvar["start_pos"], dtype=torch.float32, device=device)
        self.extvar_lmul.data = torch.tensor(extvar["length_multiplier"], dtype=torch.float32, device=device)# small fix
        self.extvar_nfse.data = torch.tensor(extvar["next_from_slider_end"], dtype=torch.bool, device=device)
        self.extvar_mtfd.data = torch.tensor(GAN_PARAMS["max_ticks_for_ds"], dtype=torch.float32, device=device)
        extvar_rel_data_numpy = np.array([
            is_slider[begin_offset: begin_offset + self.note_group_size],
            slider_lengths[begin_offset: begin_offset + self.note_group_size],
            slider_types[begin_offset: begin_offset + self.note_group_size],
            slider_cos_each[begin_offset: begin_offset + self.note_group_size],
            slider_sin_each[begin_offset: begin_offset + self.note_group_size],
            note_distances[begin_offset: begin_offset + self.note_group_size],
        ], dtype=np.float32)
        self.extvar_rel.data = torch.tensor(extvar_rel_data_numpy, dtype=torch.float32, device=device)
        self.extvar_tickdiff.data = torch.tensor(
            tick_diff[begin_offset: begin_offset + self.note_group_size],
            dtype=torch.float32,
            device=device,
        )

    def forward(self, inputs):
        mapvars = inputs
        start_pos = self.extvar_spos
        rel = self.extvar_rel
        extvar = {
            "begin": self.extvar_begin,
            "start_pos": start_pos,
            "length_multiplier": self.extvar_lmul,
            "next_from_slider_end": self.extvar_nfse,
            "tick_diff": self.extvar_tickdiff,
            "max_ticks_for_ds": self.extvar_mtfd,
            "relevant_tensors": {
                "is_slider": rel[0].bool(),
                "slider_lengths": rel[1],
                "slider_types": rel[2],
                "slider_cos_each": rel[3],
                "slider_sin_each": rel[4],
                "note_distances": rel[5],
            }
        }
        result = construct_map_with_sliders(mapvars, extvar=extvar)
        return result

def plot_current_map(mp, plot_int):
    """
    This is only used in debugging.
    """
    # to make it clearer, add the start pos
    # cat1 = torch.cat([torch.tensor(extvar["start_pos"], dtype=torch.float32, device=device) / torch.tensor([512, 384], dtype=torch.float32, device=device), torch.tensor([0, 0, 0, 0], dtype=torch.float32, device=device)])
    # print(cat1.size())
    # print(cat1)
    # npa = torch.cat([cat1.unsqueeze(0), mp.squeeze()]).detach().cpu().numpy()
    npa = mp * torch.tensor([512, 384, 0, 0, 0, 0], dtype=torch.float32, device=device)
    # print(npa)
    # npa = mp.detach().cpu()

    fig, ax = plt.subplots()
    plt.xlim(0, 512)
    plt.ylim(0, 384)
    x, y = np.transpose(npa.detach().cpu())[0:2]
    # x, y = np.random.rand(2, 20)
    line = MyLine(x, y, mfc='red', ms=12)
    line.text.set_color('red')
    line.text.set_fontsize(16)
    ax.add_line(line)
    # plt.show()
    plt.savefig(f'1000graph001{plot_int}.png')

def ran_plot():
    fig, ax = plt.subplots()
    # x, y = torch.transpose(npa)[0:2]
    x, y = np.random.rand(2, 20)
    line = MyLine(x, y, mfc='red', ms=12)
    line.text.set_color('red')
    line.text.set_fontsize(16)
    ax.add_line(line)
    plt.show()

# custom weights initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# class DeeperCNN(nn.Module):
#     def __init__(self, in_channels, out_params):
#         super(DeeperCNN, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm1d(32)
#         self.relu1 = nn.LeakyReLU(0.2)
#         self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        
#         self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm1d(64)
#         self.relu2 = nn.LeakyReLU(0.2)
#         self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        
#         self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm1d(128)
#         self.relu3 = nn.LeakyReLU(0.2)
#         self.maxpool3 = nn.MaxPool1d(kernel_size=2)
        
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(128, 64)
#         self.fc2 = nn.Linear(64, out_params)

#     def forward(self, x):
#         x = x.permute(0, 2, 1)
#         x = self.maxpool1(self.relu1(self.bn1(self.conv1(x))))
#         x = self.maxpool2(self.relu2(self.bn2(self.conv2(x))))
#         x = self.maxpool3(self.relu3(self.bn3(self.conv3(x))))
#         x = self.flatten(x)
#         x = torch.relu(self.fc1(x))
#         output = torch.tanh(self.fc2(x))
#         return output

nz = 6

class Generator(nn.Module):
    def __init__(self, out_params):
        super(Generator, self).__init__()
        # input is Z, going into a convolution
        self.conv1t = nn.ConvTranspose1d(nz, 256, kernel_size=3, padding=1, bias=False) #100, 256
        self.bn1 = nn.BatchNorm1d(256)
        # self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        #relu

        self.conv2t = nn.ConvTranspose1d(256, 512, kernel_size=3, padding=1, bias=False) #256, 512
        self.bn2 = nn.BatchNorm1d(512)
        # self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        #relu

        self.conv3t = nn.ConvTranspose1d(512, 128, kernel_size=3, padding=1, bias=False) #512, 128
        self.bn3 = nn.BatchNorm1d(128)
        # self.maxpool3 = nn.MaxPool1d(kernel_size=2)
        #tanh

        self.conv4t = nn.ConvTranspose1d(128, 64, kernel_size=3, padding=1, bias=False) #128, 64
        #relu
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(640, 320)#tanh 64, 40 if maxpool 768 else 6400
        self.fc2 = nn.Linear(320, out_params)
        

    def forward(self, x):
        x = x.permute(0, 2, 1)
        # x = x.permute(1, 0)
        # x = torch.relu(self.maxpool1(self.bn1(self.conv1t(x))))
        # x = torch.relu(self.maxpool2(self.bn2(self.conv2t(x))))
        # x = torch.tanh(self.maxpool3(self.bn3(self.conv3t(x))))
        x = torch.relu(self.bn1(self.conv1t(x)))
        x = torch.relu(self.bn2(self.conv2t(x)))
        x = torch.tanh(self.bn3(self.conv3t(x)))
        x = torch.relu(self.conv4t(x))
        x = self.flatten(x)
        output = torch.tanh(self.fc2(self.fc(x)))
        return output

checkpointflag = False

def make_models(flow_model):
    extvar["begin"] = 0
    extvar["start_pos"] = [256, 192]
    extvar["length_multiplier"] = 1
    extvar["next_from_slider_end"] = GAN_PARAMS["next_from_slider_end"]

    note_group_size = GAN_PARAMS["note_group_size"]
    #g_input_size = GAN_PARAMS["g_input_size"]
    generator = Generator(note_group_size * 4).to(device)
    mapping_layer = PyTorchCustomMappingLayer(extvar).to(device)

    # Optimizers
    beta1 = 0.5
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(beta1, 0.999))#0.001 0.0002
    generator.apply(weights_init)

    # load model
    try:
        print("loading checkpoint")
        # checkpoint = torch.load("flow_wgan-gptest-fixed-gan.pth")
        generator.load_state_dict(torch.load(flow_model))
    except:
        checkpointflag = True
        print("failed to load checkpoint")
    # try:
        # print("loading checkpoint")
        # cnn.load_state_dict(torch.load("cnn_E17400_L0.03456772863864899.pth"))
    # except:
        # print("f")
    # try:
    #     checkpoint = torch.load("flow_cnn.pth")
    #     cnn.load_state_dict(checkpoint["model_state_dict"])
    #     print("loading checkpoint")
    # except:
    #     print("making new model")
    
    # lcnn = "modelname.pth"
    # if lcnn != '':
        # print("loading checkpoint")
        # cnn.load_state_dict(torch.load(lcnn))
    return generator, mapping_layer, optimizer_g

losses = []

def generate_set_pytorch(models, begin = 0, start_pos=[256, 192], group_id=-1, length_multiplier=1, plot_map=True):
    """
    Generate one set (note_group_size) of notes.
    training until all the notes satisfy exit conditions (within boundaries).
    If the training goes on until (max_epoch = 25), it exits anyways.
    
    plot_map flag is only used for debugging.
    """
    extvar["begin"] = begin
    extvar["start_pos"] = start_pos
    extvar["length_multiplier"] = length_multiplier
    extvar["next_from_slider_end"] = GAN_PARAMS["next_from_slider_end"]
    note_group_size = GAN_PARAMS["note_group_size"]
    max_epoch = GAN_PARAMS["max_epoch"]
    good_epoch = GAN_PARAMS["good_epoch"] - 1
    g_multiplier = GAN_PARAMS["g_epochs"]
    c_multiplier = GAN_PARAMS["c_epochs"]
    g_batch = GAN_PARAMS["g_batch"]
    g_input_size = GAN_PARAMS["g_input_size"]
    c_true_batch = GAN_PARAMS["c_true_batch"]
    c_false_batch = GAN_PARAMS["c_false_batch"]
    c_randfalse_batch = GAN_PARAMS["c_randfalse_batch"]
    generator, mapping_layer, optimizer_g = models

    # set_extvar
    mapping_layer.set_extvar(extvar)

    # ----------
    #  Training base of a dcgan + loss of a wgan-gp
    # ----------
    # if checkpointflag is True:
    #     for epoch in range(num_epochs):
    #         for _ in range(5):
    #             ############################
    #             # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    #             ###########################
    #             ## Train with all-real batch
    #             netD.zero_grad()
    #             rn = np.random.randint(0, special_train_data.shape[0], (c_true_batch,))
    #             actual_train_data = torch.tensor(special_train_data[rn], dtype=torch.float32, device=device)
    #             #actual_train_labels = torch.tensor(special_train_labels[rn], dtype=torch.float32, device=device)
                
    #             # Forward pass real batch through D
    #             outputr = netD(actual_train_data) # 6 in 1 out
    #             # Calculate loss on all-real batch
    #             # errD_real = criterion(output.squeeze(1), actual_train_labels.unsqueeze(1))
    #             # Calculate gradients for D in backward pass
    #             #errD_real.backward()
    #             D_x = outputr.mean().item()
                
    #             ## Train with all-fake batch
    #             # Generate batch of latent vectors
    #             rand_maps = torch.rand(c_true_batch, 100, 100, device=device)
    #             #randfalse_labels = torch.zeros(150, device=device)
    #             # Generate fake image batch with G
                
    #             fake = netG(rand_maps)# in 6 out is 40 not 6 need ml to fix!!!!!!!!!!!!!!! done
    #             mloutput = mapping_layer(fake)
                
    #             # Classify all fake batch with D
    #             outputf = netD(mloutput.detach())# is needs to be detached
    #             # Calculate D's loss on the all-fake batch
    #             # errD_fake = criterion(output.squeeze(1), randfalse_labels.unsqueeze(1))
            
    #             # Calculate the gradients for this batch, accumulated (summed) with previous gradients
    #             # d_loss = d_loss_fn(outputr.squeeze(1), outputf.squeeze(1))
    #             gradient_penalty = compute_gradient_penalty(netD, actual_train_data.data, rand_maps.data)
    #             d_loss = -torch.mean(outputr) + torch.mean(outputf) + lambda_gp * gradient_penalty
    #             d_loss.backward()
    #             #errD_fake.backward()
    #             D_G_z1 = outputf.mean().item()
    #             # Compute error of D as sum over the fake and the real batches
    #             # errD = errD_real + errD_fake
    #             errD = d_loss
    #             # Update D
    #             optimizerD.step()
            
    #         ############################
    #         # (2) Update G network: maximize log(D(G(z)))
    #         ###########################
    #         netG.zero_grad()
    #         optimizerG.zero_grad()
    #         # Since we just updated D, perform another forward pass of all-fake batch through D
    #         output = netD(mloutput)
    #         # Calculate G's loss based on this output wasserstein_loss
    #         #errG = criterion(output.squeeze(1), torch.ones(150, device=device).unsqueeze(1))#actual_train_labels.unsqueeze(1)
    #         errG = -torch.mean(output)
    #         # add box loss too so it gan loss + box loss
    #         cri_loss = BoxCustomLoss(box_loss_border, box_loss_value, mloutput) * box_loss_weight
    #         loss_all = errG + cri_loss
    #         # Calculate gradients for G
    #         loss_all.backward(retain_graph=True)
    #         D_G_z2 = output.mean().item()
    #         # Update G
    #         optimizerG.step()
            
    #         # Output training stats
    #         print('[%d/%d] Loss_D: %.4f Loss_G: %.4f Loss_Box: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' % (epoch, num_epochs, errD.item(), loss_all.item(), cri_loss.item(), D_x, D_G_z1, D_G_z2))
            
            
    #         # Save Losses for plotting later
    #         G_losses.append(errG.item())
    #         D_losses.append(errD.item())
    #         boxlosses.append(cri_loss.item())

    #inference
    for _ in range(max_epoch):
        # map output make a new set of notes
        true_rand_maps = torch.rand(1, note_group_size, special_train_data.shape[2], device=device)# (batch_size, dim, sequence length)
        with torch.no_grad():
            true_cnnoutput = generator(true_rand_maps)
            true_mloutput = mapping_layer(true_cnnoutput)
        
        if plot_map:# plot map output
            plot_current_map(true_mloutput, extvar["begin"])
        
        # early return if found a good solution
        # good is (inside the map boundary)
        # if epoch >= good_epoch:
        current_map = true_mloutput
        if inblock_trueness(current_map[:, :, 0:2]).item() == 0 and inblock_trueness(current_map[:, :, 4:6]).item() == 0:
            break
    
    # for epoch in range(max_epoch):
    #     rand_maps = torch.rand(c_true_batch//2, note_group_size, special_train_data.shape[2], device=device)
    #     rn = np.random.randint(0, special_train_data.shape[0], (c_true_batch,))
    #     actual_train_data = torch.cat((rand_maps, torch.tensor(special_train_data[rn], dtype=torch.float32, device=device)), dim=0)

    #     optimizer_g.zero_grad()
    #     cnnoutput = generator(actual_train_data)
    #     mloutput = mapping_layer(cnnoutput)
    #     cri_loss = BoxCustomLoss(GAN_PARAMS["box_loss_border"], GAN_PARAMS["box_loss_value"], mloutput) * GAN_PARAMS["box_loss_weight"]
    #     cri_loss.backward()
    #     optimizer_g.step()
    #     losses.append(cri_loss.item())

    #     if GAN_PARAMS["verbose"]:
    #         print(f"[{group_id}/{epoch}] CNN_loss: {cri_loss.item()}")# [Group/Epoch]

    #     # map output make a new set of notes
    #     true_rand_maps = torch.rand(1, note_group_size, special_train_data.shape[2], device=device)# (batch_size, dim, sequence length)
    #     with torch.no_grad():
    #         true_cnnoutput = generator(true_rand_maps)
    #         true_mloutput = mapping_layer(true_cnnoutput)

    #     if plot_map:# plot map output
    #         plot_current_map(true_mloutput, extvar["begin"])

    #     # early return if found a good solution
    #     # good is (inside the map boundary)
    #     if epoch >= good_epoch:
    #         current_map = true_mloutput
    #         if inblock_trueness(current_map[:, :, 0:2]).item() == 0 and inblock_trueness(current_map[:, :, 4:6]).item() == 0:
    #             break
    return true_mloutput.squeeze()

def generate_map(flow_model):
    """
    Generate the map (main function)
    dist_multiplier is used here
    """
    o = []
    note_group_size = GAN_PARAMS["note_group_size"]
    pos = [np.random.randint(100, 412), np.random.randint(80, 304)]
    models = make_models(flow_model)
    converttensor = torch.cuda.is_available()

    print("# of groups: {}".format(timestamps.shape[0] // note_group_size))
    for i in tqdm(range(timestamps.shape[0] // note_group_size)):
        z = generate_set_pytorch(models, begin = i * note_group_size, start_pos = pos, length_multiplier = dist_multiplier, group_id = i, plot_map=False)[:, :6] * torch.tensor([512, 384, 1, 1, 512, 384], dtype=torch.float32, device=device) #np.array([512, 384, 1, 1, 512, 384])
        if converttensor:
            z = z.detach().cpu().numpy()# .cpu() might slow down?
        else:
            z = z.detach().numpy()# Use detach() before calling numpy()
        pos = z[-1, 0:2]
        o.append(z)
    a = np.concatenate(o, axis=0)
    return a

def put_everything_in_the_center():
    o = []
    print("max_epoch = 0: putting everyting in the center")
    for i in range(timestamps.shape[0]):
        z = [256, 192, 0, 0, 256 + slider_lengths[i], 192]
        o.append(z)
    a = np.array(o)
    return a

def generate_test():
    """
    This is only used in debugging.
    Generates a test map with plotting on.
    """
    pos = [384, 288]
    note_group_size = GAN_PARAMS["note_group_size"]
    generate_set_pytorch(begin = 3 * note_group_size, start_pos = pos, length_multiplier = dist_multiplier, group_id = 3, plot_map=True)

def print_osu_text(a):
    """
    This is only used in debugging.
    Prints .osu text directly.
    """
    for i, ai in enumerate(a):
        if not is_slider[i]:
            print("{},{},{},1,0,0:0:0".format(int(ai[0]), int(ai[1]), int(timestamps[i])))
        else:
            print("{},{},{},2,0,L|{}:{},1,{},0:0:0".format(int(ai[0]), int(ai[1]), int(timestamps[i]), int(round(ai[0] + ai[2] * slider_lengths[i])), int(round(ai[1] + ai[3] * slider_lengths[i])), int(slider_length_base[i] * slider_ticks[i])))


def step6_run_all_v2(flow_dataset_npz = "flow_dataset.npz", flow_model = "flow_wgan-gp.pth"):
    """
    Runs everything from building model to generating map.
    A lot of globals because currently it was directly cut from ipython notebook. Shouldn't hurt anything outside this file.
    """
    global objs, predictions, ticks, timestamps, is_slider, is_spinner, is_note_end, sv, slider_ticks, dist_multiplier, divisor, note_distance_basis
    global slider_length_base, slider_types, slider_type_rotation, slider_cos, slider_sin, slider_cos_each, slider_sin_each, slider_type_length, slider_lengths
    global tick_diff, note_distances, maps, labels, special_train_data, special_train_labels, extvar, plot_noise

    objs, predictions, ticks, timestamps, is_slider, is_spinner, is_note_end, sv, slider_ticks, dist_multiplier = read_map_predictions("rhythm_data.npz")

    # get divisor from GAN_PARAMS
    divisor = GAN_PARAMS["divisor"]

    # get basis
    note_distance_basis = GAN_PARAMS["note_distance_basis"]

    # get next_from_slider_end
    next_from_slider_end = GAN_PARAMS["next_from_slider_end"]


    # should be slider length each tick, which is usually SV * SMP * 100 / 4
    # e.g. SV 1.6, timing section x1.00, 1/4 divisor, then slider_length_base = 40
    slider_length_base = sv / divisor

    # weight for each type of sliders
    slider_type_probs = [0.25, 0.25, 0.25, 0.05, 0.05, 0.03, 0.03, 0.01, 0.01, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.015, 0.015, 0.01]
    slider_types = np.random.choice(len(slider_type_probs), is_slider.shape, p=slider_type_probs).astype(int)

    # these data must be kept consistent with the sliderTypes in load_map.js
    slider_type_rotation = np.array([0, -0.40703540572409336, 0.40703540572409336, -0.20131710837464062, 0.20131710837464062,
        -0.46457807316944644, 0.46457807316944644, 1.5542036732051032, -1.5542036732051032, 0, 0, 0.23783592745745077, -0.23783592745745077,
        0.5191461142465229, -0.5191461142465229, -0.16514867741462683, 0.16514867741462683, 3.141592653589793])

    # this is vector length! I should change the variable name probably...
    slider_type_length = np.array([1.0, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.64, 0.64, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.96, 0.96, 0])

    slider_cos = np.cos(slider_type_rotation)
    slider_sin = np.sin(slider_type_rotation)

    slider_cos_each = slider_cos[slider_types]
    slider_sin_each = slider_sin[slider_types]


    slider_lengths = np.array([slider_type_length[int(k)] * slider_length_base[i] for i, k in enumerate(slider_types)]) * slider_ticks

    tick_diff = np.concatenate([[100], ticks[1:] - ticks[:-1]])

    if next_from_slider_end:
        tick_diff = np.concatenate([[100], tick_diff[1:] - np.floor(slider_ticks * is_slider)[:-1]])

    # Timing section reset == tick_diff < 0
    # Use 1 as default value
    tick_diff = np.where(tick_diff < 0, 1, tick_diff)

    note_distances = np.clip(tick_diff, 1, divisor * 2) * (note_distance_basis / divisor)

    # Fallback for local version
    if not os.path.isfile(flow_dataset_npz) and flow_dataset_npz == "flow_dataset.npz":
        print("Flow dataset not found! Trying default model...")
        flow_dataset_npz = "models/default/flow_dataset.npz"

    # Fallback for local version
    if not os.path.isfile(flow_model) and flow_model == "flow_dataset.pth":
        print("Flow model not found! Trying default model...")
        flow_model = "models/default/flow_wgan-gp.pth"

    # Load the flow dataset saved in part 4
    with np.load(flow_dataset_npz) as flow_dataset:
        maps = flow_dataset["maps"]
        labels = np.ones(maps.shape[0])

    order2 = np.argsort(np.random.random(maps.shape[0]))
    special_train_data = maps[order2]
    special_train_labels = labels[order2]

    # Start model training

    # loss_ma = [90, 90, 90]
    extvar = {"begin": 10}

    plot_noise = np.random.random((1, GAN_PARAMS["g_input_size"]))

    if GAN_PARAMS["max_epoch"] == 0:
        osu_a = put_everything_in_the_center()
    else:
        osu_a = generate_map(flow_model)

    data = objs, predictions, ticks, timestamps, is_slider, is_spinner, is_note_end, sv, slider_ticks, dist_multiplier, slider_types, slider_length_base
    return osu_a, data
