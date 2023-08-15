# Part 7 action script

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from rhythm_loader import read_map_predictions
from losses_torch import GenerativeCustomLoss, BoxCustomLoss, AlwaysZeroCustomLoss
from plot_tools import MyLine

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

GAN_PARAMS = {
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

def step6_set_gan_params(params):
    """
    Basically Object.assign(GAN_PARAMS, params)
    See stackoverflow 38987
    """
    global GAN_PARAMS
    GAN_PARAMS = {**GAN_PARAMS, **params}

class ClassifierModel(nn.Module):
    """
    Classifier model to determine if a map is "fake" (generated) or "true" (part of the training set).
    Haven't experimented with the structures a lot, so you might want to try them.
    Using LSTM instead of SimpleRNN seems to yield very weird results.
    """
    def __init__(self, input_size):
        super(ClassifierModel, self).__init__()
        self.rnn_layer = nn.RNN(input_size, 64, batch_first=True)
        self.dense_layer1 = nn.Linear(64, 64)
        self.dense_layer2 = nn.Linear(64, 64)
        self.dense_layer3 = nn.Linear(64, 64)
        self.dense_layer4 = nn.Linear(64, 1)

    def forward(self, x):
        rnn_output, _ = self.rnn_layer(x)
        dense1 = self.dense_layer1(rnn_output)
        dense2 = torch.relu(self.dense_layer2(dense1))
        dense3 = torch.tanh(self.dense_layer3(dense2))
        dense4 = torch.relu(self.dense_layer4(dense3))
        output = torch.tanh(dense4)
        output = (output + 1) / 2
        return output

def inblock_trueness(vg):
    """
    Despite the weird name, it checks if all notes and slider tails are within the map boundaries.
    """
    wall_var_l = torch.tensor(vg < 0, dtype=torch.float32)
    wall_var_r = torch.tensor(vg > 1, dtype=torch.float32)
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
    # var_tensor = tf.cast(var_tensor, tf.float32)
    var_tensor = var_tensor.to(torch.float32)
    var_shape = var_tensor.shape
    wall_l = 0.15
    wall_r = 0.85
    x_max = 512
    y_max = 384
    # out = []
    # cp = tf.constant([256, 192, 0, 0])
    # cp = torch.tensor([256, 192, 0, 0], dtype=torch.float32)
    # phase = 0

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

#     note_distances_now = length_multiplier * np.expand_dims(note_distances[begin_offset:begin_offset+half_tensor], axis=0)
#     note_angles_now = np.expand_dims(note_angles[begin_offset:begin_offset+half_tensor], axis=0)

    # Load external arrays as tensors
    relevant_tensors = extvar["relevant_tensors"]
    relevant_is_slider =      relevant_tensors["is_slider"]
    relevant_slider_lengths = relevant_tensors["slider_lengths"]
    relevant_slider_types =   relevant_tensors["slider_types"]
    relevant_slider_cos =     relevant_tensors["slider_cos_each"]
    relevant_slider_sin =     relevant_tensors["slider_sin_each"]
    relevant_note_distances = relevant_tensors["note_distances"]

    # note_distances_now = length_multiplier * tf.expand_dims(relevant_note_distances, axis=0)
    note_distances_now = length_multiplier * torch.unsqueeze(relevant_note_distances, dim=0)

    # init
    # l = tf.convert_to_tensor(note_distances_now, dtype="float32")
    l = torch.tensor(note_distances_now, dtype=torch.float32)
    sl = l * 0.7

    cos_list = var_tensor[:, 0:half_tensor * 2]
    sin_list = var_tensor[:, half_tensor * 2:]
    # len_list = tf.sqrt(tf.square(cos_list) + tf.square(sin_list))
    len_list = torch.sqrt(torch.square(cos_list) + torch.square(sin_list))
    cos_list = cos_list / len_list
    sin_list = sin_list / len_list

    wall_l = 0.05 * x_max + l * 0.5
    wall_r = 0.95 * x_max - l * 0.5
    wall_t = 0.05 * y_max + l * 0.5
    wall_b = 0.95 * y_max - l * 0.5
#     rerand = tf.cast(tf.greater(l, y_max / 2), tf.float32);
#     not_rerand = tf.cast(tf.less_equal(l, y_max / 2), tf.float32);

    tick_diff = extvar["tick_diff"]

    # max_ticks_for_ds is an int variable, converted to float to avoid potential type error
    # use_ds = tf.expand_dims(tf.cast(tf.less_equal(tick_diff, extvar["max_ticks_for_ds"]), tf.float32), axis=0)
    use_ds = torch.unsqueeze(torch.tensor(tick_diff <= extvar["max_ticks_for_ds"], dtype=torch.float32), dim=0)


    # rerand = not use distance snap
    rerand = 1 - use_ds
    not_rerand = use_ds

    next_from_slider_end = extvar["next_from_slider_end"]

    # Starting position
    # if "start_pos" in extvar:
    #     _pre_px = extvar["start_pos"][0]
    #     _pre_py = extvar["start_pos"][1]
    #     _px = tf.cast(_pre_px, tf.float32)
    #     _py = tf.cast(_pre_py, tf.float32)
    # else:
    #     _px = tf.cast(256, tf.float32)
    #     _py = tf.cast(192, tf.float32)
    if "start_pos" in extvar:
        _pre_px = extvar["start_pos"][0]
        _pre_py = extvar["start_pos"][1]
        _px = torch.tensor(_pre_px, dtype=torch.float32)
        _py = torch.tensor(_pre_py, dtype=torch.float32)
    else:
        _px = torch.tensor(256, dtype=torch.float32)
        _py = torch.tensor(192, dtype=torch.float32)

    # this is not important since the first position starts at _ppos + Δpos
    # _x = tf.cast(256, tf.float32)
    # _y = tf.cast(192, tf.float32)
    _x = torch.tensor(256, dtype=torch.float32)
    _y = torch.tensor(192, dtype=torch.float32)

    # Use a buffer to save output
    # outputs = tf.TensorArray(tf.float32, half_tensor)
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
        # wall_value_l =    tf.cast(tf.less(_px, wall_l[:, k]), tf.float32)
        # wall_value_r =    tf.cast(tf.greater(_px, wall_r[:, k]), tf.float32)
        # wall_value_xmid = tf.cast(tf.greater(_px, wall_l[:, k]), tf.float32) * tf.cast(tf.less(_px, wall_r[:, k]), tf.float32)
        # wall_value_t =    tf.cast(tf.less(_py, wall_t[:, k]), tf.float32)
        # wall_value_b =    tf.cast(tf.greater(_py, wall_b[:, k]), tf.float32)
        # wall_value_ymid = tf.cast(tf.greater(_py, wall_t[:, k]), tf.float32) * tf.cast(tf.less(_py, wall_b[:, k]), tf.float32)
        wall_value_l = torch.tensor(_px < wall_l[:, k], dtype=torch.float32)
        wall_value_r = torch.tensor(_px > wall_r[:, k], dtype=torch.float32)
        wall_value_xmid = (torch.tensor(_px > wall_l[:, k], dtype=torch.float32) * torch.tensor(_px < wall_r[:, k], dtype=torch.float32))
        wall_value_t = torch.tensor(_py < wall_t[:, k], dtype=torch.float32)
        wall_value_b = torch.tensor(_py > wall_b[:, k], dtype=torch.float32)
        wall_value_ymid = (torch.tensor(_py > wall_t[:, k], dtype=torch.float32) * torch.tensor(_py < wall_b[:, k], dtype=torch.float32))

        # x_delta = tf.abs(delta_value_x) * wall_value_l - tf.abs(delta_value_x) * wall_value_r + delta_value_x * wall_value_xmid
        # y_delta = tf.abs(delta_value_y) * wall_value_t - tf.abs(delta_value_y) * wall_value_b + delta_value_y * wall_value_ymid
        x_delta = torch.abs(delta_value_x) * wall_value_l - torch.abs(delta_value_x) * wall_value_r + delta_value_x * wall_value_xmid
        y_delta = torch.abs(delta_value_y) * wall_value_t - torch.abs(delta_value_y) * wall_value_b + delta_value_y * wall_value_ymid

        # rerand_* if not using distance snap, (_p* + *_delta) if using distance snap
        _x = rerand[:, k] * rerand_x + not_rerand[:, k] * (_px + x_delta)
        _y = rerand[:, k] * rerand_y + not_rerand[:, k] * (_py + y_delta)
        # _x = rerand_x;
        # _y = rerand_y;
        # _x = _px + x_delta;
        # _y = _py + y_delta;

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

        # cp_slider = tf.transpose(tf.stack([_x / x_max, _y / y_max, _oa, _ob, (_x + _a * sln) / x_max, (_y + _b * sln) / y_max]))
        # _px_slider = tf.cond(next_from_slider_end, lambda: _x + _a * sln, lambda: _x)
        # _py_slider = tf.cond(next_from_slider_end, lambda: _y + _b * sln, lambda: _y)
        cp_slider = torch.stack([_x / x_max, _y / y_max, _oa, _ob, (_x + _a * sln) / x_max, (_y + _b * sln) / y_max]).T
        _px_slider = torch.where(next_from_slider_end, _x + _a * sln, _x)
        _py_slider = torch.where(next_from_slider_end, _y + _b * sln, _y)

        # circle part
        _a = rerand[:, k] * cos_list[:, k + half_tensor] + not_rerand[:, k] * cos_list[:, k]
        _b = rerand[:, k] * sin_list[:, k + half_tensor] + not_rerand[:, k] * sin_list[:, k]
        # _a = cos_list[:, k + half_tensor]
        # _b = sin_list[:, k + half_tensor]

        # cp_circle = tf.transpose(tf.stack([_x / x_max, _y / y_max, _a, _b, _x / x_max, _y / y_max]))
        cp_circle = torch.stack([_x / x_max, _y / y_max, _a, _b, _x / x_max, _y / y_max]).T
        _px_circle = _x
        _py_circle = _y

        # Outputs are scaled to [0,1] region
        # outputs = outputs.write(k, tf.where(relevant_is_slider[k], cp_slider, cp_circle))
        output_value = torch.where(relevant_is_slider[k], cp_slider, cp_circle)
        outputs.append(output_value)

        # Set starting point for the next circle/slider
        # _px = tf.where(tf.cast(relevant_is_slider[k], tf.bool), _px_slider, _px_circle)
        # _py = tf.where(tf.cast(relevant_is_slider[k], tf.bool), _py_slider, _py_circle)
        _px = torch.where(relevant_is_slider[k], _px_slider, _px_circle)
        _py = torch.where(relevant_is_slider[k], _py_slider, _py_circle)

    # return tf.transpose(outputs.stack(), [1, 0, 2])
    return torch.stack(outputs).permute(1, 0, 2)

class PyTorchCustomMappingLayer(nn.Module):
    def __init__(self, extvar, output_shape=(None, None), note_group_size=10):
        super(PyTorchCustomMappingLayer, self).__init__()
        self.extvar = extvar
        if output_shape[0] is None:
            output_shape = (special_train_data.shape[1], special_train_data.shape[2])
        self._output_shape = output_shape
        self.extvar_begin = nn.Parameter(torch.tensor(extvar["begin"], dtype=torch.int32), requires_grad=False)
        self.extvar_lmul = nn.Parameter(torch.tensor([extvar["length_multiplier"]], dtype=torch.float32), requires_grad=False)
        self.extvar_nfse = nn.Parameter(torch.tensor(extvar["next_from_slider_end"], dtype=torch.bool), requires_grad=False)
        self.extvar_mtfd = nn.Parameter(torch.tensor(GAN_PARAMS["max_ticks_for_ds"], dtype=torch.float32), requires_grad=False)
        # self.note_group_size = note_group_size
        self.note_group_size = GAN_PARAMS["note_group_size"]

        self.extvar_spos = nn.Parameter(torch.zeros(2, dtype=torch.float32), requires_grad=False)
        self.extvar_rel = nn.Parameter(torch.zeros(6, note_group_size, dtype=torch.float32), requires_grad=False)
        self.extvar_tickdiff = nn.Parameter(torch.zeros(note_group_size, dtype=torch.float32), requires_grad=False)

    def set_extvar(self, extvar):
        self.extvar = extvar

        begin_offset = extvar["begin"]
        relevant_tensors = {
            "is_slider": torch.tensor(is_slider[begin_offset: begin_offset + self.note_group_size], dtype=torch.bool),
            "slider_lengths": torch.tensor(slider_lengths[begin_offset: begin_offset + self.note_group_size], dtype=torch.float32),
            "slider_types": torch.tensor(slider_types[begin_offset: begin_offset + self.note_group_size], dtype=torch.float32),
            "slider_cos_each": torch.tensor(slider_cos_each[begin_offset: begin_offset + self.note_group_size], dtype=torch.float32),
            "slider_sin_each": torch.tensor(slider_sin_each[begin_offset: begin_offset + self.note_group_size], dtype=torch.float32),
            "note_distances": torch.tensor(note_distances[begin_offset: begin_offset + self.note_group_size], dtype=torch.float32),
        }
        self.extvar["relevant_tensors"] = relevant_tensors

        self.extvar_begin.data = torch.tensor(extvar["begin"], dtype=torch.int32)
        self.extvar_spos.data = torch.tensor(extvar["start_pos"], dtype=torch.float32)
        self.extvar_lmul.data = torch.tensor(extvar["length_multiplier"], dtype=torch.float32)# small fix
        self.extvar_nfse.data = torch.tensor(extvar["next_from_slider_end"], dtype=torch.bool)
        self.extvar_mtfd.data = torch.tensor(GAN_PARAMS["max_ticks_for_ds"], dtype=torch.float32)
        self.extvar_rel.data = torch.tensor([
            is_slider[begin_offset: begin_offset + self.note_group_size],
            slider_lengths[begin_offset: begin_offset + self.note_group_size],
            slider_types[begin_offset: begin_offset + self.note_group_size],
            slider_cos_each[begin_offset: begin_offset + self.note_group_size],
            slider_sin_each[begin_offset: begin_offset + self.note_group_size],
            note_distances[begin_offset: begin_offset + self.note_group_size],
        ], dtype=torch.float32)
        self.extvar_tickdiff.data = torch.tensor(
            tick_diff[begin_offset: begin_offset + self.note_group_size],
            dtype=torch.float32,
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

def plot_current_map(inputs):
    """
    This is only used in debugging.
    """
    # plot it each epoch
    mp = construct_map_with_sliders(inputs, extvar=extvar)
    # to make it clearer, add the start pos
    # npa = np.concatenate([[np.concatenate([extvar["start_pos"] / np.array([512, 384]), [0, 0]])], tf.stack(mp).numpy().squeeze()])
    npa = np.concatenate([[np.concatenate([extvar["start_pos"] / np.array([512, 384]), [0, 0]])], np.array(mp).squeeze()])
    fig, ax = plt.subplots()
    x, y = np.transpose(npa)[0:2]
    #x, y = np.random.rand(2, 20)
    line = MyLine(x, y, mfc='red', ms=12)
    line.text.set_color('red')
    line.text.set_fontsize(16)
    ax.add_line(line)
    plt.show()

class GenerativeModel(nn.Module):
    def __init__(self, in_params, out_params):
        super(GenerativeModel, self).__init__()
        self.layer1 = nn.Linear(in_params, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 128)
        self.layer4 = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, out_params)

    def forward(self, x):
        #x = x.to(self.layer1.weight.dtype)  # Convert x to the same dtype as layer weights
        x1 = torch.relu(self.layer1(x))
        x2 = torch.relu(self.layer2(x1))
        x3 = torch.tanh(self.layer3(x2))
        x4 = torch.relu(self.layer4(x3))
        out = torch.tanh(self.output_layer(x4))
        return out

class MixedModel(nn.Module):
    def __init__(self, generator, mapping_layer, discriminator, in_params):
        super(MixedModel, self).__init__()
        self.generator = generator
        self.mapping_layer = mapping_layer
        self.discriminator = discriminator

    def forward(self, inp):
        interm1 = self.generator(inp)
        interm2 = self.mapping_layer(interm1)
        end = self.discriminator(interm2)
        return interm1, interm2, end

def make_models():
    # Build models (generative, classifier, mixed)
    extvar["begin"] = 0
    extvar["start_pos"] = [256, 192]
    extvar["length_multiplier"] = 1
    extvar["next_from_slider_end"] = GAN_PARAMS["next_from_slider_end"]

    # classifier_model = ClassifierModel(special_train_data.shape[2]).to(device)
    classifier_model = ClassifierModel(special_train_data.shape[2])
    note_group_size = GAN_PARAMS["note_group_size"]
    g_input_size = GAN_PARAMS["g_input_size"]

    # gmodel = GenerativeModel(g_input_size, note_group_size * 4).to(device)
    # mapping_layer = PyTorchCustomMappingLayer(extvar).to(device)
    # mmodel = MixedModel(gmodel, mapping_layer, classifier_model, g_input_size).to(device)
    gmodel = GenerativeModel(g_input_size, note_group_size * 4)
    mapping_layer = PyTorchCustomMappingLayer(extvar)
    mmodel = MixedModel(gmodel, mapping_layer, classifier_model, g_input_size)
    # Set the discriminator to be untrainable
    for param in mmodel.discriminator.parameters():
        param.requires_grad = False

    default_weights = mmodel.state_dict()
    return gmodel, mapping_layer, classifier_model, mmodel, default_weights

def set_extvar(models, extvar):
    gmodel, mapping_layer, classifier_model, mmodel, default_weights = models
    mapping_layer.set_extvar(extvar)

def reset_model_weights(models):
    gmodel, mapping_layer, classifier_model, mmodel, default_weights = models
    weights = default_weights
    mmodel.load_state_dict(weights)

# loss function for mmodel
losses = [AlwaysZeroCustomLoss(), BoxCustomLoss(GAN_PARAMS["box_loss_border"], GAN_PARAMS["box_loss_value"]), GenerativeCustomLoss()]
loss_weights = [1e-8, GAN_PARAMS["box_loss_weight"], 1]
def combined_loss(outputs, targets, loss_weights):
    total_loss = 0
    for i, loss_fn in enumerate(losses):
        total_loss += loss_weights[i] * loss_fn(outputs[i], targets[i])
        #total_loss += criterion[i](output[i]) * loss_weights[i]# other version
    return total_loss

#https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py
#https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# adversarial_loss.cuda() #https://discuss.pytorch.org/t/why-criterion-cuda-is-not-needed-but-model-cuda-is/17410
# fix np to torch +
def generate_set_pytorch(models, begin = 0, start_pos=[256, 192], group_id=-1, length_multiplier=1, plot_map=True):
    """
    Generate one set (note_group_size) of notes.
    Trains at least (good_epoch = 6) epochs for each model, then continue training
    until all the notes satisfy exit conditions (within boundaries).
    If the training goes on until (max_epoch = 25), it exits anyways.
    
    Inside the training loop, each big epoch it trains generator for (g_epochs = 7)
    epochs, and classifier for (c_epochs = 3). The numbers are set up to balance the
    powers of those two models.
    
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

    reset_model_weights(models)
    set_extvar(models, extvar)
    _gmodel, _mapping_layer, classifier_model, mmodel, _default_weights = models
    generator = mmodel
    discriminator = classifier_model
    # Loss function
    criterion = nn.MSELoss() # Discriminator/Classifier + Generator not MGenerator
    
    # Optimizers
    # optimizer = optim.Adam(model.parameters(), lr=0.002)#0.002 gen
    optimizer_g = optim.Adam(generator.parameters(), lr=0.001)#0.001 mix
    optimizer_c = optim.Adam(discriminator.parameters(), lr=0.001)#0.001 class
    
    # ----------
    #  Training
    # ----------
    
    for i in range(max_epoch):

        # ginput_noise = np.random.random((g_batch, g_input_size))
        # glabel = [np.zeros((g_batch, note_group_size * 4)), np.ones((g_batch,)), np.ones((g_batch,))]
        ginput_noise = torch.rand(g_batch, g_input_size)
        glabel = [torch.zeros((g_batch, note_group_size * 4)), torch.ones((g_batch,)), torch.ones((g_batch,))]

        # -----------------
        #  Train Generator
        # -----------------
        # for _ in range(g_multiplier):
            # optimizer_g.zero_grad()
            # output = generator(ginput_noise)
            # g_loss = criterion(output, glabel)
            # g_loss.backward()
            # optimizer_g.step()

        # -----------------
        #  Train MGenerator
        # -----------------
        for _ in range(g_multiplier):
            optimizer_g.zero_grad()
            output = generator(ginput_noise)
            g_loss = criterion(output, glabel)
            # g_loss = combined_loss(output, glabel, loss_weights) custom loss broken RuntimeError: output with shape [] doesn't match the broadcast shape [1] on line 443
            g_loss.backward()
            optimizer_g.step()
        
        # pred_noise = np.random.random((c_false_batch, g_input_size))
        pred_noise = torch.rand(c_false_batch, g_input_size)
        pred_input = pred_noise
        _predicted_maps_data, new_false_maps, _predclass = generator(pred_input)
        # new_false_labels = np.zeros(c_false_batch)
        new_false_labels = torch.zeros(c_false_batch)

        # random numbers as negative samples
        # special_train_data.shape[2] == 6
        # randfalse_maps = np.random.rand(c_randfalse_batch, note_group_size, special_train_data.shape[2])
        # randfalse_labels = np.zeros(c_randfalse_batch)
        randfalse_maps = torch.rand(c_randfalse_batch, note_group_size, special_train_data.shape[2])
        randfalse_labels = torch.zeros(c_randfalse_batch)

        rn = np.random.randint(0, special_train_data.shape[0], (c_true_batch,))
        # actual_train_data = np.concatenate((new_false_maps, randfalse_maps, special_train_data[rn]), axis=0)
        # actual_train_labels = np.concatenate((new_false_labels, randfalse_labels, special_train_labels[rn]), axis=0)
        actual_train_data = torch.cat((new_false_maps, randfalse_maps, special_train_data[rn]), dim=0)
        actual_train_labels = torch.cat((new_false_labels, randfalse_labels, special_train_labels[rn]), dim=0)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        for _ in range(c_multiplier):
            optimizer_c.zero_grad()
            output2 = discriminator(actual_train_data)
            c_loss = criterion(output2, actual_train_labels)
            c_loss.backward()
            optimizer_c.step()

        print("Group {}, Epoch {}: G loss: {} vs. C loss: {}".format(group_id, 1+i, g_loss, c_loss))

        # make a new set of notes
        # res_noise = np.random.random((1, g_input_size))
        res_noise = torch.rand(1, g_input_size)
        _resgenerated, res_map, _resclass = generator(res_noise)
        if plot_map:
            plot_current_map(torch.tensor(res_map, dtype=torch.float32))

        # early return if found a good solution
        # good is (inside the map boundary)
        # if i >= good_epoch:
        #     current_map = res_map
        #     if inblock_trueness(current_map[:, :, 0:2]).numpy()[0] == 0 and inblock_trueness(current_map[:, :, 4:6]).numpy()[0] == 0:
        #         break

        if i >= good_epoch:
            current_map = res_map
            if inblock_trueness(current_map[:, :, 0:2]).item() == 0 and inblock_trueness(current_map[:, :, 4:6]).item() == 0:
                break

    if plot_map:
        for i in range(3): # from our testing, any random input generates nearly the same map
            # plot_noise = np.random.random((1, g_input_size))
            plot_noise = torch.rand(1, g_input_size)
            _plotgenerated, plot_mapped, _plotclass = generator(plot_noise)
            plot_current_map(torch.tensor(plot_mapped, dtype=torch.float32))

    return res_map.squeeze()

def generate_map():
    """
    Generate the map (main function)
    dist_multiplier is used here
    """
    o = []
    note_group_size = GAN_PARAMS["note_group_size"]
    pos = [np.random.randint(100, 412), np.random.randint(80, 304)]
    models = make_models()

    print("# of groups: {}".format(timestamps.shape[0] // note_group_size))
    for i in range(timestamps.shape[0] // note_group_size):
        z = generate_set_pytorch(models, begin = i * note_group_size, start_pos = pos, length_multiplier = dist_multiplier, group_id = i, plot_map=False)[:, :6] * np.array([512, 384, 1, 1, 512, 384])
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


def step6_run_all(flow_dataset_npz = "flow_dataset.npz"):
    """
    Runs everything from building model to generating map.
    A lot of globals because currently it was directly cut from ipython notebook. Shouldn't hurt anything outside this file.
    """
    global objs, predictions, ticks, timestamps, is_slider, is_spinner, is_note_end, sv, slider_ticks, dist_multiplier, divisor, note_distance_basis
    global slider_length_base, slider_types, slider_type_rotation, slider_cos, slider_sin, slider_cos_each, slider_sin_each, slider_type_length, slider_lengths
    global tick_diff, note_distances, maps, labels, special_train_data, special_train_labels, loss_ma, extvar, plot_noise

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

    # Load the flow dataset saved in part 4
    with np.load(flow_dataset_npz) as flow_dataset:
        maps = flow_dataset["maps"]
        labels = np.ones(maps.shape[0])

    order2 = np.argsort(np.random.random(maps.shape[0]))
    special_train_data = maps[order2]
    special_train_labels = labels[order2]

    # Start model training

    loss_ma = [90, 90, 90]
    extvar = {"begin": 10}

    plot_noise = np.random.random((1, GAN_PARAMS["g_input_size"]))

    if GAN_PARAMS["max_epoch"] == 0:
        osu_a = put_everything_in_the_center()
    else:
        osu_a = generate_map()

    data = objs, predictions, ticks, timestamps, is_slider, is_spinner, is_note_end, sv, slider_ticks, dist_multiplier, slider_types, slider_length_base
    return osu_a, data
