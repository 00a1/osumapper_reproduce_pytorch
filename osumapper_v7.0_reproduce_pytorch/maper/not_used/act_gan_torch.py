# Part 7 action script

import torch
import torch.nn as nn
import torch.optim as optim

# import tensorflow as tf
# from tensorflow import keras;

import numpy as np
import matplotlib.pyplot as plt
import os
# from datetime import datetime
from rhythm_loader import *;
from losses import GenerativeCustomLoss, BoxCustomLoss, AlwaysZeroCustomLoss;
from plot_tools import MyLine;

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
};

def step6_set_gan_params(params):
    """
    Basically Object.assign(GAN_PARAMS, params)
    See stackoverflow 38987
    """
    global GAN_PARAMS;
    GAN_PARAMS = {**GAN_PARAMS, **params};

# def build_classifier_model():
#     """
#     Classifier model to determine if a map is "fake" (generated) or "true" (part of the training set).
#     Haven't experimented with the structures a lot, so you might want to try them.
#     Using LSTM instead of SimpleRNN seems to yield very weird results.
#     """
#     model = keras.Sequential([
#         keras.layers.SimpleRNN(64, input_shape=(special_train_data.shape[1], special_train_data.shape[2])),
#         keras.layers.Dense(64),
#         keras.layers.Dense(64, activation=tf.nn.relu),
#         keras.layers.Dense(64, activation=tf.nn.tanh),
#         keras.layers.Dense(64, activation=tf.nn.relu),
#         keras.layers.Dense(1, activation=tf.nn.tanh),
#         keras.layers.Lambda(lambda x: (x+1)/2, output_shape=(1,)),
#     ])

#     try:
#         optimizer = tf.optimizers.Adam(0.001)
#     except:
#         optimizer = tf.train.AdamOptimizer(0.001)

#     model.compile(loss='mse',
#                 optimizer=optimizer,
#                 metrics=[keras.metrics.mae])
#     return model

class ClassifierModel(nn.Module):
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

# def inblock_trueness(vg):
#     """
#     Despite the weird name, it checks if all notes and slider tails are within the map boundaries.
#     """
#     wall_var_l = tf.cast(tf.less(vg, 0), tf.float32)
#     wall_var_r = tf.cast(tf.greater(vg, 1), tf.float32)
#     return tf.reduce_mean(tf.reduce_mean(wall_var_l + wall_var_r, axis=2), axis=1)

def inblock_trueness(vg):
    """
    Despite the weird name, it checks if all notes and slider tails are within the map boundaries.
    """
    wall_var_l = torch.tensor(vg < 0, dtype=torch.float32)
    wall_var_r = torch.tensor(vg > 1, dtype=torch.float32)
    return torch.mean(torch.mean(wall_var_l + wall_var_r, dim=2), dim=1)

# def cut_map_chunks(c):
#     """
#     Cut a map into many chunks based on the chunk_size variable (note_group_size).
#     """
#     r = [];
#     for i in range(0, (c.shape[0] - chunk_size) // step_size):
#         chunk = c[i * step_size:i * step_size + chunk_size]
#         r.append(chunk);
#     return tf.stack(r);

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
    










# class KerasCustomMappingLayer(keras.layers.Layer):
#     """
#     Custom layer used in the generative model.
#     Turns an input tensor (None, 4 * note_group_size) into a map chunk (None, note_group_size, 6).

#     EXTRA VARIABLES:
#     begin: index of current note group (first group is 0, next group is 10 for example if note_group_size is 10)
#     lmul: length_multiplier in the parameters
#     nfse: next_from_slider_end in the parameters
#     mtfd: max_ticks_for_ds in the parameters
#     spos: start_pos, the coordinates where the previous note was (before the current group)
#     rel: the slider related arrays
#     tickdiff: difference measured in ticks from the previous note (ex: 2 = from 1/1 to 1/2, 4 = from 1/1 to 1/1)
#     """
#     def __init__(self, extvar, output_shape=(None, None), *args, **kwargs):
#         self.extvar = extvar
#         if output_shape[0] is None:
#             output_shape = (special_train_data.shape[1], special_train_data.shape[2])
#         self._output_shape = output_shape
#         self.extvar_begin = tf.Variable(tf.convert_to_tensor(extvar["begin"], dtype=tf.int32), trainable=False)
#         self.extvar_lmul =  tf.Variable(tf.convert_to_tensor([extvar["length_multiplier"]], dtype=tf.float32), trainable=False)
#         self.extvar_nfse =  tf.Variable(tf.convert_to_tensor(extvar["next_from_slider_end"], dtype=tf.bool), trainable=False)
#         self.extvar_mtfd = tf.Variable(tf.convert_to_tensor(GAN_PARAMS["max_ticks_for_ds"], dtype=tf.float32), trainable=False)
#         self.note_group_size = GAN_PARAMS["note_group_size"];

#         self.extvar_spos =  tf.Variable(tf.cast(tf.zeros((2, )), tf.float32), trainable=False)
#         self.extvar_rel =   tf.Variable(tf.cast(tf.zeros((6, self.note_group_size)), tf.float32), trainable=False)
#         self.extvar_tickdiff = tf.Variable(tf.cast(tf.zeros((self.note_group_size,)), tf.float32), trainable=False)

#         super(KerasCustomMappingLayer, self).__init__(*args, **kwargs)

#     def build(self, input_shape): # since this is a static layer, no building is required
#         pass

#     def set_extvar(self, extvar):
#         self.extvar = extvar;

#         # Populate extvar with the rel variable (this will modify the input extvar)
#         begin_offset = extvar["begin"];
#         self.extvar["relevant_tensors"] = {
#             "is_slider"       : tf.convert_to_tensor(is_slider      [begin_offset : begin_offset + self.note_group_size], dtype=tf.bool),
#             "slider_lengths"  : tf.convert_to_tensor(slider_lengths [begin_offset : begin_offset + self.note_group_size], dtype=tf.float32),
#             "slider_types"    : tf.convert_to_tensor(slider_types   [begin_offset : begin_offset + self.note_group_size], dtype=tf.float32),
#             "slider_cos_each" : tf.convert_to_tensor(slider_cos_each[begin_offset : begin_offset + self.note_group_size], dtype=tf.float32),
#             "slider_sin_each" : tf.convert_to_tensor(slider_sin_each[begin_offset : begin_offset + self.note_group_size], dtype=tf.float32),
#             "note_distances" :  tf.convert_to_tensor(note_distances [begin_offset : begin_offset + self.note_group_size], dtype=tf.float32)
#         };

#         # Continue
#         self.extvar_begin.assign(extvar["begin"])
#         self.extvar_spos.assign(extvar["start_pos"])
#         self.extvar_lmul.assign([extvar["length_multiplier"]])
#         self.extvar_nfse.assign(extvar["next_from_slider_end"])
#         self.extvar_mtfd.assign(GAN_PARAMS["max_ticks_for_ds"])
#         self.extvar_rel.assign(tf.convert_to_tensor([
#             is_slider      [begin_offset : begin_offset + self.note_group_size],
#             slider_lengths [begin_offset : begin_offset + self.note_group_size],
#             slider_types   [begin_offset : begin_offset + self.note_group_size],
#             slider_cos_each[begin_offset : begin_offset + self.note_group_size],
#             slider_sin_each[begin_offset : begin_offset + self.note_group_size],
#             note_distances [begin_offset : begin_offset + self.note_group_size]
#         ], dtype=tf.float32))
#         self.extvar_tickdiff.assign(tf.convert_to_tensor(
#             tick_diff[begin_offset : begin_offset + self.note_group_size]
#         , dtype=tf.float32))

#     # Call method will sometimes get used in graph mode,
#     # training will get turned into a tensor
#     # @tf.function
#     def call(self, inputs, training=None):
#         mapvars = inputs;
#         start_pos = self.extvar_spos
#         rel = self.extvar_rel
#         extvar = {
#             "begin" : self.extvar_begin,
#             # "start_pos" : self.extvar_start_pos,
#             "start_pos" : tf.cast(start_pos, tf.float32),
#             "length_multiplier" : self.extvar_lmul,
#             "next_from_slider_end" : self.extvar_nfse,
#             "tick_diff" : self.extvar_tickdiff,
#             "max_ticks_for_ds" : self.extvar_mtfd,
#             # "relevant_tensors" : self.extvar_rel
#             "relevant_tensors" : {
#                 "is_slider"       : tf.cast(rel[0], tf.bool),
#                 "slider_lengths"  : tf.cast(rel[1], tf.float32),
#                 "slider_types"    : tf.cast(rel[2], tf.float32),
#                 "slider_cos_each" : tf.cast(rel[3], tf.float32),
#                 "slider_sin_each" : tf.cast(rel[4], tf.float32),
#                 "note_distances"  : tf.cast(rel[5], tf.float32)
#             }
#         }
#         result = construct_map_with_sliders(mapvars, extvar=extvar);
#         return result;







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
        self.extvar_lmul.data = torch.tensor([extvar["length_multiplier"]], dtype=torch.float32)
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

# # Create an instance of the PyTorchCustomMappingLayer
# extvar = {}  # You need to populate this dictionary with your specific values
# output_shape = (None, None)  # Replace with your desired output shape
# note_group_size = 10  # Replace with your desired note group size
# custom_mapping_layer = PyTorchCustomMappingLayer(extvar, output_shape, note_group_size)













def plot_current_map(inputs):
    """
    This is only used in debugging.
    """
    # plot it each epoch
    mp = construct_map_with_sliders(inputs, extvar=extvar);
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

# def generative_model(in_params, out_params, loss_func='mse'):
#     """
#     Generative model to generate a set of random numbers.
#     """
#     model = keras.Sequential([
#         keras.layers.Dense(128, input_shape=(in_params,)),# activation=tf.nn.elu, input_shape=(train_data.shape[1],)),
#         keras.layers.Dense(128, activation=tf.nn.relu),
#         keras.layers.Dense(128, activation=tf.nn.tanh),
#         keras.layers.Dense(128, activation=tf.nn.relu),
#         keras.layers.Dense(out_params, activation=tf.nn.tanh)#,
#         # keras.layers.Lambda(lambda x: (x+1)/2, output_shape=(out_params,))
#     ])

#     try:
#         optimizer = tf.optimizers.Adam(0.002) #Adamoptimizer?
#     except:
#         optimizer = tf.train.AdamOptimizer(0.002) #Adamoptimizer?

#     model.compile(loss=loss_func,
#                 optimizer=optimizer,
#                 metrics=[keras.metrics.mae])
#     return model

class GenerativeModel(nn.Module):
    def __init__(self, in_params, out_params):
        super(GenerativeModel, self).__init__()
        self.layer1 = nn.Linear(in_params, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 128)
        self.layer4 = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, out_params)

    def forward(self, x):
        x1 = torch.relu(self.layer1(x))
        x2 = torch.relu(self.layer2(x1))
        x3 = torch.tanh(self.layer3(x2))
        x4 = torch.relu(self.layer4(x3))
        out = torch.tanh(self.output_layer(x4))
        return out


# def mixed_model(generator, mapping_layer, discriminator, in_params):
#     """
#     Mix the generative model and classifier model together.
#     Training loop:
#     - Train mixed model (where the classifier is set to untrainable) to essentially train generative model
#     - Train classifier model
#     """
#     note_group_size = GAN_PARAMS["note_group_size"];
#     inp = keras.layers.Input(shape=(in_params,))
#     start_pos = keras.layers.Input(shape = (2,))
#     # rel = keras.layers.Input(shape = (7, note_group_size))
#     interm1 = generator(inp)
#     interm2 = mapping_layer(interm1)
#     end = discriminator(interm2)
#     model = keras.Model(inputs = inp, outputs = [interm1, interm2, end])# nterm1, interm2, end = predicted_maps_data, predicted_maps_mapped, _predclass
#                                                                                                # _resgenerated, res_map, _resclass
#                                                                                                # _plotgenerated, plot_mapped, _plotclass

#     discriminator.trainable = False

#     try:
#         optimizer = tf.optimizers.Adam(0.001) #Adamoptimizer?
#     except:
#         optimizer = tf.train.AdamOptimizer(0.001) #Adamoptimizer?

#     losses = [AlwaysZeroCustomLoss(), BoxCustomLoss(GAN_PARAMS["box_loss_border"], GAN_PARAMS["box_loss_value"]), GenerativeCustomLoss()];

#     model.compile(loss=losses,
#                   loss_weights=[1e-8, GAN_PARAMS["box_loss_weight"], 1],
#                 optimizer=optimizer)
#     return model

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

    # classifier_model = build_classifier_model(); #dis(classifier)
    classifier_model = ClassifierModel(special_train_data.shape[2])
    note_group_size = GAN_PARAMS["note_group_size"]
    g_input_size = GAN_PARAMS["g_input_size"]

    # gmodel = generative_model(g_input_size, note_group_size * 4) #gen
    gmodel = GenerativeModel(g_input_size, note_group_size * 4)
    # mapping_layer = KerasCustomMappingLayer(extvar)
    mapping_layer = PyTorchCustomMappingLayer(extvar)
    # mmodel = mixed_model(gmodel, mapping_layer, classifier_model, g_input_size) #gen + MappingLayer + dis(classifier)
    mmodel = MixedModel(gmodel, mapping_layer, classifier_model, g_input_size)
    # Set the discriminator to be untrainable
    for param in mmodel.discriminator.parameters():
        param.requires_grad = False

    #default_weights = mmodel.get_weights()
    default_weights = mmodel.state_dict()

    #----------------------------move init to def make model------------------------------------
    # Initialize generator and discriminator
    # generator = Generator().to(device)#temp
    # gmodel = GenerativeModel(g_input_size, note_group_size * 4)
    # mmodel = 
    # discriminator = Discriminator().to(device)
    # discriminator = ClassifierModel(special_train_data.shape[2])

    return gmodel, mapping_layer, classifier_model, mmodel, default_weights






def set_extvar(models, extvar):
    gmodel, mapping_layer, classifier_model, mmodel, default_weights = models;
    mapping_layer.set_extvar(extvar);

# def reset_model_weights(models):
#     gmodel, mapping_layer, classifier_model, mmodel, default_weights = models;
#     weights = default_weights;
#     mmodel.set_weights(weights);

def reset_model_weights(models):
    gmodel, mapping_layer, classifier_model, mmodel, default_weights = models
    weights = default_weights
    mmodel.load_state_dict(weights)

def generate_set(models, begin = 0, start_pos=[256, 192], group_id=-1, length_multiplier=1, plot_map=True):
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
    extvar["begin"] = begin;
    extvar["start_pos"] = start_pos;
    extvar["length_multiplier"] = length_multiplier;
    extvar["next_from_slider_end"] = GAN_PARAMS["next_from_slider_end"];

    note_group_size = GAN_PARAMS["note_group_size"];
    max_epoch = GAN_PARAMS["max_epoch"];
    good_epoch = GAN_PARAMS["good_epoch"] - 1;
    g_multiplier = GAN_PARAMS["g_epochs"];
    c_multiplier = GAN_PARAMS["c_epochs"];
    g_batch = GAN_PARAMS["g_batch"];
    g_input_size = GAN_PARAMS["g_input_size"];
    c_true_batch = GAN_PARAMS["c_true_batch"];
    c_false_batch = GAN_PARAMS["c_false_batch"];
    c_randfalse_batch = GAN_PARAMS["c_randfalse_batch"];

    reset_model_weights(models);
    set_extvar(models, extvar);
    gmodel, mapping_layer, classifier_model, mmodel, default_weights = models;

    for i in range(max_epoch):

        gnoise = np.random.random((g_batch, g_input_size));
        glabel = [np.zeros((g_batch, note_group_size * 4)), np.ones((g_batch,)), np.ones((g_batch,))]
        ginput = gnoise
        # trains the gen model first but the discriminator model is locked --------------------------------1
        # fit mmodel instead of gmodel
        history = mmodel.fit(ginput, glabel, epochs=g_multiplier,
                            validation_split=0.2, verbose=0,
                            callbacks=[])

        pred_noise = np.random.random((c_false_batch, g_input_size));
        pred_input = pred_noise
        predicted_maps_data, predicted_maps_mapped, _predclass = mmodel.predict(pred_input);
        new_false_maps = predicted_maps_mapped;
        new_false_labels = np.zeros(c_false_batch);

        # random numbers as negative samples
        # special_train_data.shape[2] == 6
        randfalse_maps = np.random.rand(c_randfalse_batch, note_group_size, special_train_data.shape[2]);
        randfalse_labels = np.zeros(c_randfalse_batch);

        rn = np.random.randint(0, special_train_data.shape[0], (c_true_batch,))
        actual_train_data = np.concatenate((new_false_maps, randfalse_maps, special_train_data[rn]), axis=0);
        actual_train_labels = np.concatenate((new_false_labels, randfalse_labels, special_train_labels[rn]), axis=0);

        #then trains the discriminator model ---------------------------------------------------------------2
        history2 = classifier_model.fit(actual_train_data, actual_train_labels, epochs=c_multiplier,
                            validation_split=0.2, verbose=0,
                            callbacks=[])

        # calculate the losses
        g_loss = np.mean(history.history['loss']);
        c_loss = np.mean(history2.history['loss']);
        print("Group {}, Epoch {}: G loss: {} vs. C loss: {}".format(group_id, 1+i, g_loss, c_loss));

        # delete the history to free memory
        del history, history2

        # make a new set of notes
        res_noise = np.random.random((1, g_input_size));
        res_input = res_noise
        _resgenerated, res_map, _resclass = mmodel.predict(res_input);
        if plot_map:
            # plot_current_map(tf.convert_to_tensor(res_map, dtype=tf.float32));
            plot_current_map(torch.tensor(res_map, dtype=torch.float32))

        # early return if found a good solution
        # good is (inside the map boundary)
        if i >= good_epoch:
            current_map = res_map;
            if inblock_trueness(current_map[:, :, 0:2]).numpy()[0] == 0 and inblock_trueness(current_map[:, :, 4:6]).numpy()[0] == 0:
                # debugging options to check map integrity
                # print(tf.reduce_mean(current_map));
                # print("-----MAPLAYER-----")
                # print(tf.reduce_mean(mapping_layer(conv_input(tf.convert_to_tensor(_resgenerated, dtype="float32"), extvar))));
                # print("-----CMWS-----")
                # print(tf.reduce_mean(construct_map_with_sliders(tf.convert_to_tensor(_resgenerated, dtype="float32"), extvar=mapping_layer.extvar)));
                break;

    if plot_map:
        for i in range(3): # from our testing, any random input generates nearly the same map
            plot_noise = np.random.random((1, g_input_size))
            plot_input = plot_noise
            _plotgenerated, plot_mapped, _plotclass = mmodel.predict(plot_input)
            # plot_current_map(tf.convert_to_tensor(plot_mapped, dtype=tf.float32))
            plot_current_map(torch.tensor(plot_mapped, dtype=torch.float32))

    # Don't do this in this version. It's for old versions where models are rebuilt each loop
    # del mmodel, mapping_layer;

    return res_map.squeeze();


#---------------------------------------------------------------- loss function for mmodel
losses = [AlwaysZeroCustomLoss(), BoxCustomLoss(GAN_PARAMS["box_loss_border"], GAN_PARAMS["box_loss_value"]), GenerativeCustomLoss()]
loss_weights = [1e-8, GAN_PARAMS["box_loss_weight"], 1]
def combined_loss(outputs, targets, loss_weights):
    total_loss = 0
    for i, loss_fn in enumerate(losses):
        total_loss += loss_weights[i] * loss_fn(outputs[i], targets[i])
        #total_loss += criterion[i](output[i]) * loss_weights[i]# other version
    return total_loss
#loss = combined_loss(out, y, loss_weights) # use
#----------------------------------------------------------------

#https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py
#https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# adversarial_loss.cuda() #https://discuss.pytorch.org/t/why-criterion-cuda-is-not-needed-but-model-cuda-is/17410
def generate_set_pytorch(models, begin = 0, start_pos=[256, 192], group_id=-1, length_multiplier=1, plot_map=True):
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

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

        ginput_noise = np.random.random((g_batch, g_input_size));
        glabel = [np.zeros((g_batch, note_group_size * 4)), np.ones((g_batch,)), np.ones((g_batch,))]

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
            # g_loss = criterion(output, glabel)
            g_loss = combined_loss(output, glabel, loss_weights)
            g_loss.backward()
            optimizer_g.step()
        
        # trains the gen model first but the discriminator model is locked --------------------------------1
        # fit mmodel instead of gmodel
        #history = mmodel.fit(ginput, glabel, epochs=g_multiplier, validation_split=0.2, verbose=0, callbacks=[])
        

        
        pred_noise = np.random.random((c_false_batch, g_input_size))
        pred_input = pred_noise
        _predicted_maps_data, new_false_maps, _predclass = generator(pred_input)
        new_false_labels = np.zeros(c_false_batch)

        # random numbers as negative samples
        # special_train_data.shape[2] == 6
        randfalse_maps = np.random.rand(c_randfalse_batch, note_group_size, special_train_data.shape[2])
        randfalse_labels = np.zeros(c_randfalse_batch)

        rn = np.random.randint(0, special_train_data.shape[0], (c_true_batch,))
        actual_train_data = np.concatenate((new_false_maps, randfalse_maps, special_train_data[rn]), axis=0)
        actual_train_labels = np.concatenate((new_false_labels, randfalse_labels, special_train_labels[rn]), axis=0)


        #history2 = classifier_model.fit(actual_train_data, actual_train_labels, epochs=c_multiplier, validation_split=0.2, verbose=0, callbacks=[])
        # Train the classifier gpt template
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
        res_noise = np.random.random((1, g_input_size))
        _resgenerated, res_map, _resclass = generator(res_noise)
        if plot_map:
            plot_current_map(torch.tensor(res_map, dtype=torch.float32))

        # early return if found a good solution
        # good is (inside the map boundary)
        if i >= good_epoch:
            current_map = res_map
            if inblock_trueness(current_map[:, :, 0:2]).numpy()[0] == 0 and inblock_trueness(current_map[:, :, 4:6]).numpy()[0] == 0:
                # debugging options to check map integrity
                # print(tf.reduce_mean(current_map))
                # print("-----MAPLAYER-----")
                # print(tf.reduce_mean(mapping_layer(conv_input(tf.convert_to_tensor(_resgenerated, dtype="float32"), extvar))))
                # print("-----CMWS-----")
                # print(tf.reduce_mean(construct_map_with_sliders(tf.convert_to_tensor(_resgenerated, dtype="float32"), extvar=mapping_layer.extvar)))
                break

    if plot_map:
        for i in range(3): # from our testing, any random input generates nearly the same map
            plot_noise = np.random.random((1, g_input_size))
            _plotgenerated, plot_mapped, _plotclass = generator(plot_noise)
            plot_current_map(torch.tensor(plot_mapped, dtype=torch.float32))

    return res_map.squeeze()




        # for i, (imgs, _) in enumerate(dataloader):#old
    
        #     # Adversarial ground truths
        #     valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        #     fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)
    
        #     # Configure input
        #     real_imgs = Variable(imgs.type(Tensor))
    
        #     # -----------------
        #     #  Train Generator
        #     # -----------------
    
        #     optimizer_G.zero_grad()
    
        #     # Sample noise as generator input
        #     z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
    
        #     # Generate a batch of images
        #     gen_imgs = generator(z)
    
        #     # Loss measures generator's ability to fool the discriminator
        #     g_loss = adversarial_loss(discriminator(gen_imgs), valid)
    
        #     g_loss.backward()
        #     optimizer_G.step()
    
        #     # ---------------------
        #     #  Train Discriminator
        #     # ---------------------
    
        #     optimizer_D.zero_grad()
    
        #     # Measure discriminator's ability to classify real from generated samples
        #     real_loss = adversarial_loss(discriminator(real_imgs), valid)
        #     fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        #     d_loss = (real_loss + fake_loss) / 2
    
        #     d_loss.backward()
        #     optimizer_D.step()
    
        #     print(
        #         "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
        #         % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        #     )
    
        #     batches_done = epoch * len(dataloader) + i
        #     if batches_done % opt.sample_interval == 0:
        #         save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

# ----------------------------------------------------------------
# gpt
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np

# def generate_set(models, begin=0, start_pos=[256, 192], group_id=-1, length_multiplier=1, plot_map=True):
#     extvar = {}
#     extvar["begin"] = begin
#     extvar["start_pos"] = start_pos
#     extvar["length_multiplier"] = length_multiplier
#     extvar["next_from_slider_end"] = GAN_PARAMS["next_from_slider_end"]

#     note_group_size = GAN_PARAMS["note_group_size"]
#     max_epoch = GAN_PARAMS["max_epoch"]
#     good_epoch = GAN_PARAMS["good_epoch"] - 1
#     g_multiplier = GAN_PARAMS["g_epochs"]
#     c_multiplier = GAN_PARAMS["c_epochs"]
#     g_batch = GAN_PARAMS["g_batch"]
#     g_input_size = GAN_PARAMS["g_input_size"]
#     c_true_batch = GAN_PARAMS["c_true_batch"]
#     c_false_batch = GAN_PARAMS["c_false_batch"]
#     c_randfalse_batch = GAN_PARAMS["c_randfalse_batch"]

#     reset_model_weights(models)
#     set_extvar(models, extvar)
#     gmodel, mapping_layer, classifier_model, mmodel, default_weights = models

#     criterion = nn.BCELoss()  # Binary Cross-Entropy Loss

#     for i in range(max_epoch):
#         gnoise = torch.rand(g_batch, g_input_size)
#         glabel = [torch.zeros((g_batch, note_group_size * 4)),
#                   torch.ones((g_batch,)),
#                   torch.ones((g_batch,))]
#         ginput = conv_input(gnoise, extvar)  # Define conv_input for PyTorch

#         # Fit mmodel instead of gmodel
#         optimizer = optim.Adam(mmodel.parameters())
#         for _ in range(g_multiplier):
#             optimizer.zero_grad()
#             goutput = mmodel(ginput)
#             loss = criterion(goutput[0], glabel[0]) + criterion(goutput[1], glabel[1]) + criterion(goutput[2], glabel[2])
#             loss.backward()
#             optimizer.step()

#         pred_noise = torch.rand(c_false_batch, g_input_size)
#         pred_input = conv_input(pred_noise, extvar)  # Define conv_input for PyTorch
#         predicted_maps_data, predicted_maps_mapped, _predclass = mmodel(pred_input)
#         new_false_maps = predicted_maps_mapped
#         new_false_labels = torch.zeros(c_false_batch)

#         randfalse_maps = torch.rand(c_randfalse_batch, note_group_size, special_train_data.shape[2])
#         randfalse_labels = torch.zeros(c_randfalse_batch)

#         rn = np.random.randint(0, special_train_data.shape[0], (c_true_batch,))
#         actual_train_data = torch.cat((new_false_maps, randfalse_maps, special_train_data[rn]), dim=0)
#         actual_train_labels = torch.cat((new_false_labels, randfalse_labels, special_train_labels[rn]), dim=0)

#         optimizer = optim.Adam(classifier_model.parameters())
#         for _ in range(c_multiplier):
#             optimizer.zero_grad()
#             coutput = classifier_model(actual_train_data)
#             c_loss = criterion(coutput, actual_train_labels)
#             c_loss.backward()
#             optimizer.step()

#         print("Group {}, Epoch {}: G loss: {} vs. C loss: {}".format(group_id, 1+i, g_loss.item(), c_loss.item()))

#         # Make a new set of notes
#         res_noise = torch.rand(1, g_input_size)
#         res_input = conv_input(res_noise, extvar)  # Define conv_input for PyTorch
#         _resgenerated, res_map, _resclass = mmodel(res_input)
#         if plot_map:
#             plot_current_map(res_map)

#         if i >= good_epoch:
#             current_map = res_map
#             if inblock_trueness(current_map[:, :, 0:2]).item() == 0 and inblock_trueness(current_map[:, :, 4:6]).item() == 0:
#                 break

#     if plot_map:
#         for _ in range(3):
#             plot_noise = torch.rand(1, g_input_size)
#             plot_input = conv_input(plot_noise, extvar)  # Define conv_input for PyTorch
#             _plotgenerated, plot_mapped, _plotclass = mmodel(plot_input)
#             plot_current_map(plot_mapped)

#     return res_map.squeeze()

# Define necessary functions, such as conv_input, plot_current_map, inblock_trueness
# as they were not provided in the original code and can vary based on your context.















def generate_map():
    """
    Generate the map (main function)
    dist_multiplier is used here
    """
    o = [];
    note_group_size = GAN_PARAMS["note_group_size"];
    pos = [np.random.randint(100, 412), np.random.randint(80, 304)];
    models = make_models();

    print("# of groups: {}".format(timestamps.shape[0] // note_group_size));
    for i in range(timestamps.shape[0] // note_group_size):
        z = generate_set(models, begin = i * note_group_size, start_pos = pos, length_multiplier = dist_multiplier, group_id = i, plot_map=False)[:, :6] * np.array([512, 384, 1, 1, 512, 384]);
        pos = z[-1, 0:2];
        o.append(z);
    a = np.concatenate(o, axis=0);
    return a;

def put_everything_in_the_center():
    o = [];
    print("max_epoch = 0: putting everyting in the center")
    for i in range(timestamps.shape[0]):
        z = [256, 192, 0, 0, 256 + slider_lengths[i], 192]
        o.append(z)
    a = np.array(o);
    return a;

def generate_test():
    """
    This is only used in debugging.
    Generates a test map with plotting on.
    """
    o = [];
    pos = [384, 288];
    note_group_size = GAN_PARAMS["note_group_size"];
    generate_set(begin = 3 * note_group_size, start_pos = pos, length_multiplier = dist_multiplier, group_id = 3, plot_map=True);

def print_osu_text(a):
    """
    This is only used in debugging.
    Prints .osu text directly.
    """
    for i, ai in enumerate(a):
        if not is_slider[i]:
            print("{},{},{},1,0,0:0:0".format(int(ai[0]), int(ai[1]), int(timestamps[i])));
        else:
            print("{},{},{},2,0,L|{}:{},1,{},0:0:0".format(int(ai[0]), int(ai[1]), int(timestamps[i]), int(round(ai[0] + ai[2] * slider_lengths[i])), int(round(ai[1] + ai[3] * slider_lengths[i])), int(slider_length_base[i] * slider_ticks[i])));


def step6_run_all(flow_dataset_npz = "flow_dataset.npz"):
    """
    Runs everything from building model to generating map.
    A lot of globals because currently it was directly cut from ipython notebook. Shouldn't hurt anything outside this file.
    """
    global objs, predictions, ticks, timestamps, is_slider, is_spinner, is_note_end, sv, slider_ticks, dist_multiplier, divisor, note_distance_basis, slider_length_base, slider_types, slider_type_rotation, slider_cos, slider_sin, slider_cos_each, slider_sin_each, slider_type_length, slider_lengths, tick_diff, note_distances, maps, labels, special_train_data, special_train_labels, early_stop, loss_ma, extvar, plot_noise

    objs, predictions, ticks, timestamps, is_slider, is_spinner, is_note_end, sv, slider_ticks, dist_multiplier = read_map_predictions("rhythm_data.npz");

    # get divisor from GAN_PARAMS
    divisor = GAN_PARAMS["divisor"];

    # get basis
    note_distance_basis = GAN_PARAMS["note_distance_basis"];

    # get next_from_slider_end
    next_from_slider_end = GAN_PARAMS["next_from_slider_end"];


    # should be slider length each tick, which is usually SV * SMP * 100 / 4
    # e.g. SV 1.6, timing section x1.00, 1/4 divisor, then slider_length_base = 40
    slider_length_base = sv / divisor;

    # weight for each type of sliders
    slider_type_probs = [0.25, 0.25, 0.25, 0.05, 0.05, 0.03, 0.03, 0.01, 0.01, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.015, 0.015, 0.01];
    slider_types = np.random.choice(len(slider_type_probs), is_slider.shape, p=slider_type_probs).astype(int);

    # these data must be kept consistent with the sliderTypes in load_map.js
    slider_type_rotation = np.array([0, -0.40703540572409336, 0.40703540572409336, -0.20131710837464062, 0.20131710837464062,
        -0.46457807316944644, 0.46457807316944644, 1.5542036732051032, -1.5542036732051032, 0, 0, 0.23783592745745077, -0.23783592745745077,
        0.5191461142465229, -0.5191461142465229, -0.16514867741462683, 0.16514867741462683, 3.141592653589793]);

    # this is vector length! I should change the variable name probably...
    slider_type_length = np.array([1.0, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.64, 0.64, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.96, 0.96, 0]);

    slider_cos = np.cos(slider_type_rotation);
    slider_sin = np.sin(slider_type_rotation);

    slider_cos_each = slider_cos[slider_types];
    slider_sin_each = slider_sin[slider_types];


    slider_lengths = np.array([slider_type_length[int(k)] * slider_length_base[i] for i, k in enumerate(slider_types)]) * slider_ticks;

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
        maps = flow_dataset["maps"];
        labels = np.ones(maps.shape[0]);

    order2 = np.argsort(np.random.random(maps.shape[0]));
    special_train_data = maps[order2];
    special_train_labels = labels[order2];

    # early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
    early_stop = None# not used
    # Start model training

    loss_ma = [90, 90, 90];
    extvar = {"begin": 10};

    plot_noise = np.random.random((1, GAN_PARAMS["g_input_size"]));

    if GAN_PARAMS["max_epoch"] == 0:
        osu_a = put_everything_in_the_center();
    else:
        osu_a = generate_map();

    data = objs, predictions, ticks, timestamps, is_slider, is_spinner, is_note_end, sv, slider_ticks, dist_multiplier, slider_types, slider_length_base;
    return osu_a, data;
