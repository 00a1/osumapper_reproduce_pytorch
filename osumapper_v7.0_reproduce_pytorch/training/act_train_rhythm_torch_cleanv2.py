# Part 2 action script

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

root = "mapdata/"
# set divisor
divisor = 4

# this is a global variable!
time_interval = 16

# lst file, [TICK, TIME, NOTE, IS_CIRCLE, IS_SLIDER, IS_SPINNER, IS_NOTE_END, UNUSED,SLIDING, SPINNING, MOMENTUM, EX1, EX2, EX3], length MAPTICKS
#               0,    1,    2,         3,         4,          5,           6,      7,      8,        9,       10,  11,  12,  13,
# wav file, [len(snapsize), MAPTICKS, 2, fft_size//4]

def read_npz(fn):
    with np.load(fn) as data:
        wav_data = data["wav"]
        wav_data = np.swapaxes(wav_data, 2, 3)
        train_data = wav_data
        div_source = data["lst"][:, 0]
        div_source2 = data["lst"][:, 11:14]
        div_data = np.concatenate([divisor_array(div_source), div_source2], axis=1)
        lst_data = data["lst"][:, 2:10]
        lst_data = 2 * lst_data - 1
        train_labels = lst_data
    return train_data, div_data, train_labels

def divisor_array(t):
    d_range = list(range(0, divisor))
    return np.array([[int(k % divisor == d) for d in d_range] for k in t])

def read_npz_list():
    npz_list = []
    for file in os.listdir(root):
        if file.endswith(".npz"):
            npz_list.append(os.path.join(root, file))
    return npz_list

def prefilter_data(train_data_unfiltered, div_data_unfiltered, train_labels_unfiltered):
    # Filter out slider ends from the training set, since we cannot reliably decide if a slider end is on a note.
    # Another way is to set 0.5 for is_note value, but that will break the validation algorithm.
    # Also remove the IS_SLIDER_END, IS_SPINNER_END columns which are left to be zeros.

    # Before: IS_NOTE_START, IS_CIRCLE, IS_SLIDER, IS_SPINNER, IS_NOTE_END, UNUSED, SLIDING, SPINNING
    #                     0,         1,         2,          3,           4,      5,       6,        7
    # After:  IS_NOTE_START, IS_CIRCLE, IS_SLIDER, IS_SPINNER, IS_NOTE_END, UNUSED
    #                     0,         1,         2,          3,           4,      5

    non_object_end_indices = [i for i,k in enumerate(train_labels_unfiltered) if True or k[4] == -1 and k[5] == -1]
    train_data = train_data_unfiltered[non_object_end_indices]
    div_data = div_data_unfiltered[non_object_end_indices]
    train_labels = train_labels_unfiltered[non_object_end_indices][:, [0, 1, 2, 3, 4]]

    # should be (X, 7, 32, 2) and (X, 6) in default sampling settings
    # (X, fft_window_type, freq_point, magnitude/phase)
    return train_data, div_data, train_labels

def get_data_shape():
    for file in os.listdir(root):
        if file.endswith(".npz"):
            train_data_unfiltered, div_data_unfiltered, train_labels_unfiltered = read_npz(os.path.join(root, file))
            train_data, div_data, train_labels = prefilter_data(train_data_unfiltered, div_data_unfiltered, train_labels_unfiltered)
            # should be (X, 7, 32, 2) and (X, 6) in default sampling settings
            # (X, fft_window_type, freq_point, magnitude/phase)
            # X = 76255
            # print(train_data.shape, train_labels.shape)
            if train_data.shape[0] == 0:
                continue
            return train_data.shape, div_data.shape, train_labels.shape
    print("cannot find npz!! using default shape")
    return (-1, 7, 32, 2), (-1, 3 + divisor), (-1, 5)

def read_some_npzs_and_preprocess(npz_list):
    train_shape, div_shape, label_shape = get_data_shape()
    td_list = []
    dd_list = []
    tl_list = []
    for fp in npz_list:
        if fp.endswith(".npz"):
            _td, _dd, _tl = read_npz(fp)
            if _td.shape[1:] != train_shape[1:]:
                print("Warning: something wrong found in {}! shape = {}".format(fp, _td.shape))
                continue
            td_list.append(_td)
            dd_list.append(_dd)
            tl_list.append(_tl)
    train_data_unfiltered = np.concatenate(td_list)
    div_data_unfiltered = np.concatenate(dd_list)
    train_labels_unfiltered = np.concatenate(tl_list)

    train_data2, div_data2, train_labels2 = prefilter_data(train_data_unfiltered, div_data_unfiltered, train_labels_unfiltered)
    return train_data2, div_data2, train_labels2

def train_test_split(train_data2, div_data2, train_labels2, test_split_count=233):
    """
    Split data into train and test.
    Note that there is no randomization. It doesn't really matter here, but in other machine learning it's obligatory.
    Requires at least 233 rows of data or it will throw an error. (Tick count/10, around 1.5-2 full length maps)
    """
    new_train_data = train_data2[:-test_split_count]
    new_div_data = div_data2[:-test_split_count]
    new_train_labels = train_labels2[:-test_split_count]
    test_data = train_data2[-test_split_count:]
    test_div_data = div_data2[-test_split_count:]
    test_labels = train_labels2[-test_split_count:]
    return (new_train_data, new_div_data, new_train_labels), (test_data, test_div_data, test_labels)

def set_param_fallback(PARAMS):
    try:
        divisor = PARAMS["divisor"]
    except:
        divisor = 4
    if "verbose" not in PARAMS:
        PARAMS["verbose"] = False
    if "train_epochs" not in PARAMS:
        PARAMS["train_epochs"] = 16
    if "train_epochs_many_maps" not in PARAMS:
        PARAMS["train_epochs_many_maps"] = 6
    if "too_many_maps_threshold" not in PARAMS:
        PARAMS["too_many_maps_threshold"] = 200
    if "data_split_count" not in PARAMS:
        PARAMS["data_split_count"] = 80
    if "plot_history" not in PARAMS:
        PARAMS["plot_history"] = True
    if "train_batch_size" not in PARAMS:
        PARAMS["train_batch_size"] = None
    return PARAMS


class Model(nn.Module):
    def __init__(self, input_shape, div_shape, label_shape):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[1], 16, kernel_size=(2, 2))
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=(0, 1))
        self.dropout1 = nn.Dropout(0.3)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=(1, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=(0, 1))
        self.dropout2 = nn.Dropout(0.3)
        self.flatten = nn.Flatten()
        self.lstm = nn.LSTM(input_size=496, hidden_size=64, batch_first=True)
        self.fc1 = nn.Linear(71, 71)
        self.fc2 = nn.Linear(71, 71)
        self.fc3 = nn.Linear(71, label_shape[1])

    def forward(self, wav_data, div_data):
        x = self.conv1(wav_data)
        x = torch.relu(self.pool1(x))
        x = self.dropout1(x)
        x = self.conv2(x)
        x = torch.relu(self.pool2(x))
        x = self.dropout2(x)
        x = self.flatten(x)
        lstm_input = x.view(x.size(0), -1)
        lstm_input = lstm_input.unsqueeze(1).repeat(1, time_interval, 1)
        lstm_out, _ = self.lstm(lstm_input)
        div_data = div_data.view(div_data.size(0), -1)
        concatenated = torch.cat((lstm_out[:, -1, :], div_data), dim=1)
        x = self.fc1(concatenated)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


# def plot_history(history):
#     plt.figure()
#     plt.xlabel('Epoch')
#     plt.ylabel('Mean Abs Error [Limitless]')
#     plt.plot(history["epoch"], np.array(history["loss"]), label='Train Loss')
#     plt.legend()
#     plt.show()

def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [Limitless]')
    plt.plot(history["epoch"], np.array(history["train_mae"]), label='Train MAE')
    plt.plot(history["epoch"], np.array(history["val_mae"]), label='Val MAE')
    plt.plot(history["epoch"], np.array(history["train_loss"]), label='Train Loss')
    plt.plot(history["epoch"], np.array(history["val_loss"]), label='Val Loss')
    plt.legend()
    plt.show()

def step2_build_model():
    train_shape, div_shape, label_shape = get_data_shape()
    model_v7 = Model(train_shape, div_shape, label_shape).to(device)
    print("successfully built model")
    return model_v7

# def step2_train_model(model, PARAMS):
#     global new_train_data, new_div_data, new_train_labels, test_data, test_div_data, test_labels
#     PARAMS = set_param_fallback(PARAMS)
#     train_file_list = read_npz_list()

#     # Don't worry, it will successfully overfit after those 16 epochs.
#     EPOCHS = PARAMS["train_epochs"]
#     too_many_maps_threshold = PARAMS["too_many_maps_threshold"]
#     data_split_count = PARAMS["data_split_count"]
#     batch_size = PARAMS["train_batch_size"]# old 32
#     history = {"epoch": [], "loss": []}

#     # Store training stats
#     criterion = nn.MSELoss()
#     optimizer = optim.RMSprop(model.parameters(), lr=0.001)

#     # if there is too much data, reduce epoch count
#     if len(train_file_list) >= too_many_maps_threshold:
#         EPOCHS = PARAMS["train_epochs_many_maps"]

#     if len(train_file_list) < too_many_maps_threshold:
#         train_data2, div_data2, train_labels2 = read_some_npzs_and_preprocess(train_file_list)

#         # Split some test data out
#         (new_train_data, new_div_data, new_train_labels), (test_data, test_div_data, test_labels) = train_test_split(train_data2, div_data2, train_labels2)
        
#         # new_train_data_split = np.array_split(new_train_data, batch_size)
#         # new_div_data_split = np.array_split(new_div_data, batch_size)
#         # new_train_labels_split = np.array_split(new_train_labels, batch_size)
#         train_dataset = TensorDataset(torch.tensor(new_train_data, dtype=torch.float32, device=device), torch.tensor(new_div_data, dtype=torch.float32, device=device), torch.tensor(new_train_labels, dtype=torch.float32, device=device))
#         train_loader = DataLoader(train_dataset, batch_size=batch_size)

#         for epoch in tqdm(range(EPOCHS), desc="Epoch"):
#             # total_loss = 0.0
#             # for batch in tqdm(train_loader, desc="Batch", position=1, leave=True):
#             for batch in train_loader:
#             # for batch_idx in range(batch_size):
#                 optimizer.zero_grad()
#                 new_train_data_batch, new_div_data_batch, new_train_labels_batch = batch
#                 outputs = model(new_train_data_batch, new_div_data_batch)
#                 loss = criterion(outputs, new_train_labels_batch)
#                 # outputs = model(torch.tensor(new_train_data_split[batch_idx], dtype=torch.float32, device=device), torch.tensor(new_div_data_split[batch_idx], dtype=torch.float32, device=device))
#                 # loss = criterion(outputs, torch.tensor(new_train_labels_split[batch_idx], dtype=torch.float32, device=device))
#                 loss.backward()
#                 # total_loss += loss.item()
#                 optimizer.step()
#                 if PARAMS["verbose"]:
#                     print("loss: " + str(loss.item()))

#             # epoch_loss = total_loss / batch_size
#             history["epoch"].append(epoch)
#             # history["loss"].append(epoch_loss)
#             history["loss"].append(loss.item())
                
#         if PARAMS["plot_history"]:
#             plot_history(history)
#         if not PARAMS["verbose"]:
#             print("final loss: " + str(loss.item()))
    
#     else:# too much map data! read it every turn. UPDATE CODE
#         for _ in tqdm(range(EPOCHS), desc="Epoch", position=0, leave=True):
#             for map_batch in range(np.ceil(len(train_file_list) / data_split_count).astype(int)):
#                 if map_batch == 0:
#                     train_data2, div_data2, train_labels2 = read_some_npzs_and_preprocess(train_file_list[map_batch * data_split_count : (map_batch+1) * data_split_count])
#                     (new_train_data, new_div_data, new_train_labels), (test_data, test_div_data, test_labels) = train_test_split(train_data2, div_data2, train_labels2)
#                 else:
#                     new_train_data, new_div_data, new_train_labels = read_some_npzs_and_preprocess(train_file_list[map_batch * data_split_count : (map_batch+1) * data_split_count])

#                 train_dataset = TensorDataset(torch.tensor(new_train_data, dtype=torch.float32, device=device), torch.tensor(new_div_data, dtype=torch.float32, device=device), torch.tensor(new_train_labels, dtype=torch.float32, device=device))
#                 train_loader = DataLoader(train_dataset, batch_size=batch_size)

#                 for batch in tqdm(train_loader, desc="Batch", position=1, leave=True):
#                     optimizer.zero_grad()
#                     new_train_data_batch, new_div_data_batch, new_train_labels_batch = batch
#                     outputs = model(new_train_data_batch, new_div_data_batch)
#                     loss = criterion(outputs, new_train_labels_batch)
#                     loss.backward()
#                     optimizer.step()
#                     if PARAMS["verbose"]:
#                         print("loss: " + str(loss.item()))
#         if not PARAMS["verbose"]:
#             print("final loss: " + str(loss.item()))
#     return model


def trysmaller(model, PARAMS):
    global new_train_data, new_div_data, new_train_labels, test_data, test_div_data, test_labels
    PARAMS = set_param_fallback(PARAMS)
    train_file_list = read_npz_list()
    train_data2, div_data2, train_labels2 = read_some_npzs_and_preprocess(train_file_list)
    (new_train_data, new_div_data, new_train_labels), (test_data, test_div_data, test_labels) = train_test_split(train_data2, div_data2, train_labels2)
    new_train_data = torch.tensor(new_train_data, dtype=torch.float32, device=device)
    new_div_data = torch.tensor(new_div_data, dtype=torch.float32, device=device)
    new_train_labels = torch.tensor(new_train_labels, dtype=torch.float32, device=device)

    gpu_memory = torch.cuda.get_device_properties(0).total_memory
    train_batch_size = int((gpu_memory * 0.4) / (new_train_data.element_size() * new_train_data.nelement()))
    print(train_batch_size)
    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    EPOCHS = PARAMS["train_epochs"]

    for _ in tqdm(range(EPOCHS), desc="Epoch"):
        optimizer.zero_grad()
        outputs = model(new_train_data, new_div_data)
        loss = criterion(outputs, new_train_labels)
        loss.backward()
        print(loss.item())
        optimizer.step()


def trysmallerbatch(model, PARAMS):
    global new_train_data, new_div_data, new_train_labels, test_data, test_div_data, test_labels
    PARAMS = set_param_fallback(PARAMS)
    train_file_list = read_npz_list()
    data_split_count = PARAMS["data_split_count"]
    EPOCHS = PARAMS["train_epochs"]
    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    for _ in tqdm(range(EPOCHS), desc="Epoch"):
        for map_batch in range(np.ceil(len(train_file_list) / data_split_count).astype(int)):
            if map_batch == 0:
                train_data2, div_data2, train_labels2 = read_some_npzs_and_preprocess(train_file_list[map_batch * data_split_count : (map_batch+1) * data_split_count])
                (new_train_data, new_div_data, new_train_labels), (test_data, test_div_data, test_labels) = train_test_split(train_data2, div_data2, train_labels2)
            else:
                new_train_data, new_div_data, new_train_labels = read_some_npzs_and_preprocess(train_file_list[map_batch * data_split_count : (map_batch+1) * data_split_count])
            
            new_train_data = torch.tensor(new_train_data, dtype=torch.float32, device=device)
            new_div_data = torch.tensor(new_div_data, dtype=torch.float32, device=device)
            new_train_labels = torch.tensor(new_train_labels, dtype=torch.float32, device=device)
            
            optimizer.zero_grad()
            outputs = model(new_train_data, new_div_data)
            loss = criterion(outputs, new_train_labels)
            loss.backward()
            print(loss.item())
            optimizer.step()

def step2_train_model(model, PARAMS):
    global new_train_data, new_div_data, new_train_labels, test_data, test_div_data, test_labels
    PARAMS = set_param_fallback(PARAMS)
    train_file_list = read_npz_list()

    # Don't worry, it will successfully overfit after those 16 epochs.
    EPOCHS = PARAMS["train_epochs"]
    too_many_maps_threshold = PARAMS["too_many_maps_threshold"]
    data_split_count = PARAMS["data_split_count"]
    batch_size = PARAMS["train_batch_size"]# old 32
    history = {"epoch": [], "train_mae": [], "val_mae": [], "train_loss": [], "val_loss": []}

    # Store training stats
    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)

    # if there is too much data, reduce epoch count
    if len(train_file_list) >= too_many_maps_threshold:
        EPOCHS = PARAMS["train_epochs_many_maps"]

    if len(train_file_list) < too_many_maps_threshold:
        train_data2, div_data2, train_labels2 = read_some_npzs_and_preprocess(train_file_list)

        # Split some test data out
        (new_train_data, new_div_data, new_train_labels), (test_data, test_div_data, test_labels) = train_test_split(train_data2, div_data2, train_labels2)

        # validation_split=0.2
        val_size = int(0.2 * len(new_train_data))
        train_data3, val_data3 = new_train_data[val_size:], new_train_data[:val_size]
        train_div_data3, val_div_data3 = new_div_data[val_size:], new_div_data[:val_size]
        train_labels3, val_labels3 = new_train_labels[val_size:], new_train_labels[:val_size]

        train_dataset = TensorDataset(torch.tensor(train_data3, dtype=torch.float32, device=device), torch.tensor(train_div_data3, dtype=torch.float32, device=device), torch.tensor(train_labels3, dtype=torch.float32, device=device))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True)

        for epoch in tqdm(range(EPOCHS), desc="Epoch"):
            total_loss = 0.0
            total_mae = 0.0
            # for batch in tqdm(train_loader, desc="Batch", position=1, leave=True):
            for batch in train_loader:
                optimizer.zero_grad()
                new_train_data_batch, new_div_data_batch, new_train_labels_batch = batch
                outputs = model(new_train_data_batch, new_div_data_batch)
                loss = criterion(outputs, new_train_labels_batch)
                loss.backward()
                total_loss += loss.item()
                optimizer.step()
        
                # Calculate batch MAE
                batch_mae = torch.mean(torch.abs(outputs - new_train_labels_batch)).item()
                total_mae += batch_mae
            
            if PARAMS["verbose"]:
                print("loss: " + str(loss.item()))
        
            epoch_loss = total_loss / batch_size
            epoch_mae = total_mae / batch_size
            history["epoch"].append(epoch)
            history["train_loss"].append(epoch_loss)
            # history["loss"].append(loss.item())
            history["train_mae"].append(epoch_mae)
        
            # Validation phase
            with torch.no_grad():
                val_outputs = model(torch.tensor(val_data3, dtype=torch.float32, device=device), torch.tensor(val_div_data3, dtype=torch.float32, device=device))
                val_loss = criterion(val_outputs, torch.tensor(val_labels3, dtype=torch.float32, device=device))
                # Calculate MAE
                val_mae = torch.mean(torch.abs(val_outputs - torch.tensor(val_labels3, dtype=torch.float32, device=device)))
                history["val_loss"].append(val_loss.item())
                history["val_mae"].append(val_mae.item())
        
            # Early stopping logic
            if len(history["val_loss"]) > 20 and np.mean(history["val_loss"][-20:]) < min(history["val_loss"]):
                break
                
        if PARAMS["plot_history"]:
            plot_history(history)
        if not PARAMS["verbose"]:
            print("final loss: " + str(loss.item()))
    
    else:# too much map data! read it every turn. UPDATE CODE
        for _ in tqdm(range(EPOCHS), desc="Epoch", position=0, leave=True):
            for map_batch in range(np.ceil(len(train_file_list) / data_split_count).astype(int)):
                if map_batch == 0:
                    train_data2, div_data2, train_labels2 = read_some_npzs_and_preprocess(train_file_list[map_batch * data_split_count : (map_batch+1) * data_split_count])
                    (new_train_data, new_div_data, new_train_labels), (test_data, test_div_data, test_labels) = train_test_split(train_data2, div_data2, train_labels2)
                else:
                    new_train_data, new_div_data, new_train_labels = read_some_npzs_and_preprocess(train_file_list[map_batch * data_split_count : (map_batch+1) * data_split_count])

                train_dataset = TensorDataset(torch.tensor(new_train_data, dtype=torch.float32, device=device), torch.tensor(new_div_data, dtype=torch.float32, device=device), torch.tensor(new_train_labels, dtype=torch.float32, device=device))
                train_loader = DataLoader(train_dataset, batch_size=batch_size)

                for batch in tqdm(train_loader, desc="Batch", position=1, leave=True):
                    optimizer.zero_grad()
                    new_train_data_batch, new_div_data_batch, new_train_labels_batch = batch
                    outputs = model(new_train_data_batch, new_div_data_batch)
                    loss = criterion(outputs, new_train_labels_batch)
                    loss.backward()
                    optimizer.step()
                    if PARAMS["verbose"]:
                        print("loss: " + str(loss.item()))
        if not PARAMS["verbose"]:
            print("final loss: " + str(loss.item()))
    return model

def step2_evaluate(model):
    """
    Evaluate model using AUC score.
    Previously I used F1 but I think AUC is more appropriate for this type of data.

    High value (close to 1.00) doesn't always mean it's better. Usually it means you put identical maps in the training set.
    It shouldn't be possible to reach very high accuracy since that will mean that music 100% dictates map rhythm.
    """
    model.eval()
    _train_shape, _div_shape, label_shape = get_data_shape()

    with torch.no_grad():
        test_predictions = model(torch.tensor(test_data, dtype=torch.float32, device=device), torch.tensor(test_div_data, dtype=torch.float32, device=device))

    flat_test_preds = test_predictions.cpu().numpy().reshape(-1, label_shape[1])
    flat_test_labels = test_labels.reshape(-1, label_shape[1])

    pred_result = (flat_test_preds + 1) / 2
    actual_result = (flat_test_labels + 1) / 2

    # Individual column predictions
    column_names = ["is_note_start", "is_circle", "is_slider", "is_spinner", "is_note_end"]
    for i, k in enumerate(column_names):
        if i == 3:  # No one uses spinners anyways
            continue
        if i == 2 and np.sum(actual_result[:, i]) == 0:  # No sliders (Taiko)
            continue
        auc_score = roc_auc_score(actual_result[:, i], pred_result[:, i])
        print("{} auc score: {}".format(k, auc_score))

def step2_save(model):
    torch.save(model.state_dict(), "saved_rhythm_model.pth")
