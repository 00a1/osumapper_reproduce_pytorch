#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import torch.nn as nn
import torch.optim as optim

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y

class PyTorchCustomModel(nn.Module):
    def __init__(self, train_shape, div_shape, label_shape, time_interval):
        super(PyTorchCustomModel, self).__init__()

        self.model1 = nn.Sequential(
            TimeDistributed(nn.Conv2d(train_shape[1], 16, kernel_size=(2, 2))),
            TimeDistributed(nn.MaxPool2d(kernel_size=(1, 2))),
            TimeDistributed(nn.ReLU()),
            TimeDistributed(nn.Dropout(0.3)),
            TimeDistributed(nn.Conv2d(16, 16, kernel_size=(2, 3))),
            TimeDistributed(nn.MaxPool2d(kernel_size=(1, 2))),
            TimeDistributed(nn.ReLU()),
            TimeDistributed(nn.Dropout(0.3)),
            TimeDistributed(nn.Flatten()),
            nn.LSTM(64, batch_first=True, bidirectional=False)
        )

        self.input2 = nn.Identity()

        self.concat_layer = nn.Linear(64 + div_shape[1], 71)
        self.dense1 = nn.Linear(71, 71)
        self.dense2 = nn.Linear(71, label_shape[1])

    def forward(self, x1, x2):
        out1 = self.model1(x1)
        out2 = self.input2(x2)
        concat = torch.cat((out1, out2), dim=2)
        dense1 = torch.tanh(self.concat_layer(concat))
        dense2 = torch.relu(self.dense1(dense1))
        output = torch.tanh(self.dense2(dense2))
        return output

# Replace these values with your data shapes
train_shape = (batch_size, time_interval, channels, height, width)
div_shape = (batch_size, time_interval, div_features)
label_shape = (batch_size, output_features)
time_interval = ...

# Create an instance of the PyTorchCustomModel
custom_model = PyTorchCustomModel(train_shape, div_shape, label_shape, time_interval)

# Define the loss function and optimizer
loss_func = nn.MSELoss()
optimizer = optim.RMSprop(custom_model.parameters(), lr=0.001)

# You can train and use the model as follows (assuming you have input1_data, input2_data, and target_data tensors):
# optimizer.zero_grad()
# outputs = custom_model(input1_data, input2_data)
# loss = loss_func(outputs, target_data)
# loss.backward()
# optimizer.step()