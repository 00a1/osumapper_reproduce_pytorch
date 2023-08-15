import torch
import torch.nn as nn
import torch.optim as optim

class ClassifierModel(nn.Module):
    def __init__(self, input_size):
        super(ClassifierModel, self).__init__()
        self.rnn_layer = nn.RNN(input_size, 64, batch_first=True)
        self.dense_layer1 = nn.Linear(64, 64)
        self.dense_layer2 = nn.Linear(64, 64)
        self.dense_layer3 = nn.Linear(64, 64)
        self.dense_layer4 = nn.Linear(64, 1)
        self.output_activation = nn.Tanh()

    def forward(self, x):
        rnn_output, _ = self.rnn_layer(x)
        dense1 = self.dense_layer1(rnn_output)
        dense2 = torch.relu(self.dense_layer2(dense1))
        dense3 = torch.tanh(self.dense_layer3(dense2))
        dense4 = torch.relu(self.dense_layer4(dense3))
        output = self.output_activation(dense4)
        output = (output + 1) / 2
        return output

# Create an instance of the PyTorch model
input_size = special_train_data.shape[2]
model = ClassifierModel(input_size)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Assuming you have your training data as a PyTorch tensor
# train_data_tensor = ...

# Convert the input data to the appropriate shape (batch_size, sequence_length, input_size)
train_data_tensor = train_data_tensor.permute(0, 2, 1)

# Forward pass
outputs = model(train_data_tensor)

# Calculate the loss and perform backpropagation
loss = criterion(outputs, target_tensor)
optimizer.zero_grad()
loss.backward()
optimizer.step()


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Generate some sample data
# Replace this with your actual data
num_samples = 1000
sequence_length = 10
input_size = 32
special_train_data = np.random.rand(num_samples, sequence_length, input_size)

# Define the RNN model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        rnn_output, _ = self.rnn(x)
        output = self.fc(rnn_output)
        return output

# Set random seed for reproducibility
torch.manual_seed(42)

# Define model hyperparameters
hidden_size = 64
output_size = 1
learning_rate = 0.001
num_epochs = 10

# Create the model
model = RNNModel(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Convert data to PyTorch tensors
input_data = torch.tensor(special_train_data, dtype=torch.float32)
target_data = torch.rand(num_samples, sequence_length, output_size, dtype=torch.float32)

# Training loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(input_data)
    loss = criterion(outputs, target_data)

    # Backpropagation and optimization
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training completed!")

# You can now use the trained model for predictions or further tasks.





#----------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim

class GenerativeModel(nn.Module):
    def __init__(self, in_params, out_params):
        super(GenerativeModel, self).__init__()

        self.fc1 = nn.Linear(in_params, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, out_params)
        self.activation1 = nn.ELU()
        self.activation2 = nn.ReLU()
        self.activation3 = nn.Tanh()
        self.activation4 = nn.ReLU()
        self.output_activation = nn.Tanh()

    def forward(self, x):
        fc1_output = self.fc1(x)
        activation1_output = self.activation1(fc1_output)
        fc2_output = self.fc2(activation1_output)
        activation2_output = self.activation2(fc2_output)
        fc3_output = self.fc3(activation2_output)
        activation3_output = self.activation3(fc3_output)
        fc4_output = self.fc4(activation3_output)
        activation4_output = self.activation4(fc4_output)
        output = self.fc5(activation4_output)
        
        output = self.output_activation(output)

        return output

# Define input and output dimensions
in_params = 10
out_params = 6

# Create an instance of the PyTorch model
pytorch_model = GenerativeModel(in_params, out_params)

# Define optimizer
optimizer = optim.Adam(pytorch_model.parameters(), lr=0.002)

# Loss function
criterion = nn.MSELoss()

# Create some example input data
input_data = torch.randn(3, in_params)  # Batch size of 3, input features of in_params

# Forward pass
output = pytorch_model(input_data)

print("Output shape:", output.shape)

# Compile the PyTorch model (not exactly the same as Keras)
# You will need to define optimizer and loss function separately
# and call these during the training loop.