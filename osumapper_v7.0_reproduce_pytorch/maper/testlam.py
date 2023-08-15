import torch
import torch.nn as nn

class LambdaLayer(nn.Module):
    def __init__(self):
        super(LambdaLayer, self).__init__()

    def forward(self, x):
        return (x + 1) / 2

# Create an instance of the LambdaLayer
lambda_layer = LambdaLayer()

# Test the LambdaLayer
input_tensor = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
output_tensor = lambda_layer(input_tensor)
print(output_tensor)#tensor([[-0.5000,  0.0000,  0.5000,  1.0000,  1.5000]])
