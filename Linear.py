import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, input_dim, output_dim=5):
        super(Linear, self).__init__()
        self.lin = nn.Sequential(
            nn.Linear(input_dim, 250, bias=True),
            nn.Tanh(),
            nn.Linear(250, 500, bias=True),
            nn.Tanh(),
            nn.Linear(500, 50, bias=True),
            nn.Tanh(),
            nn.Linear(50, output_dim, bias=True))

    def forward(self, inputs):
        return self.lin(inputs)
    
    def printHello(self):
        return "Hello"
