import logging

import torch

logger = logging.getLogger(__name__)

class TinyModel(torch.nn.Module):
    """A sample model.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TinyModel, self).__init__()

        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_dim, output_dim)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x
