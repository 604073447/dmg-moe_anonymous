import torch.nn as nn


# Maps one pooled layer feature to an age prediction.
class BasicHead(nn.Module):
    def __init__(self, hid: int, in_channels: int):
        super().__init__()
        self.hid = hid
        self.in_channels = in_channels
        self.out_channels = (1,)
        self.reg = nn.Linear(in_channels, 1)

    # Produces the age prediction for one expert depth.
    def forward(self, x):
        reg_out = self.reg(x)
        return {'age': reg_out[:, 0]}
