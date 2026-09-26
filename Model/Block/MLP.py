import torch.nn as nn

from Model.utils import activation_resolver, norm_resolver


class MLP(nn.Module):
    def __init__(self, in_features, out_features=64,
                 norm_type: str = 'BatchNorm1d', act_name='ReLU', dropout=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.linear = nn.Linear(in_features, out_features)
        self.norm = norm_resolver(norm_type, out_features)
        self.act = activation_resolver(act_name)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.linear(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.dropout(out)
        return out
