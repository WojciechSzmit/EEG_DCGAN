import torch
import torch.nn as nn

class Generator1D(nn.Module):
    def __init__(self, noise_dim, channels_eeg, features_g, signal_length):
        super(Generator1D, self).__init__()
        self.net = nn.Sequential(
            self._block(noise_dim, features_g * 16, 4, 1, 0),
            self._block(features_g * 16, features_g * 8, 4, 2, 1),
            self._block(features_g * 8, features_g * 4, 4, 2, 1),
            self._block(features_g * 4, features_g * 2, 4, 2, 1),
            self._block(features_g * 2, features_g, 4, 2, 1),
            self._block(features_g, features_g // 2, 4, 2, 1),
            nn.ConvTranspose1d(features_g // 2, channels_eeg, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )
        self.signal_length = signal_length

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)