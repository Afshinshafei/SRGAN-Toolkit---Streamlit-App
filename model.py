"""SRGAN Model Architecture."""
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import h5py
import pickle
import os
from scipy.interpolate import RectBivariateSpline
from utils import IMG_HEIGHT, IMG_WIDTH, open_netcdf_robust


class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(64, 64, 3, stride=1, padding=0),
            nn.BatchNorm2d(64)
        )

    def forward(self, x):
        return x + self.layers(x)


class Generator(nn.Module):
    def __init__(self, num_channels, num_res_blocks=8, scale_factor=8):
        super().__init__()
        self.num_res_blocks = num_res_blocks

        self.initial_conv = nn.Sequential(
            nn.ReplicationPad2d(4),
            nn.Conv2d(num_channels, 64, kernel_size=9, stride=1, padding=0),
            nn.PReLU()
        )

        self.resBlocks = nn.ModuleList([ResidualBlock(64) for i in range(self.num_res_blocks)])

        self.post_resid_conv = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(64, 64, 3, stride=1, padding=0),
            nn.BatchNorm2d(64)
        )

        self.conv_prelu = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(64, 64, 3, stride=1, padding=0),
            nn.PReLU()
        )

        self.final_conv = nn.Sequential(
            nn.ReplicationPad2d(4),
            nn.Conv2d(64, 2, 9, stride=1, padding=0)
        )

    def forward(self, x):
        initial_conv_out = self.initial_conv(x)
        res_block_out = self.resBlocks[0](initial_conv_out)
        for i in range(1, self.num_res_blocks):
            res_block_out = self.resBlocks[i](res_block_out)
        post_resid_conv_out = self.post_resid_conv(res_block_out) + initial_conv_out
        conv_prelu_out = self.conv_prelu(post_resid_conv_out)
        final_out = self.final_conv(conv_prelu_out)
        return torch.tanh(final_out)


class InferenceDataset(data.Dataset):
    def __init__(self, lr_data_file, elevation_file, stats_file, base_path='.'):
        # Load LR dataset
        self.lr_data_file = h5py.File(os.path.join(base_path, lr_data_file) if base_path != '.' else lr_data_file, 'r')
        self.lr_pr_data = self.lr_data_file['pr']
        self.lr_tas_data = self.lr_data_file['tas']

        # Load statistics for normalization
        stats_path = os.path.join(base_path, stats_file) if base_path != '.' else stats_file
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
        self.lr_in_mean = np.array([stats['pr_mean'], stats['tas_mean']])
        self.lr_in_std = np.array([stats['pr_std'], stats['tas_std']])

        # Load and process elevation data
        elev_path = os.path.join(base_path, elevation_file) if base_path != '.' else elevation_file
        nc_data = open_netcdf_robust(elev_path)
        elevation_data = nc_data['HSURF'].isel(time=0).squeeze()
        flipped_elevation = np.flipud(elevation_data.values)
        elev_mean = flipped_elevation.mean()
        elev_std = flipped_elevation.std()
        normalized_elevation = (flipped_elevation - elev_mean) / elev_std
        c, h2, w2 = (1,) + normalized_elevation.shape
        x = np.arange(h2)
        y = np.arange(w2)
        xnew = np.linspace(0, h2 - 1, IMG_HEIGHT)
        ynew = np.linspace(0, w2 - 1, IMG_WIDTH)
        interpolated_elevation = np.zeros((c, IMG_HEIGHT, IMG_WIDTH))
        f = RectBivariateSpline(x, y, normalized_elevation)
        interpolated_elevation[0, :, :] = f(xnew, ynew)
        self.elevation = interpolated_elevation

    def __getitem__(self, index):
        # Low-Resolution (LR) data
        pr_lr = self.lr_pr_data[index]
        tas_lr = self.lr_tas_data[index]

        # Apply log transformation to precipitation
        pr_lr = np.log1p(pr_lr)

        # Concatenate LR data before flipping
        low_res = np.concatenate((pr_lr[np.newaxis, :, :], tas_lr[np.newaxis, :, :]), axis=0)

        # Flip data along the same axis as in training for FCN
        low_res = np.flip(low_res, axis=1).copy()

        # Interpolate low-res to high-res dimensions
        c, h2, w2 = low_res.shape
        x = np.arange(h2)
        y = np.arange(w2)
        xnew = np.linspace(0, h2 - 1, IMG_HEIGHT)
        ynew = np.linspace(0, w2 - 1, IMG_WIDTH)
        interpolated_low_res = np.zeros((c, IMG_HEIGHT, IMG_WIDTH))
        for i in range(c):
            f = RectBivariateSpline(x, y, low_res[i, :, :])
            interpolated_low_res[i, :, :] = f(xnew, ynew)

        eps = 1e-9  # Small epsilon to avoid division by zero

        # Normalize LR images using their respective statistics
        interpolated_low_res[0] = (interpolated_low_res[0] - self.lr_in_mean[0]) / (self.lr_in_std[0] + eps)  # Normalize pr
        interpolated_low_res[1] = (interpolated_low_res[1] - self.lr_in_mean[1]) / (self.lr_in_std[1] + eps)  # Normalize tas

        # Scale normalized low-res to [-1, 1]
        low_res_min = np.amin(interpolated_low_res, axis=(1, 2), keepdims=True)
        low_res_max = np.amax(interpolated_low_res, axis=(1, 2), keepdims=True)
        interpolated_low_res = 2 * (interpolated_low_res - low_res_min) / (low_res_max - low_res_min + eps) - 1

        # Add the elevation data to the input image
        input_data = np.concatenate([interpolated_low_res, self.elevation], axis=0)

        input_data_tensor = torch.from_numpy(input_data).float()

        return input_data_tensor

    def __len__(self):
        return len(self.lr_pr_data)

    def close(self):
        self.lr_data_file.close()

