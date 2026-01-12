import math
import time

import torch
import torch.nn
import torch.nn.functional as F
from torchvision import transforms


class TransformerNetLight(torch.nn.Module):
    def __init__(self):
        super(TransformerNetLight, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        # self.res3 = ResidualBlock(128)
        # self.res4 = ResidualBlock(128)
        # self.res5 = ResidualBlock(128)

        # Upsampling Layers
        # self.deconv1 = ConvLayer(128, 256, kernel_size=3, stride=1)
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        # self.in4 = torch.nn.InstanceNorm2d(64, affine=True)

        # self.deconv2 = ConvLayer(64, 128, kernel_size=3, stride=1)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        # self.in5 = torch.nn.InstanceNorm2d(32, affine=True)

        # self.deconv3 = ConvLayer(32, 12, kernel_size=9, stride=2)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()
        # self.pixelshuffle = torch.nn.PixelShuffle(2)

    def forward(self, X):
        y = self.in1(self.conv1(X))
        y = self.in2(self.conv2(y))
        y = self.in3(self.conv3(y))

        y = self.res1(y)
        y = self.res2(y)

        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))

        y = self.deconv3(y)

        return y
    
    def preprocess_image(self, X) -> torch.Tensor:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
        return transform(X).unsqueeze(0)

    def process_image(self, X) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(X)[0]

    def process_image_motion_vectors(self, X, stylized_X, previous_stylized_X, motion_vectors, normal_map, depth_map) -> torch.Tensor:
        stylized_color = stylized_X / 255
        current_frame_color = X

        motion = motion_vectors[:, :2, :, :]

        B, _, H, W = stylized_color.shape
        device = stylized_color.device

        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
        base_grid = torch.stack((grid_x, grid_y), dim=0).float() # (2, H, W)

        previousUV = base_grid.unsqueeze(0) - motion

        norm_uv = previousUV.clone()
        norm_uv[:, 0] = (norm_uv[:, 0] / (W - 1) * 2.0) - 1.0
        norm_uv[:, 1] = (norm_uv[:, 1] / (H - 1) * 2.0) - 1.0
        norm_uv = norm_uv.permute(0, 2, 3, 1)

        motion_previous_stylized_color = F.grid_sample(previous_stylized_X.cpu(), norm_uv.cpu(), align_corners=True).to(device)

        depth = depth_map[:, 0:1, :, :]
        depth_weight = torch.lerp(torch.tensor(0.0, device=device), torch.tensor(0.2, device=device), depth)
        depth_weight_c = torch.lerp(torch.tensor(0.0, device=device), torch.tensor(0.4, device=device), depth)

        normal = normal_map * 2.0 - 1.0

        facing_ratio = normal[:, 2:3, :, :]
        facing_ratio = (facing_ratio + 1.0) * 0.5

        normal_intensity = torch.clamp(normal[:, 2:3, :, :], min=0.0) + 1.0
        
        current_frame_color = current_frame_color * normal_intensity

        stylized_new_color = torch.lerp(stylized_color, current_frame_color, depth_weight)
        output_color = torch.lerp(stylized_new_color, motion_previous_stylized_color, depth_weight_c)

        output_color = torch.clamp(output_color, 0.0, 1.0)

        return output_color


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out