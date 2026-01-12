import torch

class DepthwiseSeparableConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DepthwiseSeparableConv, self).__init__()
        # Optimized: Using internal padding instead of a separate ReflectionPad2d
        # to allow for better kernel fusion in CUDA/MKL
        padding = kernel_size // 2
        
        self.depthwise = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size, 
            stride=stride, groups=in_channels, padding=padding, padding_mode='reflect', bias=False
        )
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        self.conv = DepthwiseSeparableConv(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        return self.conv(x)

class TransformerNetFusion_separated_scaled(torch.nn.Module):
    def __init__(self, alpha=1.0):
        super(TransformerNetFusion_separated_scaled, self).__init__()
        
        # Helper function to scale channels
        def scale(ch): return int(ch * alpha)

        # Initial convolution layers - Scaling channels by alpha
        self.conv1 = ConvLayer(3, scale(32), kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(scale(32), affine=True)
        
        self.conv2 = ConvLayer(scale(32), scale(64), kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(scale(64), affine=True)
        
        self.conv3 = ConvLayer(scale(64), scale(128), kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(scale(128), affine=True)

        # Residual layers
        self.res1 = ResidualBlock(scale(128))
        self.res2 = ResidualBlock(scale(128))
        self.res3 = ResidualBlock(scale(128))

        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(scale(128), scale(64), kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(scale(64), affine=True)
        
        self.deconv2 = UpsampleConvLayer(scale(64), scale(32), kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(scale(32), affine=True)
        
        self.deconv3 = ConvLayer(scale(32), 3, kernel_size=9, stride=1)

        self.relu = torch.nn.ReLU()
        
        # Adjust linear layer to match the scaled bottleneck width
        self.linear = torch.nn.Linear(64, scale(128))

    def forward(self, X, gbuffer_features=None):
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))

        # Fusion logic
        if gbuffer_features is not None:
            g = self.linear(gbuffer_features).unsqueeze(2).unsqueeze(3)
            y = y * (0.9 * g) + (0.1 * g)

        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y

class ResidualBlock(torch.nn.Module):
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
        return out + residual

class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if self.upsample:
            self.upsample_layer = torch.nn.Upsample(scale_factor=self.upsample, mode='nearest')
        
        padding = kernel_size // 2
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, padding_mode='reflect')

    def forward(self, x):
        if self.upsample:
            x = self.upsample_layer(x)
        return self.conv2d(x)