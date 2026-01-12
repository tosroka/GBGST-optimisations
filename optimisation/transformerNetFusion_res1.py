import math
import torch

class TransformerNetFusion_res1(torch.nn.Module):
    def __init__(self):
        super(TransformerNetFusion_res1, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        # Residual layers
        self.res1 = ResidualBlock(128)
        #self.res2 = ResidualBlock(128)
        #self.res3 = ResidualBlock(128)
        #self.res4 = ResidualBlock(128)
        # self.res5 = ResidualBlock(128)
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()
        # linear layer for depth map
        self.linear = torch.nn.Linear(64, 128)
        # self.linear = torch.nn.Linear(1, 64)
       
    # def forward(self, X, gbuffer_features=None):
        
    #     y = self.relu(self.in1(self.conv1(X)))
    #     y = self.relu(self.in2(self.conv2(y)))
    #     y = self.relu(self.in3(self.conv3(y)))

      
    #     y = self.res1(y)
     
       
    #     y = self.res2(y)
      
    #     y = self.res3(y)
        
    
        
    #     y = self.relu(self.in4(self.deconv1(y)))
  
    #     y = self.relu(self.in5(self.deconv2(y)))
   
    #     y = self.deconv3(y)
    #     return y
    



    def forward(self, X, gbuffer_features=None):
        
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))

        
        if (gbuffer_features is not None):
            g = self.linear(gbuffer_features).unsqueeze(2).unsqueeze(3)
            y = y * (0.9 * g) + (0.1 * g) #  (0.9*gbuffer_features[0]) + (0.1*gbuffer_features[0])
        # if (gbuffer_features is not None):
        y = self.res1(y)
     
        # if (gbuffer_features is not None):
        #     g = self.linear(gbuffer_features).unsqueeze(2).unsqueeze(3)
        #     y = y * (0.9 * g) + (0.1 * g)
        # y = self.res2(y)
        # if (gbuffer_features is not None):
        #     g = self.linear(gbuffer_features).unsqueeze(2).unsqueeze(3)
        #     y = y * (0.9 * g) + (0.1 * g)
        # y = self.res3(y)
        
        # if (gbuffer_features is not None):
        #     g = self.linear(gbuffer_features).unsqueeze(2).unsqueeze(3)
        #     y = y * (0.9 * g) + (0.1 * g) # (0.9*gbuffer_features[1]) + (0.1*gbuffer_features[1])
        # y = self.res4(y)
        # y = self.res5(y)
        
        y = self.relu(self.in4(self.deconv1(y)))
        # if (gbuffer_features is not None):
        #     y = y * (gbuffer_features[2]) # (0.9*gbuffer_features[2]) + (0.1*gbuffer_features[2])
        y = self.relu(self.in5(self.deconv2(y)))
        # if (gbuffer_features is not None):
        #     y = y * (1.0 * gbuffer_features[3]) # + (0.4*gbuffer_features[3])
        y = self.deconv3(y)
        # print(y.size())
        return y
    


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


class ResidualBlockFusion(torch.nn.Module):
    # modelled after that used by Johnson et al. (2016)
    # see https://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf
    def __init__(self,channels):
        super(ResidualBlockFusion,self).__init__()
        self.n_params = channels * 4
        self.channels = channels
        in_channels = 128

        # self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        # self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        # self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        # self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        # self.relu = torch.nn.ReLU()

        self.reflection_pad = torch.nn.ReflectionPad2d(1)
        self.conv1 = torch.nn.Conv2d(channels,channels,3,stride=1,padding=0)
        self.instancenorm = torch.nn.InstanceNorm2d(channels, affine=True)
        self.fc_beta1 = torch.nn.Linear(in_channels,channels)
        self.fc_gamma1 = torch.nn.Linear(in_channels,channels)
        self.fc_beta2 = torch.nn.Linear(in_channels,channels)
        self.fc_gamma2 = torch.nn.Linear(in_channels,channels)
        self.relu = torch.nn.ReLU(inplace=False)
        self.conv2 = torch.nn.Conv2d(channels,channels,3,stride=1,padding=0)
        
    def forward(self, x, feats=None):
        # x: B x C x H x W  
        # style: B x self.n_params
        
        gamma1 = 1
        beta1 = 0
        gamma2 = 1
        beta2 = 0

        if (feats is not None):
            beta1 = self.fc_beta1(feats).unsqueeze(2).unsqueeze(3) # B x C x 1 x 1
            gamma1 = self.fc_gamma1(feats).unsqueeze(2).unsqueeze(3) # B x C x 1 x 1
            beta2 = self.fc_beta2(feats).unsqueeze(2).unsqueeze(3) # B x C x 1 x 1
            gamma2 = self.fc_gamma2(feats).unsqueeze(2).unsqueeze(3) # B x C x 1 x 1

        y = self.reflection_pad(x)
        y = self.conv1(y)
        y = self.instancenorm(y)
        # y = self.in1(self.conv1(x))
        if (feats is not None):
            y = (0.9 * gamma1) * y
            y += (0.1 * beta1)
        # y = self.relu(y)
        y = self.reflection_pad(y)
        y = self.conv2(y)
        y = self.instancenorm(y)
        # y = self.in2(self.conv2(x))
        if (feats is not None):
            y = (0.9 * gamma2) * y
            y += (0.1*beta2)
        return x + y


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        self.upsample_layer = torch.nn.Upsample(scale_factor=self.upsample)
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            # x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
            x_in = self.upsample_layer(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out