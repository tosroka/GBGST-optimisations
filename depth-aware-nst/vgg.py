from collections import namedtuple

import torch
from torchvision import models

class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()

        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
    

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h 
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h 

        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out 

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        vgg = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, (1, 1)),
            torch.nn.ReflectionPad2d((1, 1, 1, 1)),
            torch.nn.Conv2d(3, 64, (3, 3)),
            torch.nn.ReLU(),  # relu1-1
            torch.nn.ReflectionPad2d((1, 1, 1, 1)),
            torch.nn.Conv2d(64, 64, (3, 3)),
            torch.nn.ReLU(),  # relu1-2
            torch.nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            torch.nn.ReflectionPad2d((1, 1, 1, 1)),
            torch.nn.Conv2d(64, 128, (3, 3)),
            torch.nn.ReLU(),  # relu2-1
            torch.nn.ReflectionPad2d((1, 1, 1, 1)),
            torch.nn.Conv2d(128, 128, (3, 3)),
            torch.nn.ReLU(),  # relu2-2
            torch.nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            torch.nn.ReflectionPad2d((1, 1, 1, 1)),
            torch.nn.Conv2d(128, 256, (3, 3)),
            torch.nn.ReLU(),  # relu3-1
            torch.nn.ReflectionPad2d((1, 1, 1, 1)),
            torch.nn.Conv2d(256, 256, (3, 3)),
            torch.nn.ReLU(),  # relu3-2
            torch.nn.ReflectionPad2d((1, 1, 1, 1)),
            torch.nn.Conv2d(256, 256, (3, 3)),
            torch.nn.ReLU(),  # relu3-3
            torch.nn.ReflectionPad2d((1, 1, 1, 1)),
            torch.nn.Conv2d(256, 256, (3, 3)),
            torch.nn.ReLU(),  # relu3-4
            torch.nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            torch.nn.ReflectionPad2d((1, 1, 1, 1)),
            torch.nn.Conv2d(256, 512, (3, 3)),
            torch.nn.ReLU(),  # relu4-1, this is the last layer used
            torch.nn.ReflectionPad2d((1, 1, 1, 1)),
            torch.nn.Conv2d(512, 512, (3, 3)),
            torch.nn.ReLU(),  # relu4-2
            torch.nn.ReflectionPad2d((1, 1, 1, 1)),
            torch.nn.Conv2d(512, 512, (3, 3)),
            torch.nn.ReLU(),  # relu4-3
            torch.nn.ReflectionPad2d((1, 1, 1, 1)),
            torch.nn.Conv2d(512, 512, (3, 3)),
            torch.nn.ReLU(),  # relu4-4
            torch.nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            torch.nn.ReflectionPad2d((1, 1, 1, 1)),
            torch.nn.Conv2d(512, 512, (3, 3)),
            torch.nn.ReLU(),  # relu5-1
            torch.nn.ReflectionPad2d((1, 1, 1, 1)),
            torch.nn.Conv2d(512, 512, (3, 3)),
            torch.nn.ReLU(),  # relu5-2
            torch.nn.ReflectionPad2d((1, 1, 1, 1)),
            torch.nn.Conv2d(512, 512, (3, 3)),
            torch.nn.ReLU(),  # relu5-3
            torch.nn.ReflectionPad2d((1, 1, 1, 1)),
            torch.nn.Conv2d(512, 512, (3, 3)),
            torch.nn.ReLU()  # relu5-4
        )

        for x in range(4):
            self.slice1.add_module(str(x), vgg[x])
        
        for x in range(4, 11):
            self.slice2.add_module(str(x), vgg[x])
        
        for x in range(11, 18):
            self.slice3.add_module(str(x), vgg[x])
        
        for x in range(18, 31):
            self.slice4.add_module(str(x), vgg[x])

        for x in range(31, 44):
            self.slice5.add_module(str(x), vgg[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
    

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_1 = h 
        h = self.slice2(h)
        h_relu2_1 = h
        h = self.slice3(h)
        h_relu3_1 = h
        h = self.slice4(h)
        h_relu4_1 = h 
        h = self.slice5(h)
        h_relu5_1 = h 

        vgg_outputs = namedtuple("VggOutputs", ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'])
        out = vgg_outputs(h_relu1_1, h_relu2_1, h_relu3_1, h_relu4_1, h_relu5_1)
        return out 

