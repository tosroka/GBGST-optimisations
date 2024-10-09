''' 
based on utils of PyTorch implementation of Johnson et al 
https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
'''
import torch
from PIL import Image
import torch.nn.functional as F

def depth_guided_loss(stylized_img, depth_map, sigma=0.1, kernel_size=3):
    # Define a convolution kernel that computes difference with central pixel
    center = kernel_size // 2
    kernel = torch.ones(1, 1, kernel_size, kernel_size).to(stylized_img.device) 
    kernel[0, 0, center, center] = -kernel_size**2 + 1

    # Compute local difference using convolution for stylized image and depth map
    img_diff = F.conv2d(stylized_img, kernel.repeat(3, 1, 1, 1), padding=center, groups=3)
    depth_diff = F.conv2d(depth_map, kernel, padding=center)

    # Compute depth similarity
    depth_similarity = torch.exp(-depth_diff ** 2 / (2 * sigma ** 2))

    # Compute loss using depth similarity and image difference
    loss = (depth_similarity * img_diff**2).mean()

    return loss

def normal_guided_loss(stylized_img, normal_map, sigma=0.1, kernel_size=3):
    # Define a convolution kernel that computes difference with central pixel
    center = kernel_size // 2
    kernel = torch.ones(1, 1, kernel_size, kernel_size).to(stylized_img.device) 
    kernel[0, 0, center, center] = -kernel_size**2 + 1

    # Compute local difference using convolution for stylized image
    img_diff = F.conv2d(stylized_img, kernel.repeat(3, 1, 1, 1), padding=center, groups=3)
    
    # Compute local difference for the single-channel normal map
    normal_diff = F.conv2d(normal_map, kernel, padding=center)

    # Compute normal similarity
    normal_similarity = torch.exp(-normal_diff ** 2 / (2 * sigma ** 2))

    # Compute loss using normal similarity and image difference
    loss = (normal_similarity * img_diff**2).mean()

    return loss

def load_image(filename, size=None, scale=None):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int (img.size[1] / scale)), Image.ANTIALIAS)
    
    return img


def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)

    return gram


def normalize_batch(batch):
    # normalises using imagenet and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.255]).view(-1, 1, 1)
    batch = batch.div_(255.0)

    return (batch - mean) / std