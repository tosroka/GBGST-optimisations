import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2 

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        img_path = os.path.join(self.data_dir, file_name)
        
        # Load RGB image
        rgb_image = Image.open(img_path).convert('RGB')
        
        # Load depth map (assuming .exr format)
        depth_file_name = file_name.replace('.png', '.Depth.exr')
        depth_path = os.path.join(self.data_dir, depth_file_name)
        depth_map = self.load_depth_map(depth_path)
        
        # Load normal map (assuming .exr format)
        normal_file_name = file_name.replace('.png', '.Normal.exr')
        normal_path = os.path.join(self.data_dir, normal_file_name)
        normal_map = self.load_normal_map(normal_path)

        if self.transform:
            rgb_image = self.transform(rgb_image)

        return {
            'rgb': rgb_image,
            'depth': depth_map,
            'normal': normal_map
        }

    def load_depth_map(self, depth_path):
        # Load the depth map from .exr file
       
        depth_map = cv2.imread(depth_path,cv2.IMREAD_UNCHANGED)
        depth_map = depth_map[:,:,2]
        min = depth_map.min()
        max = depth_map.max()
        # print(min, max)
        depth_map = (255-((depth_map-min)*224/max))#.astype(np.uint8)
        depth_map = cv2.resize(depth_map, (640, 360))
        depth_map[depth_map==255] = 0
        depth_map_tensor = torch.from_numpy(depth_map)
        return depth_map_tensor

        # depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        # depth_map = depth_map[:,:,2]
        # min_depth = depth_map.min()
        # max_depth = depth_map.max()
        
        # # Normalize the depth map to [0, 1]
        # depth_map = (depth_map - min_depth) / (max_depth - min_depth)
        
        # # Resize the depth map to match the RGB image size (256x256)
        # depth_map = cv2.resize(depth_map, (512, 512))
        
        # # Convert to uint8
        # depth_map = (depth_map * 255).astype(np.uint8)
        
        # # Invert the depth map (if needed)
        # depth_map = 255 - depth_map
        
        # # Convert to tensor
        # depth_map_tensor = torch.from_numpy(depth_map)
        
        
        # return depth_map_tensor


        

    def load_normal_map(self, normal_path):
        # Load the normal map from .exr file
        normal_map = cv2.imread(normal_path,cv2.IMREAD_UNCHANGED)
        normal_map = normal_map[:,:,2]
        min = normal_map.min()
        max = normal_map.max()
        # print(min, max)
        normal_map = (255-((normal_map-min)*224/max))#.astype(np.uint8)
        normal_map = cv2.resize(normal_map, (640, 360))
        normal_map[normal_map==255] = 0
        normal_map_tensor = torch.from_numpy(normal_map)
        return normal_map_tensor