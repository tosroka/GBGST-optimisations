import argparse

import numpy as np
import torch

from torchvision import transforms
import torch.onnx

import utils
import cv2
import kornia

import matplotlib; matplotlib.use('agg')
from pylab import *
import matplotlib.pyplot as plt
from matplotlib import pylab 
from PIL import Image 

import urllib.request
from diffimg import diff


def main():
    
    parser = argparse.ArgumentParser(description='parser for calculating depth with MiDaS')
    parser.add_argument("--input-image", type=str, required=False,
                                 help="path to the image content image")
    parser.add_argument("--output-image",  type=str, required=False,
                                 help="path to output image")
    
    args = parser.parse_args()

    device = torch.device("cuda")
    print("Device: ", torch.cuda.get_device_name(0))


    img_input = args.input_image
    output_path = args.output_image
    
    # img_org = utils.load_image(img_input) # cv2.imread(img_original)
    # img_org_col = cv2.cvtColor(np.float32(img_org), cv2.COLOR_BGR2RGB)

    # img_midas_depth = midas_depth_image(img_org_col, device, args)
    # minimum_value = np.min(img_midas_depth)
    # maximum_value = np.max(img_midas_depth)
    
    transform = transforms.Compose([ 
        transforms.ToTensor(),
    ])

    # final_image = normal_map_(img_input)
    # final_image = depth_map_(img_input)

    # img1 = 'images/content/frame_0400.jpg'
    # img2 = 'images/content/frame_0399.jpg'

    # img1 = '../../unity_games_test/games/game_4_seed_scene_1/frame_0475.jpg'
    # img2 = '../../unity_games_test/games/game_4_seed_scene_1/frame_0474.jpg'

    # img1 = '../../CSBNet/results/starry_night/game_4_seed_scene_3_14/frame_1588_stylized_14.jpg'
    # img2 = '../../CSBNet/results/starry_night/game_4_seed_scene_3_14/frame_1587_stylized_14.jpg'

    # img1 = '../../../Unity/my_results_cvpr_2/wave/game_4_seed_scene_1/frame_0320.jpg'
    # img2 = '../../../Unity/my_results_cvpr_2/wave/game_4_seed_scene_1/frame_0319.jpg'

    # # img1 = '../../my_results_cvpr_image/starry_night/game_4_seed_scene_3/frame_1588.png'
    # # img2 = '../../my_results_cvpr_image/starry_night/game_4_seed_scene_3/frame_1587.png'

    # image1 = cv2.imread(img1)
    # image2 = cv2.imread(img2)
    # # compute difference
    # difference = cv2.subtract(image1, image2)

    # # color the mask red
    # Conv_hsv_Gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    # ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
    # difference[mask != 255] = [0, 0, 255]

    # # add the red mask to the images to make the differences obvious
    # image1[mask != 255] = [0, 0, 255]
    # image2[mask != 255] = [0, 0, 255]

    # # store images
    # # cv2.imwrite('images/hitmaps/diffOverImage1.png', image1)
    # # cv2.imwrite('diffOverImage2.png', image2)
    # cv2.imwrite('images/heatmaps/truth/ours_game_4_seed_scene_1_wave_diff.png', difference)


    rgb_image = Image.open(img_input).convert('RGB')
    rgb_image = transform(rgb_image).unsqueeze(0)

    x_feat = kornia.filters.sobel(1-rgb_image)[0].permute(1,2,0)
    single_figure_with_colorbar(x_feat, output_path)



def single_figure_with_colorbar(image, fig_name, min_value=0, max_value=0):

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    plt.figure(frameon=False)
    ax = plt.gca()
    im = ax.imshow(image) #, vmin=min_value, vmax=max_value) # uncomment for same scale colourbar
    
    # Turn off tick labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    # hide axes
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    # uncomment below to add colorbar
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # plt.colorbar(im, cax=cax) # uncomment to add colourbar

    plt.axis('off')
    plt.savefig(fig_name, bbox_inches='tight',transparent=True, pad_inches=0, dpi=300)

def midas_depth_image(img_org, device, args):

    model_type = "DPT_Large"   

    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    input_batch_1 = transform(img_org).to(device)

    with torch.no_grad():
        prediction_org = midas(input_batch_1)

        prediction_1 = torch.nn.functional.interpolate(
            prediction_org.unsqueeze(1),
            size=img_org.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()


    output_1 = prediction_1.cpu().numpy()
    
    return output_1


def normal_map_(normal_path):

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


def depth_map_(depth_path):

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

if __name__ == "__main__":
    main()