import argparse
import os
import sys
import time
import re

import numpy as np
import torch
torch.cuda.empty_cache()

from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torchvision
import torch.onnx

import utils
from transformer_net import TransformerNet
from transformer_net_light import TransformerNetLight
from transformer_net_fusion import TransformerNetFusion
from vgg import Vgg16
from gbuffer_encoder import GBufferEncoder
from gbuffer_encoder_net import GBufferEncoderNet


# for depth use MiDaS (https://pytorch.org/hub/intelisl_midas_v2/)
import torchvision.models as models

from decimal import Decimal

import matplotlib.pyplot as plt
from pylab import *
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True                                                                               

# from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from tqdm import trange

from aim import Run 
from aim import Image as aimImage

import pyfeats
import kornia
import lpips

import cv2
from custom_dataset import CustomDataset


import datetime
def get_time_in_format(format_string= "%d_%m__%H_%M"):
  """
  Gets the current time in the specified format.

  Args:
    format_string: The format string to use.

  Returns:
    A string containing the current time in the specified format.
  """

  now = datetime.datetime.now()
  return now.strftime(format_string)

def check_paths(args):
    try:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        if args.checkpoint_model_dir is not None and not (os.path.exists(args.checkpoint_model_dir)):
            os.makedirs(args.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


def train(args):

    device = torch.device("cuda" if args.cuda else "cpu")
    print("Device: ", torch.cuda.get_device_name(0))
    # calculate the save file name
    save_model_filename = str(time.ctime()).replace(':','_').replace(' ', '_') + "_" + str(
        args.style_image).replace('.jpg', '').replace('images/styles/', '') + "_" + str(
            args.depth_loss) + "_" + str('%.E' % Decimal (args.content_weight)).replace('+','') + "_" + str(
            '%.E' % Decimal (args.style_weight)).replace('+','') + "_" + str(
                '%.E' % Decimal (args.depth_weight)).replace('+', '') + ".pth" # changed extension to .pth
    print('Save model filename: ', save_model_filename)

    if (args.depth_loss):
        print("Training with depth loss (MiDaS), depth weight = ", args.depth_weight)
    else:
        print("Training without depth loss")

    # writer = SummaryWriter()
    if (args.aim):
        run = Run(experiment="GBuffer fusion; no depth,normal guided loss, pre-trained, feathers")

        # Log run parameters
        run["hparams"] = {   
            "epochs": args.epochs,
            "content_loss": args.content_weight,
            "style_loss": args.style_weight,
            "depth_loss": args.depth_loss,
            "depth_weight": args.depth_weight,
            "normal_loss": args.normal_loss,
            "normal_weight": args.normal_weight,
            "seg_loss": args.sem_loss,
            "seg-weight": args.sem_weight

        }

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # transform = transforms.Compose([
    #     # transforms.Resize((args.image_size, args.image_size)),
    #     transforms.Resize((640,360)),
    #     # transforms.CenterCrop(args.image_size),
    #     transforms.ToTensor(),
    #     transforms.Lambda(lambda x: x.mul(255))
    # ])

    transform = transforms.Compose([
        transforms.Resize((360,640)),
        # transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness = 0.1, contrast = 0.1, saturation=0.1, hue=0.1),
        # transforms.RandomRotation(30),
        # transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Standard normalization

        # transforms.Normalize(mean=[80.3968/255+0.5, 76.1473/255+0.5, 69.7807/255+0.5], std=[46.1483/255+0.1, 44.9569/255+0.1, 48.1648/255+0.1]),  # Synth normalization
        # transforms.Normalize(mean=[0.3153, 0.2986, 0.2736], std=[0.1809, 0.1763, 0.1889]),  # Synth normalization
        # transforms.Normalize(mean=[0.4712, 0.4476, 0.4081], std=[0.2381, 0.2332, 0.2364]),  # coco normalization
        # transforms.Normalize(mean=[120.1552/255, 114.1339/255, 104.0595/255], std=[60.7264/255, 59.4565/255, 60.2773/255]),  # coco normalization
        
        transforms.Lambda(lambda x: x.mul(255)),
        
        # transforms.Normalize(mean=[80.3968, 76.1473, 69.7807], std=[46.1483, 44.9569, 48.1648])  # Synthetic normalization
        # transforms.Normalize(mean=[120.1552, 114.1339, 104.0595], std=[60.7264, 59.4565, 60.2773])  # MS COCO 
    ])

    # train_dataset = datasets.ImageFolder(args.dataset, transform)
    # # wikiart = datasets.ImageFolder('../wikiart', transform)
    # # train_dataset += wikiart
    # sintel = datasets.ImageFolder('../mixed_dataset/MPI-Sintel-training_images/training/final/', transform)
    # train_dataset += sintel
    # gta = datasets.ImageFolder('../mixed_dataset/GTA-V/gta_trainfinal/gta_train', transform)
    # train_dataset += gta
    # data_dir = '../../../Unity/nst_hdrp_sample/Dataset/terrain_01/solo/sequence.0/'
    # train_dataset = CustomDataset(data_dir, transform=transform)

    # terrrain_02 = '../../../Unity/nst_hdrp_sample/Dataset/terrain_02/solo_2/sequence.0/'
    # train_dataset += CustomDataset(terrrain_02, transform=transform)

    # botd_data_dir = '../../../Unity/book_of_the_dead/Dataset/hd/solo/sequence.0/'
    # train_dataset += CustomDataset(botd_data_dir, transform=transform)

    # fontaine_day_data_dir = '../../../Unity/FontainebleauDemo/Dataset/day/sequence.0/'
    # fontaine_night_data_dir = '../../../Unity/FontainebleauDemo/Dataset/night/sequence.0/'
    # train_dataset += CustomDataset(fontaine_day_data_dir, transform=transform)
    # train_dataset += CustomDataset(fontaine_night_data_dir, transform=transform)
    
    seedhunter_data_dir = '../../../Unity/SeedHunter/Dataset/solo/sequence.0/'
    train_dataset = CustomDataset(seedhunter_data_dir, transform=transform)

    seed_hunter_2_data_dir = '../../../Unity/SeedHunter/Dataset/seed_hunter_2/solo_2/sequence.0/'
    train_dataset += CustomDataset(seedhunter_data_dir, transform=transform)
    
    # challenger_day_data_dir =  '../../../Unity/FontainebleauDemo/Dataset/challenger/solo/sequence.0/'
    # challenger_night_data_dir =  '../../../Unity/FontainebleauDemo/Dataset/challenger/solo_1/sequence.0/'
    # train_dataset += CustomDataset(challenger_day_data_dir, transform=transform)
    # train_dataset += CustomDataset(challenger_night_data_dir, transform=transform)

    # sample = train_dataset[10]
    # rgb_image = sample['rgb']
    # print(rgb_image.size())
    # r = rgb_image.permute(1,2,0).cpu().numpy()#.resize(256)
    # plt.imshow(r); plt.axis('off'); plt.show()
    # depth_map = sample['depth']
    # d = depth_map.cpu().numpy()#.resize(256)
    # plt.imshow(d); plt.axis('off'); plt.show()
    # normal_map = sample['normal']
    # n = normal_map.cpu().numpy()#.resize(256)
    # plt.imshow(n); plt.axis('off'); plt.show()
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    transformer = TransformerNetFusion().to(device)
    state_model_name =  'saved_models/cvpr_games/Wed_Sep_27_10_22_07_2023_composition_vii_0_1E05_1E10_1E03_COCO.pth' # 'saved_models/Wed_Sep_27_10_22_07_2023_composition_vii_0_1E05_1E10_1E03.pth' # 'saved_models/Mon_Sep_25_23_14_12_2023_starry_night_0_1E05_1E10_1E03.pth' #'saved_models/Mon_Sep_25_16_15_17_2023_starry_night_0_1E05_1E10_1E03.pth'
    state_dict = torch.load(state_model_name)
            
    transformer.load_state_dict(state_dict, strict=False)
    transformer.to(device)

    

    gbufferEncoder = GBufferEncoderNet().to(device)

    optimizer = Adam(list(transformer.parameters()) + list(gbufferEncoder.parameters()), args.lr)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16(requires_grad=False).to(device)


    timing = get_time_in_format()
    print(timing)
    # freeze first 3 layers
    if (args.freeze):
        for name, param in transformer.named_parameters():
            if 'conv' in name and 'conv1' == name or 'conv2' == name or 'conv3' == name:
                param.requires_grad = False
            if 'in' in name and 'in1' == name or 'in2' == name or 'in3' == name:
                param.requires_grad = False


    style_transform = transforms.Compose([
        # transforms.Resize((640, 360)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    style = utils.load_image(args.style_image, size=args.style_size)
    style = style_transform(style)
    style = style.repeat(args.batch_size, 1, 1, 1).to(device)
    
    features_style = vgg(utils.normalize_batch(style))
    gram_style = [utils.gram_matrix(y) for y in features_style]


    # sobel_style = kornia.filters.sobel(transforms.Resize((640, 360))(style))
    #######################################################################################
    # depth estimation network (MiDaS)
    if (args.depth_loss):
        model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
        # ##model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
        # ##model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

        midas = torch.hub.load("intel-isl/MiDaS", model_type)
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            midas_transform = midas_transforms.dpt_transform
        else:
            midas_transform = midas_transforms.small_transform
        
        for param in midas.parameters():
            param.requires_grad = False
        midas = midas.to(device)         
        midas.eval()

    if (args.sem_loss):
        ###########################
        # semantic segmentation
        deeplab = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
        deeplab.to(device)
        deeplab.eval()

    transformer.train()
    gbufferEncoder.train()
       
    log_count = 0
    for e in range(args.epochs):
        
        
        agg_content_loss = 0.
        agg_style_loss = 0.

        if (args.depth_loss):
            agg_depth_loss = 0.
            agg_normal_loss = 0.
        
        if (args.normal_loss):
            agg_normal_loss = 0.
            agg_depth_loss = 0.
        
        if (args.sem_loss):
            agg_sem_loss = 0.

        count = 0

        for batch_id, sample in enumerate(tqdm(train_loader, total=len(train_loader))):
            
            # rgb_image = sample['rgb'][0]
            # print(rgb_image.size())
            # r = rgb_image.permute(1,2,0).cpu().numpy()#.resize(256)
            # plt.imshow(r); plt.axis('off'); plt.show()
            # depth_map = sample['depth'][0]
            # d = depth_map.cpu().numpy()#.resize(256)
            # plt.imshow(d); plt.axis('off'); plt.show()
            # normal_map = sample['normal'][0]
            # n = normal_map.cpu().numpy()#.resize(256)
            # plt.imshow(n); plt.axis('off'); plt.show()

            n_batch = len(sample['rgb'])
            # print(n_batch)
            count += n_batch
            optimizer.zero_grad()

            rgb_image = sample['rgb'].to(device)
            depth_map = sample['depth'].to(device)
            normal_map = sample['normal'].to(device)
            x_feat = kornia.filters.sobel(rgb_image)
            # x_feat = kornia.filters.sobel(style)
            # rgb = rgb_image / 255
            # mid = rgb[0].permute(1,2,0).cpu().detach().numpy()#.resize(256)
            # plt.imshow(mid); plt.axis('off'); plt.show()
            # depth_map /= 255
            # normal_map /= 255
            # print(torch.min(depth_map), torch.max(depth_map))
            # print(torch.min(depth_map), torch.max(depth_map))

            gbuffer_input = torch.cat((depth_map.unsqueeze(1), normal_map.unsqueeze(1), x_feat), dim=1)
            gbuffer_features = gbufferEncoder(gbuffer_input)

            y = transformer(rgb_image, gbuffer_features)
            y = utils.normalize_batch(y)
            rgb_image = utils.normalize_batch(rgb_image)
            # x = utils.normalize_batch(x)

            features_y = vgg(y)
            features_x = vgg(rgb_image)

            content_loss = args.content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)
            
            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = utils.gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:args.batch_size, :, :])
            style_loss *= args.style_weight 
            total_loss = content_loss + style_loss   

            # depth loss
            if (args.depth_loss):
                # depth_map_normalized = depth_map / 255.0 
                # print(torch.min(depth_map_normalized), torch.max(depth_map_normalized))
                x_midas = midas(rgb_image)
                # # d = depth_map[0].cpu().numpy()#.resize(256)
                # # plt.imshow(d); plt.axis('off'); plt.show()
                y_midas = midas(y)
                # # mid = y_midas[0].cpu().detach().numpy()#.resize(256)
                # # plt.imshow(mid); plt.axis('off'); plt.show()
                depth_loss = args.depth_weight * mse_loss(y_midas, x_midas)
                # depth_loss = args.depth_weight * utils.depth_guided_loss(y, depth_map_normalized.unsqueeze(1))
                total_loss += depth_loss 
            
            if (args.normal_loss):
                normal_map_normalized = normal_map / 255.0 
                normal_loss = args.normal_weight * utils.normal_guided_loss(y, normal_map_normalized.unsqueeze(1))
               
                total_loss += normal_loss 
            
        

            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()
            if (args.depth_loss):
                agg_depth_loss += depth_loss.item()
            
            if (args.normal_loss):
                agg_normal_loss += normal_loss.item()
            
            
            if (batch_id + 1) % args.log_interval == 0:

                img_grid = stylize_aim(args, transformer, timing)
                img_grid = img_grid.clone().clamp(0, 255).numpy()
                img_grid = img_grid.transpose(1, 2, 0).astype("uint8")
                img_grid = Image.fromarray(img_grid)

                if ((not args.depth_loss) and (not args.sem_loss)):
                    mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                        time.ctime(), e + 1, count, len(train_dataset),
                                    agg_content_loss / (batch_id + 1),
                                    agg_style_loss / (batch_id + 1),
                                    (agg_content_loss + agg_style_loss) / (batch_id + 1)
                    )
                    # writer.add_scalar("content_loss", agg_content_loss / (batch_id + 1), log_count)
                    # writer.add_scalar("style_loss", agg_style_loss / (batch_id + 1), log_count)
                    # writer.add_images('stylised_result', img_grid, log_count)

                    if (args.aim):
                        run.track(agg_content_loss / (batch_id + 1), name='content loss', epoch=e, context={'subset': 'val'})
                        run.track(agg_style_loss / (batch_id + 1), name='style loss', epoch=e, context={'subset': 'val'})
                        run.track(aimImage(img_grid), name='train image', context={ "subset": "train" })

                elif (args.depth_loss and not args.sem_loss):
                    mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\tdepth: {:.6f}\tnormal: {:.6f}\ttotal: {:.6f}".format(
                        time.ctime(), e + 1, count, len(train_dataset),
                                    agg_content_loss / (batch_id + 1),
                                    agg_style_loss / (batch_id + 1),
                                    agg_depth_loss / (batch_id + 1),
                                    agg_normal_loss / (batch_id + 1),
                                    (agg_content_loss + agg_style_loss + agg_depth_loss + agg_normal_loss) / (batch_id + 1)
                    )
                    # writer.add_scalar("content_loss", agg_content_loss / (batch_id + 1), log_count)
                    # writer.add_scalar("style_loss", agg_style_loss / (batch_id + 1), log_count)
                    # writer.add_scalar("depth_loss", agg_depth_loss / (batch_id + 1), log_count)
                    # writer.add_images('stylised_result', img_grid, log_count)  

                    if (args.aim):  
                        run.track(agg_content_loss / (batch_id + 1), name='content loss', epoch=e, context={'subset': 'val'})
                        run.track(agg_style_loss / (batch_id + 1), name='style loss', epoch=e, context={'subset': 'val'})
                        run.track(agg_depth_loss / (batch_id + 1), name='depth loss', epoch=e, context={'subset': 'val'})

                        run.track(aimImage(img_grid), name='train image', context={ "subset": "train" })

                elif (args.sem_loss and not args.depth_loss):
                    mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\tsemantic: {:.6f}\ttotal: {:.6f}".format(
                        time.ctime(), e + 1, count, len(train_dataset),
                                    agg_content_loss / (batch_id + 1),
                                    agg_style_loss / (batch_id + 1),
                                    agg_sem_loss / (batch_id + 1),
                                    (agg_content_loss + agg_style_loss + agg_sem_loss) / (batch_id + 1)
                    )
                    # writer.add_scalar("content_loss", agg_content_loss / (batch_id + 1), log_count)
                    # writer.add_scalar("style_loss", agg_style_loss / (batch_id + 1), log_count)
                    # writer.add_scalar("sem_loss", agg_sem_loss / (batch_id + 1), log_count)
                    # writer.add_images('stylised_result', img_grid, log_count)    
                    if (args.aim):
                        run.track(agg_content_loss / (batch_id + 1), name='content loss', epoch=e, context={'subset': 'val'})
                        run.track(agg_style_loss / (batch_id + 1), name='style loss', epoch=e, context={'subset': 'val'})
                        run.track(agg_sem_loss / (batch_id + 1), name='semantic loss', epoch=e, context={'subset': 'val'})

                        run.track(aimImage(img_grid), name='train image', context={ "subset": "train" })

                else: 
                    mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\tdepth: {:.6f}\tsemantic: {:.6f}\ttotal: {:.6f}".format(
                        time.ctime(), e + 1, count, len(train_dataset),
                                    agg_content_loss / (batch_id + 1),
                                    agg_style_loss / (batch_id + 1),
                                    agg_depth_loss / (batch_id + 1),
                                    agg_sem_loss / (batch_id + 1),
                                    (agg_content_loss + agg_style_loss + agg_depth_loss + agg_sem_loss) / (batch_id + 1)
                    )
                    # writer.add_scalar("content_loss", agg_content_loss / (batch_id + 1), log_count)
                    # writer.add_scalar("style_loss", agg_style_loss / (batch_id + 1), log_count)
                    # writer.add_scalar("depth_loss", agg_depth_loss / (batch_id + 1), log_count)
                    # writer.add_scalar("sem_loss", agg_sem_loss / (batch_id + 1), log_count)
                    # writer.add_images('stylised_result', img_grid, log_count)    
                    if (args.aim):
                        run.track(agg_content_loss / (batch_id + 1), name='content loss', epoch=e, context={'subset': 'val'})
                        run.track(agg_style_loss / (batch_id + 1), name='style loss', epoch=e, context={'subset': 'val'})
                        run.track(agg_depth_loss / (batch_id + 1), name='depth loss', epoch=e, context={'subset': 'val'})
                        run.track(agg_sem_loss / (batch_id + 1), name='semantic loss', epoch=e, context={'subset': 'val'})

                        print(img_grid.size())
                        run.track(aimImage(img_grid.squeeze(0)), name='train image', context={ "subset": "train" })



                log_count += 1
                print(mesg)

            if args.checkpoint_model_dir is not None and (batch_id + 1) % args.checkpoint_interval == 0:
                transformer.eval().cpu()
                ckpt_model_filename = "ckpt_epoch_" + str(e) + "_batch_id_" + str(batch_id + 1) + ".pth"
                ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
                torch.save(transformer.state_dict(), ckpt_model_path)
                transformer.to(device).train()
            
           

    # writer.flush()
    # writer.close()
    # save model
    transformer.eval().cpu()
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(transformer.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


# stylize for aim ------------------------------------------------------------------------------------------------------------------------
def stylize_aim(args, style_model, timing):
    device = torch.device("cuda" if args.cuda else "cpu")

    content_image = utils.load_image("images/content/unity1.png")
    content_transform = transforms.Compose([
        # transforms.Resize((512,512)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)
  
    with torch.no_grad():
        output = style_model(content_image).cpu()
    # print(output.size())
    utils.save_image(timing + '.png', output[0])       
    return output[0]

# ------------------------------------------------------------------------------------------------------------------------
def stylize(args):
    device = torch.device("cuda" if args.cuda else "cpu")

    content_image = utils.load_image(args.content_image, scale=args.content_scale)
    content_transform = transforms.Compose([
        # transforms.Resize((512,512)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)
    # model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    # midas = torch.hub.load("intel-isl/MiDaS", model_type)
    # for param in midas.parameters():
    #     param.requires_grad = False
    # midas = midas.to(device)         
    # midas.eval()
    # depth_image = midas(content_image)
    # depth_image = transforms.Resize(512)(depth_image)
    
    if args.model.endswith(".onnx"):
        output = stylize_onnx_caffe2(content_image, args)
    else:
        with torch.no_grad():
            style_model = TransformerNetFusion() 
            state_dict = torch.load(args.model)
            # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
            for k in list(state_dict.keys()):
                if re.search(r'in\d+\.running_(mean|var)$', k):
                    del state_dict[k]
            style_model.load_state_dict(state_dict)
            style_model.to(device)
            if args.export_onnx:
                assert args.export_onnx.endswith(".onnx"), "Export model file should end with .onnx"
                output = torch.onnx._export(style_model, content_image, args.export_onnx).cpu()
            else:
                
                output = style_model(content_image).cpu()
                

    # rep = {".jpg": ".png", ".png": "text"} # define desired replacements here
    output_image_name = 'images/output/' + str(
        args.model).replace('saved_models/', '').replace('.pth', '_') + str(
            args.content_image).replace('images/content/','').replace('.jpg','.png')
    # utils.save_image(args.output_image, output[0])
    if (args.output_image):
        print('Output image name: ', args.output_image)
        utils.save_image(args.output_image, output[0])
    else:
        print('Output image name: ', output_image_name)
        utils.save_image(output_image_name, output[0])


def stylize_p(content_img, model, cuda=1, content_path='',content_scale=None):
    device = torch.device("cuda" if cuda else "cpu")

    content_image = utils.load_image(content_img, scale=content_scale)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)


    with torch.no_grad():
        style_model = TransformerNetFusion()
        state_dict = torch.load(model)
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        output = style_model(content_image)# .cpu()
        end.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()
        print(f"Total inference time per input: {start.elapsed_time(end):.3f} milliseconds")

    output.cpu()
    output_image_name = 'images/output/' + str(
        model).replace('saved_models/', '').replace('cvpr_games_oct_30','').replace('.pth', '/') + str(
            content_img).replace('images/content/','').replace(content_path, '').replace('.jpg','.png')
   
    # utils.save_image(output_image_name, output[0])


def stylize_image(image, model, device):
    content_image = utils.load_image(image)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)
    styled_image = model(image)[0].squeeze()
    # styled_image = postprocess_reconet(styled_image)
    return styled_image


def stylize_onnx_caffe2(content_image, args):
    """
    Read ONNX model and run it using Caffe2
    """

    assert not args.export_onnx

    import onnx
    import onnx_caffe2.backend

    model = onnx.load(args.model)

    prepared_backend = onnx_caffe2.backend.prepare(model, device='CUDA' if args.cuda else 'CPU')
    inp = {model.graph.input[0].name: content_image.numpy()}
    c2_out = prepared_backend.run(inp)[0]

    return torch.from_numpy(c2_out)


def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--epochs", type=int, default=2,
                                  help="number of training epochs, default is 2")
    train_arg_parser.add_argument("--batch-size", type=int, default=4,
                                  help="batch size for training, default is 4")
    train_arg_parser.add_argument("--dataset", type=str, required=False,
                                  help="path to training dataset, the path should point to a folder "
                                       "containing another folder with all the training images")
    train_arg_parser.add_argument("--style-image", type=str, default="images/style-images/mosaic.jpg",
                                  help="path to style-image")
    train_arg_parser.add_argument("--save-model-dir", type=str, required=True,
                                  help="path to folder where trained model will be saved.")
    train_arg_parser.add_argument("--checkpoint-model-dir", type=str, default=None,
                                  help="path to folder where checkpoints of trained models will be saved")
    train_arg_parser.add_argument("--image-size", type=int, default=512,
                                  help="size of training images, default is 256 X 256")
    train_arg_parser.add_argument("--style-size", type=int, default=None,
                                  help="size of style-image, default is the original size of style image")
    train_arg_parser.add_argument("--cuda", type=int, required=True,
                                  help="set it to 1 for running on GPU, 0 for CPU")
    train_arg_parser.add_argument("--seed", type=int, default=42,
                                  help="random seed for training")
    train_arg_parser.add_argument("--content-weight", type=float, default=1e5,
                                  help="weight for content-loss, default is 1e5")
    train_arg_parser.add_argument("--style-weight", type=float, default=1e10,
                                  help="weight for style-loss, default is 1e10")
    train_arg_parser.add_argument("--lr", type=float, default=1e-3,
                                  help="learning rate, default is 1e-3")
    train_arg_parser.add_argument("--log-interval", type=int, default=500,
                                  help="number of images after which the training loss is logged, default is 500")
    train_arg_parser.add_argument("--checkpoint-interval", type=int, default=2000,
                                  help="number of batches after which a checkpoint of the trained model will be created")
    train_arg_parser.add_argument("--depth-loss", type=int, default=0,
                                  help="set it to 1 to train with depth loss, 0 train without depth loss, default is 1")
    train_arg_parser.add_argument("--depth-weight", type=float, default=1e3,
                                help="weight for depth-loss, default is 1e3")
    train_arg_parser.add_argument("--normal-loss", type=int, default=0,
                                  help="set it to 1 to train with normal loss, 0 train without normal loss, default is 1")
    train_arg_parser.add_argument("--normal-weight", type=float, default=1e3,
                                help="weight for normal-loss, default is 1e3")
    train_arg_parser.add_argument("--sem-loss", type=int, default=0,
                                  help="set it to 1 to train with semantic loss, 0 train without semantic loss, default is 1")
    train_arg_parser.add_argument("--sem-weight", type=float, default=1e10,
                                help="weight for semantic-loss, default is 1e10")
    train_arg_parser.add_argument("--feats", type=int, default=0,
                                  help="set it to 1 to train with pyfeats loss, 0 train without feats loss, default is 1")
    train_arg_parser.add_argument("--aim", type=int, default=0,
                                help="set to 1 to track experiment using Aim")
    train_arg_parser.add_argument("--freeze", type=int, default=0,
                                help="set to 1 to freeze first 3 layers")


    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
    eval_arg_parser.add_argument("--content-image", type=str, required=True,
                                 help="path to content image you want to stylize")
    eval_arg_parser.add_argument("--content-scale", type=float, default=None,
                                 help="factor for scaling down the content image")
    eval_arg_parser.add_argument("--output-image", type=str, required=False,
                                 help="path for saving the output image")
    eval_arg_parser.add_argument("--model", type=str, required=True,
                                 help="saved model to be used for stylizing the image. If file ends in .pth - PyTorch path is used, if in .onnx - Caffe2 path")
    eval_arg_parser.add_argument("--cuda", type=int, required=True,
                                 help="set it to 1 for running on GPU, 0 for CPU")
    eval_arg_parser.add_argument("--export_onnx", type=str,
                                 help="export ONNX model to a given file")

    args = main_arg_parser.parse_args()

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    if args.subcommand == "train":
        check_paths(args)
        train(args)
    else:
        stylize(args)


if __name__ == "__main__":
    main()