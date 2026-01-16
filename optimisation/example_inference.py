import torch
from transformerNetFusion_separated_scaled import TransformerNetFusion_separated_scaled as tnf_scaled
from transformerNetFusion_res1 import TransformerNetFusion_res1 as tnf_res1
from transformerNetFusion_res2 import TransformerNetFusion_res2 as tnf_res2
from transformer_net_fusion import TransformerNetFusion as tnf
import matplotlib.pyplot as plt
import numpy as np
import cv2

# import as pth
from pathlib import Path

IMAGES_PATH = Path("data/objects365_val_patch1")


base_path = Path('saved_models/cvpr_games_oct_30/Sat_Oct_28_18_44_34_2023_starry_night_1_1E05_1E10_1E03.pth')
base_distilled_path = Path("optimisation/baseline_distilled.pth")
SEPARATED_PATH = Path("optimisation/transformerNetFusion_separated.pth")
SEPARATED_2xSCALED_PATH = Path("optimisation/transformerNetFusion_separated_2xscaled.pth")
SEPARATED_4xSCALED_PATH = Path("optimisation/transformerNetFusion_separated_4xscaled.pth")
RES1_PATH = Path("optimisation/transformerNetFusion_res1.pth")
RES2_PATH = Path("optimisation/transformerNetFusion_res2.pth")

ideal_output = None

def stylize_and_display_image_pth(image_path: Path, pth_path: Path, TNF: torch.nn.Module=tnf(), dtype=np.float32):
    global ideal_output
    print("showing",image_path)
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (640, 640))
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    input_tensor = image.astype(np.float32)
    input_tensor = np.transpose(input_tensor, (2,0,1)) # HWC to CHW
    input_tensor = np.expand_dims(input_tensor, axis=0) # add batch dim

    model = TNF.to('cuda')
    model.load_state_dict(torch.load(pth_path))
    model.eval()
    with torch.no_grad():
        input_tensor_torch = torch.from_numpy(input_tensor).to('cuda')
        output_tensor_torch = model(input_tensor_torch)
        output_tensor = output_tensor_torch.cpu().numpy()
    print("Output shape:", output_tensor.shape)
    # display output as image
    output_image = np.squeeze(output_tensor, axis=0) # remove batch dim
    output_image = np.transpose(output_image, (1,2,0)) # CHW to
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)
    plt.imshow(output_image)
    plt.axis('off')
    plt.show()

    if pth_path==base_path:
        ideal_output = output_tensor
    else:
        mse = np.mean((ideal_output - output_tensor) ** 2)
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        print(f"PSNR: {psnr:.2f} dB")



stylize_and_display_image_pth(IMAGES_PATH / "objects365_v1_00083540.jpg", base_path, TNF=tnf(), dtype=np.float32)
stylize_and_display_image_pth(IMAGES_PATH / "objects365_v1_00083540.jpg", base_distilled_path, TNF=tnf(), dtype=np.float32)
stylize_and_display_image_pth(IMAGES_PATH / "objects365_v1_00083540.jpg", SEPARATED_PATH, TNF=tnf_scaled(alpha=1), dtype=np.float32)
stylize_and_display_image_pth(IMAGES_PATH / "objects365_v1_00083540.jpg", SEPARATED_2xSCALED_PATH, TNF=tnf_scaled(alpha=0.75), dtype=np.float32)
stylize_and_display_image_pth(IMAGES_PATH / "objects365_v1_00083540.jpg", SEPARATED_4xSCALED_PATH, TNF=tnf_scaled(alpha=0.5), dtype=np.float32)
stylize_and_display_image_pth(IMAGES_PATH / "objects365_v1_00083540.jpg", RES1_PATH, TNF=tnf_res1(), dtype=np.float32)
stylize_and_display_image_pth(IMAGES_PATH / "objects365_v1_00083540.jpg", RES2_PATH, TNF=tnf_res2(), dtype=np.float32)