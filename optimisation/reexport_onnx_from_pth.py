from pathlib import Path
import torch

import onnxoptimizer
import onnx
from onnx import version_converter
from onnxconverter_common import float16

IMAGES_PATH = Path("data/objects365_val_patch1/") # relative to root

from transformer_net_fusion import TransformerNetFusion as tnf
from transformerNetFusion_res2 import TransformerNetFusion_res2 as tnf_res2
from transformerNetFusion_res1 import TransformerNetFusion_res1 as tnf_res1
from transformerNetFusion_separated import TransformerNetFusion_separated as tnf_separated
from transformerNetFusion_separated_scaled import TransformerNetFusion_separated_scaled as tnf_separated_scaled



for model in Path("optimisation").glob("*.pth"):
    print("------")
    print(f"Processing model: {model}")
    # load the model, optimise with onnx, then sav
    try:
        import torchvision.transforms as T
    except Exception:
        Image = None
        T = None

    # load correct model based on filename
    if "res2" in model.name:
        print("using res2")
        TNF = tnf_res2()
    elif "res1" in model.name:
        print("using res1")
        TNF = tnf_res1()
    elif "xscaled" in model.name:
        if "2x" in model.name:
            print("using 2x")
            TNF = tnf_separated_scaled(alpha=0.75)
        else:
            print("using 4x")
            TNF = tnf_separated_scaled(alpha=0.5)
    elif "separated" in model.name:
        print("using separated")
        TNF = tnf_separated()
    else:
        print("using default")
        TNF = tnf()
    TNF.load_state_dict(torch.load(model))
           
    net = TNF
    net.eval()

    input_shape = (1,3,640,640) # checked beforehand
    dummy_input = (torch.rand(input_shape, dtype=torch.float32),)

    onnx_path = model.with_suffix(".onnx").with_name(model.stem + "_distilled.onnx")
    try:
        torch.onnx.export(
            net,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=9,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamo=False
        )
    except Exception as e:
        print(f"ONNX export failed for {model}: {e}")
        continue

    # moveto version 9 for barracuda
    try:
        print("Converting to opset 9")
        onnx_model = onnx.load(str(onnx_path))
        # print model version
        onnx_version = onnx_model.opset_import[0].version
        print(f"Original ONNX opset version: {onnx_version}")
        converted_model = version_converter.convert_version(onnx_model, 9)
        optimized_model = onnxoptimizer.optimize(converted_model)
        onnx.save(optimized_model, str(onnx_path))
    except Exception as e:
        print(f"ONNX optimization failed for {onnx_path}: {e}")
        continue
    
    # take the new model and also save _fp16
    onnx_path_fp16 = model.with_suffix(".onnx").with_name(model.stem + "_distilled_fp16.onnx")
    try:
        print("Converting to FP16")
        onnx_model = onnx.load(str(onnx_path))
        fp16 = float16.convert_float_to_float16(onnx_model)
        onnx.save(fp16, str(onnx_path_fp16))
    except Exception as e:
        print(f"FP16 conversion failed for {onnx_path}: {e}")

    print(f"Saved ONNX: {onnx_path}")