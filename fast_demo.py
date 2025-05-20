import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
import tqdm
import os

from torch import autocast

#Custom models
from models.stereoanywhere import StereoAnywhere as StereoAnywhere

#Monocular models - VANILLA
from models.depth_anything_v2 import get_depth_anything_v2

torch._dynamo.config.capture_scalar_outputs = True
torch.set_float32_matmul_precision('high')


# Custom wrapper for StereoAnywhere (to handle padding and mono model)
class StereoAnywhereWrapper(torch.nn.Module):
    def __init__(self, args, stereo_model, mono_model):
        super(StereoAnywhereWrapper, self).__init__()
        self.args = args
        self.stereo_model = stereo_model
        self.mono_model = mono_model

    def forward(self, left_image, right_image, mono_left=None, mono_right=None):
        # Assuming the model takes a batch of images as input
        if self.mono_model is not None:
            #mono_depths = self.mono_model.infer_image(torch.cat([left_image, right_image], 0), input_size_width=self.args.mono_width, input_size_height=self.args.mono_height)
            mono_depth_left = self.mono_model.infer_image(left_image, input_size_width=self.args.mono_width, input_size_height=self.args.mono_height)
            mono_depth_right = self.mono_model.infer_image(right_image, input_size_width=self.args.mono_width, input_size_height=self.args.mono_height)
            mono_depths = torch.cat([mono_depth_left, mono_depth_right], 0)
            mono_depths = (mono_depths - mono_depths.min()) / (mono_depths.max() - mono_depths.min())
            mono_left = mono_depths[0].unsqueeze(0)
            mono_right = mono_depths[1].unsqueeze(0)
        else:
            mono_left = torch.zeros_like(left_image[:, 0:1]) if mono_left is None else mono_left
            mono_right = torch.zeros_like(right_image[:, 0:1]) if mono_right is None else mono_right

        # Pad 32
        ht, wt = left_image.shape[-2], left_image.shape[-1]
        pad_ht = (((ht // 32) + 1) * 32 - ht) % 32
        pad_wd = (((wt // 32) + 1) * 32 - wt) % 32
        _pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]

        left_image = F.pad(left_image, _pad, mode='replicate')
        right_image = F.pad(right_image, _pad, mode='replicate')
        mono_left = F.pad(mono_left, _pad, mode='replicate')
        mono_right = F.pad(mono_right, _pad, mode='replicate')

        pred_disps,_ = self.stereo_model(left_image, right_image, mono_left, mono_right, test_mode=True, iters=self.args.iters)
        pred_disp = -pred_disps.squeeze(1)

        hd, wd = pred_disp.shape[-2:]
        c = [_pad[2], hd-_pad[3], _pad[0], wd-_pad[1]]
        pred_disp = pred_disp[:, c[0]:c[1], c[2]:c[3]]

        return pred_disp

# Depth Anything V2 - TensorRT - Utils

# --------- Load the TensorRT engine ---------
def load_engine(engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

# --------- Allocate buffers using named tensors ---------
def dav2_rt_allocate_buffers(context):
    stream = cuda.Stream()

    input_name = "input"
    output_name = "output"

    input_shape = context.get_tensor_shape(input_name)
    output_shape = context.get_tensor_shape(output_name)

    input_dtype = trt.nptype(context.engine.get_tensor_dtype(input_name))
    output_dtype = trt.nptype(context.engine.get_tensor_dtype(output_name))

    input_size = trt.volume(input_shape)
    output_size = trt.volume(output_shape)

    h_input = cuda.pagelocked_empty(input_size, dtype=input_dtype)
    d_input = cuda.mem_alloc(h_input.nbytes)

    h_output = cuda.pagelocked_empty(output_size, dtype=output_dtype)
    d_output = cuda.mem_alloc(h_output.nbytes)

    # Bind addresses
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))

    return h_input, d_input, h_output, d_output, stream, input_shape, output_shape

# --------- Preprocess input image ---------
def dav2_rt_preprocess_image(img, input_shape):
    # Assuming a HWC image numpy array
    orig_h, orig_w = img.shape[:2]
    _, _, h, w = input_shape  # assuming NCHW

    # Resize image to the input shape
    img = cv2.resize(img, (w, h))
    # Convert to float32 and normalize
    img = img.astype(np.float32) / 255.0
    # Normalize using ImageNet mean and std
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    for i in range(3):
        img[:, :, i] = (img[:, :, i] - IMAGENET_MEAN[i]) / IMAGENET_STD[i]

    img = img.transpose((2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, axis=0)  # add batch dim
    return img, (orig_h, orig_w)

# --------- Run inference ---------
def dav2_rt_infer(engine, images):
    if not isinstance(images, list):
        images = [images]

    context = engine.create_execution_context()

    h_input, d_input, h_output, d_output, stream, input_shape, output_shape = dav2_rt_allocate_buffers(context)

    outputs = []

    for image in images:
        # Preprocess and copy to input
        input_image, (orig_h, orig_w) = dav2_rt_preprocess_image(image, input_shape)
        np.copyto(h_input, input_image.ravel())

        # Copy input to device
        cuda.memcpy_htod_async(d_input, h_input, stream)
        
        # Run inference
        context.execute_async_v3(stream.handle)
        
        # Copy output from device
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()

        # Post-process
        output = h_output.reshape(output_shape[-2], output_shape[-1])
        output = cv2.resize(output, (orig_w, orig_h))  # Resize to original shape

        outputs.append(output)

    return tuple(outputs) if len(outputs) > 1 else outputs[0]

# Depth Anything V2 - TensorRT - Utils ---- END

def main():
    parser = argparse.ArgumentParser(description='StereoAnywhere Fast Inference')

    parser.add_argument('--left', nargs='+', required=True, help='left image path(s)')
    parser.add_argument('--right', nargs='+', required=True, help='right image path(s)')

    parser.add_argument('--iscale', type=float, default=1.0, help='scale factor for input images')
    parser.add_argument('--outdir', default=None, type=str, help='output directory. If not specified (None), will be saved in the same directory as input images with .npy extension')
    parser.add_argument('--display_qualitatives', action='store_true', help='display qualitative results')
    parser.add_argument('--save_qualitatives', action='store_true', help='save qualitative results')

    parser.add_argument('--torch_compile', action='store_true', help='use torch.compile for optimization')
    parser.add_argument('--half', action='store_true', help='use half precision for inference')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision for inference')

    parser.add_argument('--stereomodel', default='stereoanywhere', help='select stereo model')
    parser.add_argument('--monomodel', default='DAv2', help='select mono model')

    parser.add_argument('--loadstereomodel', required=True, help='load stereo model')         
    parser.add_argument('--loadmonomodel', required=True, help='load mono model')

    parser.add_argument('--mono_width', type=int, default=518, help='Input width for the mono model')
    parser.add_argument('--mono_height', type=int, default=518, help='Input height for the mono model')
    parser.add_argument('--n_downsample', type=int, default=2)
    parser.add_argument('--n_additional_hourglass', type=int, default=0)
    parser.add_argument('--volume_channels', type=int, default=8)
    parser.add_argument('--vol_downsample', type=float, default=0)
    parser.add_argument('--vol_n_masks', type=int, default=8)
    parser.add_argument('--use_truncate_vol', action='store_true')
    parser.add_argument('--mirror_conf_th', type=float, default=0.98)
    parser.add_argument('--mirror_attenuation', type=float, default=0.9)
    parser.add_argument('--use_aggregate_stereo_vol', action='store_true')
    parser.add_argument('--use_aggregate_mono_vol', action='store_true')
    parser.add_argument('--normal_gain', type=int, default=10)
    parser.add_argument('--lrc_th', type=float, default=1.0)
    parser.add_argument('--iters', type=int, default=32, help='Number of iterations for recurrent networks')

    args = parser.parse_args()

    if args.outdir is not None:
        os.makedirs(args.outdir, exist_ok=True)
        
    dtype = torch.float16 if args.half else torch.float32
            
    stereonet = StereoAnywhere(args)

    stereonet = nn.DataParallel(stereonet)
    pretrain_dict = torch.load(args.loadstereomodel, map_location='cpu')
    pretrain_dict  = pretrain_dict['state_dict'] if 'state_dict' in pretrain_dict else pretrain_dict
    stereonet.load_state_dict(pretrain_dict, strict=True)  
    stereonet = stereonet.module
    stereonet = stereonet.eval()

    if args.monomodel == 'DAv2':
        mono_model = get_depth_anything_v2(args.loadmonomodel).eval().to(dtype)
    elif args.monomodel == 'DAv2RT':
        mono_model = None
        dav2rt_engine = load_engine(args.loadmonomodel)

    wrapper = StereoAnywhereWrapper(args, stereonet, mono_model)
    wrapper = wrapper.cuda().eval().to(dtype)
    optimized_model = torch.compile(wrapper) if args.torch_compile else wrapper

    if args.display_qualitatives:
        cv2.namedWindow("Disparity", cv2.WINDOW_NORMAL)

    for _left, _right in tqdm.tqdm(zip(args.left, args.right), desc="Processing stereo images", total=len(args.left)):
        
        # Load images
        left_image = cv2.imread(_left)
        right_image = cv2.imread(_right)

        # check if images are grayscale and convert to BGR
        if len(left_image.shape) == 2:
            left_image = cv2.cvtColor(left_image, cv2.COLOR_GRAY2BGR)
        if len(right_image.shape) == 2:
            right_image = cv2.cvtColor(right_image, cv2.COLOR_GRAY2BGR)

        original_shape = left_image.shape

        left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
        right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)

        # Mono model inference
        if args.monomodel == 'DAv2RT':
            # start_time = time.time()
            mono_left, mono_right = dav2_rt_infer(dav2rt_engine, [left_image, right_image])
            # mono_time = time.time() - start_time
            # print(f"Mono inference time: {mono_time:.4f} seconds")

            if args.save_qualitatives or args.display_qualitatives:
                mono_left_jet = (mono_left - mono_left.min()) / (mono_left.max() - mono_left.min())
                mono_left_jet = (mono_left_jet * 255).astype(np.uint8)
                mono_left_jet = cv2.applyColorMap(mono_left_jet, cv2.COLORMAP_JET)
                
                if args.save_qualitatives:
                    _output = f"{os.path.splitext(_left)[0]}_mono_left.png" if args.outdir is None else os.path.join(args.outdir, os.path.splitext(os.path.basename(_left))[0] + '_mono_left.png')
                    cv2.imwrite(_output, mono_left_jet)

            mono_left = torch.from_numpy(mono_left).unsqueeze(0).unsqueeze(0)
            mono_right = torch.from_numpy(mono_right).unsqueeze(0).unsqueeze(0)
            mono_depths = torch.cat([mono_left, mono_right], 0)
            mono_depths = (mono_depths - mono_depths.min()) / (mono_depths.max() - mono_depths.min())
            mono_left = mono_depths[0].unsqueeze(0).cuda().to(dtype)
            mono_right = mono_depths[1].unsqueeze(0).cuda().to(dtype)
        else:
            mono_left = None
            mono_right = None

        left_image = cv2.resize(left_image, (round(original_shape[1] / args.iscale), round(original_shape[0] / args.iscale)))
        right_image = cv2.resize(right_image, (round(original_shape[1] / args.iscale), round(original_shape[0] / args.iscale)))
        mono_left = F.interpolate(mono_left, size=(round(original_shape[0] / args.iscale), round(original_shape[1] / args.iscale)), mode='bilinear', align_corners=False) if mono_left is not None else None
        mono_right = F.interpolate(mono_right, size=(round(original_shape[0] / args.iscale), round(original_shape[1] / args.iscale)), mode='bilinear', align_corners=False) if mono_right is not None else None

        # Prepare inputs
        left_image = (torch.from_numpy(left_image).permute(2, 0, 1).unsqueeze(0) / 255.0).cuda().to(dtype)
        right_image = (torch.from_numpy(right_image).permute(2, 0, 1).unsqueeze(0) / 255.0).cuda().to(dtype) 

        inputs = (left_image, right_image, mono_left, mono_right)
        with autocast('cuda', enabled=args.mixed_precision):
            pred_disp = optimized_model(*inputs)
        
        pred_disp = pred_disp.detach().squeeze().float().cpu().numpy()
        pred_disp = cv2.resize(pred_disp, (original_shape[1], original_shape[0]))

        # Save the output
        _output = f"{os.path.splitext(_left)[0]}.npy" if args.outdir is None else os.path.join(args.outdir, os.path.splitext(os.path.basename(_left))[0] + '.npy')
        np.save(_output, pred_disp)
        
        if args.save_qualitatives or args.display_qualitatives:
            pred_disp = (pred_disp - pred_disp.min()) / (pred_disp.max() - pred_disp.min())

            pred_disp = (pred_disp * 255).astype(np.uint8)
            pred_disp = cv2.applyColorMap(pred_disp, cv2.COLORMAP_JET)

            if args.save_qualitatives:
                _output = f"{os.path.splitext(_left)[0]}_disp_jet.png" if args.outdir is None else os.path.join(args.outdir, os.path.splitext(os.path.basename(_left))[0] + '_disp_jet.png')
                cv2.imwrite(_output, pred_disp)

            if args.display_qualitatives:
                cv2.imshow("Disparity", pred_disp)
                cv2.waitKey(1)
                
    cv2.destroyAllWindows()

        
if __name__ == "__main__":
    main()
