import os
from copy import deepcopy
from statistics import median

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import torchcomplex.nn.functional as cF
import torchio as tio
import torchvision.utils as vutils
import wandb
from sewar.full_ref import ssim as SSIM2DCalc
from sewar.full_ref import uqi as UQICalc
from skimage.metrics import (normalized_root_mse, peak_signal_noise_ratio,
                             structural_similarity)
from torchcomplex.utils.signaltools import resample

__author__ = "Soumick Chatterjee, Chompunuch Sarasaen"
__copyright__ = "Copyright 2020, Faculty of Computer Science, Otto von Guericke University Magdeburg, Germany"
__credits__ = ["Soumick Chatterjee", "Chompunuch Sarasaen"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Soumick Chatterjee"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Production"


class Interpolator():
    def __init__(self, mode=None):
        if mode in ["sinc", "nearest", "linear", "bilinear", "bicubic", "trilinear", "area"]:
            self.mode = mode
        else:
            self.mode = None

    def perform_sinc(self, images, out_shape):
        axes = np.argwhere(np.equal(images.shape[2:], out_shape) == False).squeeze(1) #2 dims for batch and channel
        out_shape = [out_shape[i] for i in axes]
        return resample(images, out_shape, axis=axes+2) #2 dims for batch and channel

    def __call__(self, images, out_shape):
        if self.mode is None:
            return images
        elif images.is_complex():
            return cF.interpolate(images, size=out_shape, mode=self.mode)
        elif self.mode == "sinc":
            return self.perform_sinc(images, out_shape)
        else:
            return F.interpolate(images, size=out_shape, mode=self.mode)

def tensorboard_images(writer, inputs, outputs, targets, epoch, section='train'):
    writer.add_image('{}/output'.format(section),
                     vutils.make_grid(outputs[0, 0, ...],
                                      normalize=True,
                                      scale_each=True),
                     epoch)
    if inputs is not None:
        writer.add_image('{}/input'.format(section),
                        vutils.make_grid(inputs[0, 0, ...],
                                        normalize=True,
                                        scale_each=True),
                        epoch)
    if targets is not None:
        writer.add_image('{}/target'.format(section),
                        vutils.make_grid(targets[0, 0, ...],
                                        normalize=True,
                                        scale_each=True),
                        epoch)

def SaveNIFTI(data, file_path):
    """Save a NIFTI file using given file path from an array
    Using: NiBabel"""
    if(np.iscomplex(data).any()):
        data = abs(data)
    nii = nib.Nifti1Image(data, np.eye(4)) 
    nib.save(nii, file_path)

def process_valBatch(batch):
    inp = []
    gt = []
    gt_flag = []
    for i in range(len(batch['tag'])):
        gt_flag.append(True)
        batch_tag = batch['tag'][i]
        if batch_tag == "CorruptNGT" or batch_tag == "GTOnly":
            inp.append(batch['inp'][tio.DATA][i,...])
            gt.append(batch['gt'][tio.DATA][i,...])
        elif batch_tag == "FlyCorrupt":
            gt.append(batch['im'][tio.DATA][i,0,...].unsqueeze(1))
            if batch['im'][tio.DATA].shape[1] == 2:
                inp.append(batch['im'][tio.DATA][i,1,...].unsqueeze(1))
            else: #Use motion free image
                inp.append(deepcopy(gt[-1]))
        elif batch_tag == "CorruptOnly":
            inp.append(batch['inp'][tio.DATA][i,...])
            gt.append(batch['inp'][tio.DATA][i,...])
            gt_flag[-1] = False
    inp = torch.stack(inp,dim=0)
    gt = torch.stack(gt,dim=0)
    return inp, gt, gt_flag

def getSSIM(gt, out, gt_flag=None, data_range=1):
    if gt_flag is None:
        gt_flag = np.ones(gt.shape[0])
    vals = []
    for i in range(gt.shape[0]):
        if not bool(gt_flag[i]):
            continue
        for j in range(gt.shape[1]):
            vals.append(structural_similarity(gt[i,j,...], out[i,j,...], data_range=data_range))
    return median(vals)

def calc_metircs(gt, out, tag):
    ssim, ssimMAP = structural_similarity(gt, out, data_range=1, full=True)
    nrmse = normalized_root_mse(gt, out)
    psnr = peak_signal_noise_ratio(gt, out, data_range=1)
    uqi = UQICalc(gt, out)
    metrics = {
        "SSIM"+tag: ssim, 
        "NRMSE"+tag: nrmse, 
        "PSNR"+tag: psnr, 
        "UQI"+tag: uqi
    }
    return metrics, ssimMAP

def convert_image(img, source, target):
    """
    Convert an image from a source format to a target format.

    :param img: image
    :param source: source format, one of 'pil' (PIL image), '[0, 1]' or '[-1, 1]' (pixel value ranges)
    :param target: target format, one of 'pil' (PIL image), '[0, 255]', '[0, 1]', '[-1, 1]' (pixel value ranges),
                   'imagenet-norm' (pixel values standardized by imagenet mean and std.),
                   'y-channel' (luminance channel Y in the YCbCr color format, used to calculate PSNR and SSIM)
    :return: converted image
    """
    assert source in {'pil', '[0, 1]', '[-1, 1]'}, "Cannot convert from source format %s!" % source
    assert target in {'pil', '[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm',
                      'y-channel'}, "Cannot convert to target format %s!" % target

    # Convert from source to [0, 1]
    if source == 'pil':
        img = FT.to_tensor(img)

    elif source == '[0, 1]':
        pass  # already in [0, 1]

    elif source == '[-1, 1]':
        img = (img + 1.) / 2.

    # Convert from [0, 1] to target
    if target == 'pil':
        img = FT.to_pil_image(img)

    elif target == '[0, 255]':
        img = 255. * img

    elif target == '[0, 1]':
        pass  # already in [0, 1]

    elif target == '[-1, 1]':
        img = 2. * img - 1.

    elif target == 'imagenet-norm':
        if img.ndimension() == 3:
            img = (img - imagenet_mean) / imagenet_std
        elif img.ndimension() == 4:
            img = (img - imagenet_mean_cuda) / imagenet_std_cuda

    elif target == 'y-channel':
        # Based on definitions at https://github.com/xinntao/BasicSR/wiki/Color-conversion-in-SR
        # torch.dot() does not work the same way as numpy.dot()
        # So, use torch.matmul() to find the dot product between the last dimension of an 4-D tensor and a 1-D tensor
        img = torch.matmul(255. * img.permute(0, 2, 3, 1)[:, 4:-4, 4:-4, :], rgb_weights) / 255. + 16.

    return img


class ResSaver():
    def __init__(self, out_path, save_inp=False, do_norm=False):
        self.out_path = out_path
        self.save_inp = save_inp
        self.do_norm = do_norm

    def CalcNSave(self, out, inp, gt, outfolder, already_numpy=False):
        outpath = os.path.join(self.out_path, outfolder)
        os.makedirs(outpath, exist_ok=True)

        if not already_numpy:
            inp = inp.numpy()
            out = out.numpy()
        SaveNIFTI(out, os.path.join(outpath, "out.nii.gz"))

        if self.save_inp:            
            SaveNIFTI(inp, os.path.join(outpath, "inp.nii.gz"))

        if gt is not None: 
            if not already_numpy:           
                gt = gt.numpy()

            if self.do_norm:
                out = convert_image(out, source='[-1, 1]', target='[0, 1]') #out/out.max()
                inp = convert_image(inp, source='[-1, 1]', target='[0, 1]') #inp/inp.max()
                gt = convert_image(gt, source='[-1, 1]', target='[0, 1]') #gt/gt.max()

            out_metrics, out_ssimMAP = calc_metircs(gt, out, tag="Out")
            SaveNIFTI(out_ssimMAP, os.path.join(outpath, "ssimMAPOut.nii.gz"))

            inp_metrics, inp_ssimMAP = calc_metircs(gt, inp, tag="Inp")
            SaveNIFTI(inp_ssimMAP, os.path.join(outpath, "ssimMAPInp.nii.gz"))

            metrics = {**out_metrics, **inp_metrics}
            return metrics


#The WnB functions are here, but not been tested (even not finished)
def WnB_ArtefactLog_DS(run, datasets, meta={}, names = ["training", "validation", "test"], description="Train-Val(-Test) Split"):
    raw_data = wandb.Artifact("DSSplit", 
                                type="dataset",
                                description=description,
                                metadata={"sizes": [len(dataset) for dataset in datasets], **meta})

    for name, dataset in zip(names, datasets):
        with raw_data.new_file(name + ".npz", mode="wb") as file:
            np.savez(file, ds=dataset)
        run.log_artifact(raw_data)

def WnB_ReadArtefact_DS(run, tag="latest", names = ["training", "validation", "test"]):
    raw_data_artifact = run.use_artifact('DSSplit:'+tag)
    raw_dataset = raw_data_artifact.download()
    datasets = []
    for split in names:
        raw_split = np.load(os.path.join(raw_dataset, split + ".npz"))['ds']
        datasets.append(raw_split)
    return datasets

def WnB_ArtefactLog_Model(run, model, config, description="MyModel"):
    model_artifact = wandb.Artifact("Model", 
                                    type="model",
                                    description=description,
                                    metadata=dict(config))
    model.save("initialized_model.keras")
    model_artifact.add_file("initialized_model.keras")
    wandb.save("initialized_model.keras")
    run.log_artifact(model_artifact)

def WnB_ReadArtefact_Model(run, tag="latest"):
    model_artifact = run.use_artifact('Model:'+tag)
    model_dir = model_artifact.download()
    model_path = os.path.join(model_dir, "initialized_model.keras")
    model = keras.models.load_model(model_path)
    model_config = model_artifact.metadata
    return model, model_config
