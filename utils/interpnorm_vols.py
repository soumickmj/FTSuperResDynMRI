import os
import random
from glob import glob

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

__author__ = "Soumick Chatterjee, Chompunuch Sarasaen"
__copyright__ = "Copyright 2020, Faculty of Computer Science, Otto von Guericke University Magdeburg, Germany"
__credits__ = ["Soumick Chatterjee", "Chompunuch Sarasaen"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Soumick Chatterjee"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Production"

path_woZPad = "/project/schatter/Chimp/Data/usCHAOSWoT2/Center6p25MaskWoPad"
path_GT = "/project/schatter/Chimp/Data/hrCHAOS"

outpath_interpNorm = "/project/schatter/Chimp/Data/usCHAOSWoT2/Center6p25MaskWoPad_Tri_Norm"
outpath_GTNorm = "/project/schatter/Chimp/Data/hrCHAOS_Norm"

files = glob(path_woZPad+"/**/*.nii", recursive=True) + glob(path_woZPad+"/**/*.nii.gz", recursive=True)
files_gt = glob(path_GT+"/**/*.nii", recursive=True) + glob(path_GT+"/**/*.nii.gz", recursive=True)

def SaveNIFTI(data, path):
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    nib.save(nib.Nifti1Image(data, np.eye(4)), path) 

for file in tqdm(files):
    filename = os.path.basename(file)
    gt_files = [f for f in files_gt if filename in f]
    gt_path = gt_files[0]

    gt = nib.load(gt_path).dataobj[...]
    gt_max = gt.max()
    gt = (gt.astype(np.float32))/gt_max
    SaveNIFTI(gt, gt_path.replace(path_GT, outpath_GTNorm))

    images = nib.load(file).dataobj[...]
    img_max = images.max()
    images =  torch.from_numpy(images.astype(np.float32))
    images = F.interpolate(images.unsqueeze(0).unsqueeze(0), mode="trilinear", size=gt.shape).squeeze()
    images = (images/img_max).numpy()
    SaveNIFTI(images, file.replace(path_woZPad, outpath_interpNorm))
