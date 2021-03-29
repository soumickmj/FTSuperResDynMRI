import fnmatch
import os
import random
from glob import glob

import numpy as np
import torch
import torchio as tio
from torchio.data.io import read_image

from .motion import MotionCorrupter

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2020, Faculty of Computer Science, Otto von Guericke University Magdeburg, Germany"
__credits__ = ["Soumick Chatterjee"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Soumick Chatterjee"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Production"

def create_trainDS(path, p=1, **kwargs):
    files = glob(path+"/**/*.nii", recursive=True) + glob(path+"/**/*.nii.gz", recursive=True)
    subjects = []
    for file in files:
        subjects.append(tio.Subject(
                                    im=tio.ScalarImage(file),
                                    filename=os.path.basename(file),
                        ))
    moco = MotionCorrupter(**kwargs)
    transforms = [
                    tio.Lambda(moco.perform, p = p)
                ]
    transform = tio.Compose(transforms)
    subjects_dataset = tio.SubjectsDataset(subjects, transform=transform)
    return subjects_dataset

def create_trainDS_precorrupt(path_gt, path_corrupt, p=1, norm_mode=0):
    files = glob(path_gt+"/**/*.nii", recursive=True) + glob(path_gt+"/**/*.nii.gz", recursive=True)
    subjects = []
    for file in files:
        subjects.append(tio.Subject(
                                    im=tio.ScalarImage(file),
                                    filename=os.path.basename(file),
                        ))
    transforms = [
                    ReadCorrupted(path_corrupt=path_corrupt, p=p, norm_mode=norm_mode)
                    
                ]
    transform = tio.Compose(transforms)
    subjects_dataset = tio.SubjectsDataset(subjects, transform=transform)
    return subjects_dataset

def createTIODS(path_gt, path_corrupt, is_infer=False, p=1, transforms = [], **kwargs):
    files_gt = glob(path_gt+"/**/*.nii", recursive=True) + glob(path_gt+"/**/*.nii.gz", recursive=True)
    if path_corrupt:
        files_inp = glob(path_corrupt+"/**/*.nii", recursive=True) + glob(path_corrupt+"/**/*.nii.gz", recursive=True)
        corruptFly = False
    else:
        files_inp = files_gt.copy()
        corruptFly = True
    subjects = []

    for file in files_inp:
        filename = os.path.basename(file)
        gt_files = [f for f in files_gt if filename in f]
        if len(gt_files) > 0:
            gt_path = gt_files[0]
            files_gt.remove(gt_path)
            subjects.append(tio.Subject(
                                        gt=tio.ScalarImage(gt_path),
                                        inp=tio.ScalarImage(file),
                                        filename=filename,
                                        tag="CorruptNGT",
                            ))

    if corruptFly:
        moco = MotionCorrupter(**kwargs)
        transforms.append(tio.Lambda(moco.perform, p = p))
    transform = tio.Compose(transforms)
    subjects_dataset = tio.SubjectsDataset(subjects, transform=transform)
    return subjects_dataset

def __process_TPs(files):
    f_dicts = []
    for f in files:
        f_info = {"path": f}
        f_parts = os.path.normpath(f).split(os.sep)
        tp = fnmatch.filter(f_parts, "TP*")[0]
        f_info["filename"] = "_".join(f_parts[f_parts.index(tp)+1:])
        f_info["tp"] = int(tp[2:])
        f_dicts.append(f_info)
    f_dicts = sorted(f_dicts, key=lambda k: k['tp'])    
    filenames = list(set(dic["filename"] for dic in f_dicts))
    return f_dicts, filenames       

class ProcessTIOSubsTPs():
    def __init__(self):
        pass

    def __call__(self, subject):
        gt_tp_prev = subject['gt_tp_prev'][tio.DATA]
        inp_tp = subject['inp'][tio.DATA]
        subject["inp"][tio.DATA] = torch.cat([gt_tp_prev, inp_tp], dim=0)
        return subject

def createTIODynDS(path_gt, path_corrupt, is_infer=False, p=1, transforms = [], **kwargs):
    files_gt = glob(path_gt+"/**/*.nii", recursive=True) + glob(path_gt+"/**/*.nii.gz", recursive=True)
    if path_corrupt:
        files_inp = glob(path_corrupt+"/**/*.nii", recursive=True) + glob(path_corrupt+"/**/*.nii.gz", recursive=True)
        corruptFly = False
    else:
        files_inp = files_gt.copy()
        corruptFly = True
    subjects = []

    inp_dicts, files_inp = __process_TPs(files_inp)
    gt_dicts, _ = __process_TPs(files_gt)
    for filename in files_inp:
        inp_files = [d for d in inp_dicts if filename in d['filename']]
        gt_files = [d for d in gt_dicts if filename in d['filename']]
        tps = list(set(dic["tp"] for dic in inp_files))
        tp_prev = tps.pop(0)
        for tp in tps:
            inp_tp_prev = [d for d in inp_files if tp_prev == d['tp']]
            gt_tp_prev = [d for d in gt_files if tp_prev == d['tp']]
            inp_tp = [d for d in inp_files if tp == d['tp']]
            gt_tp = [d for d in gt_files if tp == d['tp']]
            tp_prev = tp
            if len(gt_tp_prev) > 0 and len(gt_tp) > 0:
                subjects.append(tio.Subject(
                                        gt_tp_prev=tio.ScalarImage(gt_tp_prev[0]['path']),
                                        inp_tp_prev=tio.ScalarImage(inp_tp_prev[0]['path']),
                                        gt=tio.ScalarImage(gt_tp[0]['path']),
                                        inp=tio.ScalarImage(inp_tp[0]['path']),
                                        filename=filename,
                                        tp=tp,
                                        tag="CorruptNGT",
                                ))

            else:
                print("Warning: Not Implemented if GT is missing. Skipping Sub-TP.")
                continue

    if corruptFly:
        moco = MotionCorrupter(**kwargs)
        transforms.append(tio.Lambda(moco.perform, p = p))
    transforms.append(ProcessTIOSubsTPs())
    transform = tio.Compose(transforms)
    subjects_dataset = tio.SubjectsDataset(subjects, transform=transform)
    return subjects_dataset

def create_patchDS(train_subs, val_subs, patch_size, patch_qlen, patch_per_vol, inference_strides): 
    train_queue = None
    val_queue = None
    
    if train_subs is not None:
        sampler = tio.data.UniformSampler(patch_size)
        train_queue = tio.Queue(
                subjects_dataset=train_subs,
                max_length=patch_qlen,
                samples_per_volume=patch_per_vol,
                sampler=sampler,
                num_workers=0,
                start_background=True
            )

    if val_subs is not None:
        overlap = np.subtract(patch_size, inference_strides)
        grid_samplers = []    
        for i in range(len(val_subs)):
            grid_sampler = tio.inference.GridSampler(val_subs[i], patch_size, overlap)
            grid_samplers.append(grid_sampler)
        val_queue = torch.utils.data.ConcatDataset(grid_samplers)

    return train_queue, val_queue

class ReadCorrupted(tio.transforms.Transform):
    def __init__(self, path_corrupt, p=1, norm_mode=0):
        super().__init__(p=p)
        self.path_corrupt=path_corrupt
        self.norm_mode = norm_mode

    def apply_transform(self, subject):
        corrupted_query = subject.filename.split(".")[0]+"*"
        files = glob(self.path_corrupt+"/**/"+corrupted_query, recursive=True)
        corrupt_path = files[random.randint(0, len(files)-1)]
        transformed, _ = read_image(corrupt_path)

        vol = subject['im'][tio.DATA].float()
        transformed = transformed.float()
        if self.norm_mode==1:
            vol = vol/vol.max()
            transformed = transformed/transformed.max()
        elif self.norm_mode==2:
            vol = (vol-vol.min())/(vol.max()-vol.min())
            transformed = (transformed-transformed.min())/(transformed.max()-transformed.min())
        subject['im'][tio.DATA] = torch.cat([vol,transformed], 0)
        return subject
