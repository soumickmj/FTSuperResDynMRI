import argparse
import logging
import os
import random
from statistics import median

import numpy as np
import pandas as pd
import torch
import torch.autograd.profiler as profiler
from torch.autograd import Variable
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.ReconResNet import ResNet
from models.ShuffleUNet.net3d import ShuffleUNet
from utils.data import *
from utils.utilities import ResSaver, process_valBatch

__author__ = "Soumick Chatterjee, Chompunuch Sarasaen"
__copyright__ = "Copyright 2020, Faculty of Computer Science, Otto von Guericke University Magdeburg, Germany"
__credits__ = ["Soumick Chatterjee", "Chompunuch Sarasaen"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Soumick Chatterjee"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Archived"

def parseARGS():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainID', action="store", default="")
    parser.add_argument('--gpu', action="store", default="6")
    parser.add_argument('--seed', action="store", default=1701, type=int)
    parser.add_argument('--num_workers', action="store", default=0, type=int)
    parser.add_argument('--batch_size', action="store", default=8, type=int)
    parser.add_argument('--gt_vols_test', action="store", default="")
    parser.add_argument('--corrupt_vols_test', action="store", default="")
    parser.add_argument('--log_path', action="store", default="")
    parser.add_argument('--save_path', action="store", default=r"D:\Rough\Scratch")
    parser.add_argument('--checkpoint', action="store", default="")
    parser.add_argument('--cuda', action="store_true", default=True)
    parser.add_argument('--amp', action="store_true", default=True)
    parser.add_argument('--do_profile', action="store_true", default=False)
    parser.add_argument('--non_deter', action="store_true", default=False)
    parser.add_argument('--save_inp', action="store_true", default=False, help="Save the input volumes in the Output folder")
    parser.add_argument('--do_norm', action="store_true", default=False, help="Normalise with dividing-by-max before calculating metrics")

    parser.add_argument('--input_shape', action="store", default="", help="length, width, depth ")    
    parser.add_argument('--log_freq', action="store", default=10, type=int, help="For Tensorboard logs, n_iteration")

    #Network Params
    parser.add_argument('--modelID', action="store", default=0, type=int, help="0: RecoNResNet, 1: ShuffleUNet")
    parser.add_argument('--n_channels', action="store", default=1, type=int)
    parser.add_argument('--model_res_blocks', action="store", default=14, type=int)
    parser.add_argument('--model_starting_nfeatures', action="store", default=64, type=int)
    parser.add_argument('--model_updown_blocks', action="store", default=2, type=int)
    parser.add_argument('--model_do_batchnorm', action="store_true", default=False)
    parser.add_argument('--model_forwardV', action="store", default=0, type=int)
    parser.add_argument('--model_upinterp_algo', action="store", default="convtrans", help='"convtrans", or interpolation technique: "sinc", "nearest", "linear", "bilinear", "bicubic", "trilinear", "area"')
    parser.add_argument('--model_post_interp_convtrans', action="store_true", default=False)

    #Controlling motion corruption, whether to run on the fly or use the pre-created ones. If corrupt_vols_test is not provided, only then the following params will be used
    parser.add_argument('--corrupt_prob', action="store", default=1.0, type=float, help="Probability of the corruption to be applied or corrupted volume to be used")
    parser.add_argument('--motion_mode', action="store", default=2, type=int, help="Mode 0: TorchIO's, 1: Custom direction specific")
    parser.add_argument('--motion_degrees', action="store", default=10, type=int)
    parser.add_argument('--motion_translation', action="store", default=10, type=int)
    parser.add_argument('--motion_num_transforms', action="store", default=10, type=int)
    parser.add_argument('--motion_image_interpolation', action="store", default='linear')
    parser.add_argument('--motion_norm_mode', action="store", default=2, type=int, help="norm_mode 0: No Norm, 1: Divide by Max, 2: MinMax")
    parser.add_argument('--motion_noise_dir', action="store", default=1, type=int, help="noise_dir 0 1 or 2 (only for motion_mode 1, custom direction specific). noise_dir=2 will act as motion_mode 0. noise_dir=-1 will randomly choose 0 or 1.")
    parser.add_argument('--motion_mu', action="store", default=0.0, type=float, help="Only for motion_mode 2")
    parser.add_argument('--motion_sigma', action="store", default=0.1, type=float, help="Only for motion_mode 2")
    parser.add_argument('--motion_random_sigma', action="store_true", default=False, help="Only for motion_mode 2 - to randomise the sigma value, treating the provided sigma as upper limit and 0 as lower")
    parser.add_argument('--motion_n_threads', action="store", default=8, type=int, help="Only for motion_mode 2 - to apply motion for each thread encoding line parallel, max thread controlled by this. Set to 0 to perform serially.")

    return parser.parse_args()

args = parseARGS()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
torch.backends.cudnn.benchmark = args.non_deter
if not args.non_deter:
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__" :
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
    tb_writer = SummaryWriter(log_dir = os.path.join(args.log_path,args.trainID))
    os.makedirs(args.save_path, exist_ok=True)
    logname = os.path.join(args.save_path, 'log_test_'+args.trainID+'.txt')

    logging.basicConfig(filename=logname,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

    motion_params = {k.split('motion_')[1]: v for k, v in vars(args).items() if k.startswith('motion')}
    testDS = createTIODS(args.gt_vols_test, args.corrupt_vols_test, is_infer=True, p=args.corrupt_prob, **motion_params)
    
    test_loader = DataLoader(dataset=testDS,batch_size=args.batch_size,shuffle=False, num_workers=args.num_workers)

    if args.modelID == 0:
        model_params = {k.split('model_')[1]: v for k, v in vars(args).items() if k.startswith('model_')}
        model=ResNet(n_channels=args.n_channels,is3D=True,**model_params)
    elif args.modelID == 1:
        model=ShuffleUNet(in_ch=args.n_channels, num_features=args.model_starting_nfeatures, out_ch=args.n_channels)

    if args.do_profile:
        dummy = torch.randn(args.batch_size, args.n_channels, *args.input_shape)
        with profiler.profile(profile_memory=True, record_shapes=True, use_cuda=True) as prof:
            model(dummy)
            prof.export_chrome_trace(os.path.join(args.save_path, 'model_trace'))
    model.to(device)

    chk = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(chk['state_dict'])
    trained_epoch = chk['epoch'] 
    model.eval()

    saver = ResSaver(os.path.join(args.save_path, "Results"), save_inp=args.save_inp, do_norm=args.do_norm)

    with torch.no_grad():
        runningSSIM = []
        test_ssim = []
        test_metrics = []
        print('Epoch '+ str(trained_epoch)+ ': Test')
        for i, batch in enumerate(tqdm(test_loader)):
            inp, gt, gt_flag = process_valBatch(batch)

            inp = Variable(inp).float().to(device)

            with autocast(enabled=args.amp):
                out = model(inp)
            out = out.type(inp.dtype) 

            for b in range(len(batch['filename'])):
                metrics = saver.CalcNSave(out[b,...].detach().cpu().squeeze(), inp[b,...].detach().cpu().squeeze(), gt[b,...].squeeze().float() if gt_flag[b] else None, batch['filename'][b].split(".")[0])

                if metrics is not None:
                    metrics['file'] = batch['filename']
                    test_metrics.append(metrics)

                    ssim = round(metrics['SSIMOut'],4)
                    test_ssim.append(ssim)
                    runningSSIM.append(ssim)
                    logging.info('[%d/%d] Test SSIM: %.4f' % (i, len(test_loader), ssim))
                    #For tensorboard
                    if i % args.log_freq == 0:
                        niter = len(test_loader)+i
                        tb_writer.add_scalar('Test/SSIM', median(runningSSIM), niter)
                        runningSSIM = []
    
    if len(test_metrics) > 0:
        print("Avg SSIM: "+str(median(test_ssim)))
        df = pd.DataFrame.from_dict(test_metrics)
        df.to_csv(os.path.join(args.save_path, 'Results.csv'), index=False)
