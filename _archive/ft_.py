import argparse
import logging
import math
import os
import random
import statistics
import sys

import numpy as np
import torch
import torch.autograd.profiler as profiler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchio as tio
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import wandb
from models import *
from models.ReconResNet import ResNet
from models.ShuffleUNet.net import ShuffleUNet
from models.ThisNewNet import ThisNewNet
from utils.data import *
from utils.datasets import SRDataset
from utils.pLoss.perceptual_loss import PerceptualLoss
from utils.utilities import getSSIM, tensorboard_images

__author__ = "Soumick Chatterjee, Chompunuch Sarasaen"
__copyright__ = "Copyright 2020, Faculty of Computer Science, Otto von Guericke University Magdeburg, Germany"
__credits__ = ["Soumick Chatterjee", "Chompunuch Sarasaen"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Soumick Chatterjee"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Production"

modelIDs = {
    0: "UNET",
    1: "SRCNN",
    2: "SRCNNv2",
    3: "SRCNNv3",
    4: "UNETvSeg",
    5: "UNETvSegDS",
    6: "DenseNet",
    7: "UNETSRCNN",
    8: "SRCNNUNET",
    9: "ReconResNet",
    10: "ShuffleUNet",
    11: "UNETMSS",
}

lossIDs = {
    0: "pLoss",
    1: "MAE",
    2: "MultiSSIM",
    3: "SSIM3D"
}

def parseARGS():
    ap = argparse.ArgumentParser()
    ap.add_argument("-g", "--gpu", default="0", help="GPU ID(s).")
    ap.add_argument("--seed", default=2020, type=int, help="Seed")
    ap.add_argument("-ds", "--dataset", default=r'/mnt/public/sarasaen/Data/StaticFT/ChimpAbdomen/Protocol2/', help="Path to Dataset Folder.")
    ap.add_argument("-us", "--us", default='Center6p25MaskWoPad', help="Undersample.")
    ap.add_argument("-s", "--scalefact", default='(1,1,1)', help="Scaling Factor. For Zero padded data, set the dim to 1. [As a 3 valued tuple, factor for each dim. Supply seperated by coma or as a tuple, no spaces in between.].")
    ap.add_argument("-uf", "--usfolder", default='usTest', help="Undersampled Folder.")
    ap.add_argument("-hf", "--hrfolder", default='hrTest', help="HighRes (Fully-sampled) Folder.")
    ap.add_argument("-o", "--outfolder", default='staticTPSR', help="Output Folder.")
    ap.add_argument("-ms", "--modelsuffix", default='NewFT', help="Any Suffix To Add with the Model Name.")
    ap.add_argument("-bs", "--batchsize", type=int, default=96, help="Batch Size.")
    ap.add_argument("-nw", "--nworkers", type=int, default=8, help="Number of Workers.")

    ap.add_argument("-cp", "--chkpoint", default=None, help="Checkpoint (of the current training) to Load.")
    ap.add_argument("-cpft", "--chkpointft", default="", help="(To be used for Fine-Tuning) Checkpoint to Load for Fine-Tuning.")
    ap.add_argument("-c", "--cuda", type=bool, default=True, help="Use CUDA.")
    ap.add_argument("-mg", "--mulgpu", type=bool, default=False, help="Use Multiple GPU.")
    ap.add_argument("-amp", "--amp", type=bool, default=True, help="Use AMP.")
    ap.add_argument("-v", "--val", type=bool, default=False, help="Do Validation.")
    ap.add_argument("-vp", "--valdsper", type=float, default=0.3, help="Percentage of the DS to be used for Validation.")
    ap.add_argument("-p", "--profile", type=bool, default=False, help="Do Model Profiling.")

    ap.add_argument("-ep", "--epochs", type=int, default=150, help="Total Number of Epochs. To use Number of Iterations, set it to None")
    ap.add_argument("-it", "--iterations", type=int, default=1e6, help="Total Number of Iterations. To be used if number of Epochs is None")
    ap.add_argument("-lr", "--lr", type=float, default=1e-4, help="Total Number of Epochs.")
    ap.add_argument("-ps", "--patchsize", default='(24,24,24)', help="Patch Size. Supply seperated by coma or as a tuple, no spaces in between. Set it to None if not desired.")
    ap.add_argument("-pst", "--patchstride", default='(1,1,1)', help="Stride of patches, to be used during validation")
    ap.add_argument("-l", "--logfreq", type=int, default=10, help="log Frequency.")
    ap.add_argument("-sf", "--savefreq", type=int, default=1, help="saving Frequency.")
    ap.add_argument("-ml", "--medianloss", type=int, default=True, help="Use Median to get loss value (Final Reduction).")

    ap.add_argument(
        "-mid",
        "--modelid",
        type=int,
        default=0,
        help=f"Model ID.{str(modelIDs)}",
    )

    ap.add_argument("-mbn", "--batchnorm", type=bool, default=False, help="(Only for Model ID 0, 11) Do BatchNorm.")
    ap.add_argument("-mum", "--upmode", default='upconv', help="(Only for Model ID 0, 11) UpMode for model ID 0 and 11: [upconv, upsample], for model ID 9: [convtrans, <interp algo>]")
    ap.add_argument("-mdp", "--mdepth", type=int, default=3, help="(Only for Model ID 0, 6, 11) Depth of the Model.")
    ap.add_argument("-d", "--dropprob", type=float, default=0.0, help="(Only for Model ID 0, 6, 11) Dropout Probability.")
    ap.add_argument("-mslvl", "--msslevel", type=int, default=2, help="(Only for Model ID 11) Depth of the Model.")
    ap.add_argument("-msltn", "--msslatent", type=int, default=1, help="(Only for Model ID 11) Use the latent as one of the MSS level.")
    ap.add_argument("-msup", "--mssup", default="trilinear", help="(Only for Model ID 11) Interpolation to use on the MSS levels.")
    ap.add_argument("-msinb4", "--mssinterpb4", type=int, default=1, help="(Only for Model ID 11) Apply Interpolation before applying conv for the MSS levels. If False, interp will be applied after conv.")
    ap.add_argument("-nc", "--nchannel", type=int, default=1, help="Number of Channels in the Data.")
    ap.add_argument("-is", "--inshape", default='(256,256,30)', help="Input Shape. Supply seperated by coma or as a tuple, no spaces in between. Will only be used if Patch Size is None.")
    ap.add_argument("-f", "--nfeatures", type=int, default=64, help="(Not for DenseNet) N Starting Features of the Network.")
    ap.add_argument(
        "-lid", "--lossid", type=int, default=0, help=f"Loss ID.{str(lossIDs)}"
    )

    ap.add_argument("-plt", "--plosstyp", default="L1", help="(Only for Loss ID 0) Perceptual Loss Type.")
    ap.add_argument("-pll", "--plosslvl", type=int, default=3, help="(Only for Loss ID 0) Perceptual Loss Level.")
    ap.add_argument("-lrd", "--lrdecrate", type=int, default=100, help="(To be used for Fine-Tuning) Factor by which lr will be divided to find the actual lr. Set it to 1 if not desired")
    ap.add_argument("-ft", "--finetune", type=int, default=1, help="Is it a Fine-tuning traing or not (main-train).")
    ap.add_argument("-ftep", "--fteprt", type=float, default=0.001, help="(To be used for Fine-Tuning) Fine-Tune Epoch Rate.")
    ap.add_argument("-ftit", "--ftitrt", type=float, default=0.10, help="(To be used for Fine-Tuning, if fteprt is None) Fine-Tune Iteration Rate.")
    ap.add_argument("-int", "--preint", default="trilinear", help="Pre-interpolate before sending it to the Network. Set it to None if not needed.")
    ap.add_argument("-nrm", "--prenorm", default=True, type=bool, help="Rescale intensities beteen 0 and 1")    

    ap.add_argument("-tls", "--tnnlslc", type=int, default=2, help="Solo per ThisNewNet. loss_slice_count. Default 2")
    ap.add_argument("-tli", "--tnnlinp", type=int, default=1, help="Solo per ThisNewNet. loss_inplane. Default 1")

    #WnB related params
    ap.add_argument("-wnb", "--wnbactive", type=bool, default=True, help="WandB: Whether to use or not")
    ap.add_argument("-wnbp", "--wnbproject", default='SuperResMRI', help="WandB: Name of the project")
    ap.add_argument("-wnbe", "--wnbentity", default='mickchimp', help="WandB: Name of the entity")
    ap.add_argument("-wnbg", "--wnbgroup", default='staticTPSR', help="WandB: Name of the group")
    ap.add_argument("-wnbpf", "--wnbprefix", default='', help="WandB: Prefix for TrainID")
    ap.add_argument("-wnbml", "--wnbmodellog", default='all', help="WandB: While watching the model, what to save: gradients, parameters, all, None")
    ap.add_argument("-wnbmf", "--wnbmodelfreq", type=int, default=100, help="WandB: The number of steps between logging gradients")

    return ap.parse_args()

args = parseARGS()
# os.environ["TMPDIR"] = "/scratch/schatter/tmp"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
torch.set_num_threads(1)
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == "__main__" :
    args.scalefact = tuple(map(int, args.scalefact.replace('(','').replace(')','').split(',')))  
    args.homepath = os.path.expanduser("~/Documents")
    if args.patchsize:
        args.patchsize = tuple(map(int, args.patchsize.replace('(','').replace(')','').split(',')))
    if args.patchstride:
        args.patchstride = tuple(map(int, args.patchstride.replace('(','').replace(')','').split(',')))
    if args.inshape:
        args.inshape = tuple(map(int, args.inshape.replace('(','').replace(')','').split(',')))
    args.modelname = args.usfolder + "_" + modelIDs[args.modelid] + args.modelsuffix   

    if args.modelid == 0 or args.modelid == 6 or args.modelid == 11: 
        args.modelname += "do" + str(args.dropprob) +  "dp" + str(args.mdepth)
    if args.modelid == 0 or args.modelid == 11:        
        args.modelname += args.upmode
        if args.batchnorm:
            args.modelname += "BN"    
    if args.modelid == 11:
        args.modelname += "MSS"+str(args.msslevel)
        args.modelname += "Latent" if args.msslatent else "NoLatent"
        args.modelname += args.mssup
        args.modelname += "InterpB4" if args.mssinterpb4 else "NoInterpB4"
    trainID = args.modelname + '_' + args.us + '_' + lossIDs[args.lossid]
    if args.lossid == 0:
        trainID += args.plosstyp + 'lvl' + str(args.plosslvl)
    if args.finetune:
        trainID += "_FT_lrdec" + str(args.lrdecrate)
        if args.fteprt:
            trainID += "_eprt" + str(args.fteprt)
        else:
            trainID += "_itrt" + str(args.ftitrt)

    print("Training: "+trainID)
    
    if args.modelid == 2:
        SRCNN3D = SRCNN3Dv2
    elif args.modelid == 3:
        SRCNN3D = SRCNN3Dv3

    if args.medianloss:
        loss_reducer = statistics.median
    else:
        loss_reducer = statistics.mean

    dir_path = args.dataset + args.usfolder+ '/' + args.us + '/'
    label_dir_path = args.dataset + args.hrfolder + '/'

    log_path = os.path.join(args.dataset, args.outfolder, 'TBLogs', trainID)
    save_path = os.path.join(args.dataset, args.outfolder, trainID)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    tb_writer = SummaryWriter(log_dir = log_path)
    os.makedirs(save_path, exist_ok=True)
    logname = os.path.join(args.homepath, 'log_'+trainID+'.txt')

    logging.basicConfig(filename=logname,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

    transforms = []
    if not args.patchsize:
        transforms.append(tio.transforms.CropOrPad(target_shape=args.inshape))

    trainDS = SRDataset(logger=logging, patch_size=args.patchsize[0], dir_path=dir_path, label_dir_path=label_dir_path, #TODO: implement non-iso patch-size, now only using the first element
                        stride_depth=args.patchstride[2], stride_length=args.patchstride[0], stride_width=args.patchstride[1], Size=None, fly_under_percent=None, #TODO: implement fly_under_percent, if needed 
                        patch_size_us=None, pre_interpolate=args.preint, norm_data=args.prenorm, pre_load=True) #TODO implement patch_size_us if required - patch_size//scaling_factor

    model_scale_factor=tuple(np.roll(args.scalefact,shift=1))

    if args.val:
        train_size = int((1-args.valdsper) * len(trainDS))
        val_size = len(trainDS) - train_size
        trainDS, valDS = torch.utils.data.random_split(trainDS, [train_size, val_size])
    else:
        valDS = None

    if bool(args.patchsize):
        args.inshape = args.patchsize
    
    train_loader = DataLoader(dataset=trainDS, batch_size=args.batchsize,shuffle=True, num_workers=args.nworkers, pin_memory=True)
    val_loader = None if not args.val else DataLoader(dataset=valDS,batch_size=args.batchsize,shuffle=False, num_workers=args.nworkers, pin_memory=True)

    if args.modelid == 0:
        model = UNet(in_channels=args.nchannel, n_classes=args.nchannel, depth=args.mdepth, wf=round(math.log(args.nfeatures,2)), batch_norm=args.batchnorm, up_mode=args.upmode, dropout=bool(args.dropprob))
    elif (args.modelid == 1) or (args.modelid == 2) or (args.modelid == 3):
        model = SRCNN3D(n_channels=args.nchannel, scale_factor=model_scale_factor, num_features=args.nfeatures)    
    elif (args.modelid == 4) or (args.modelid == 5):
        model = UNetVSeg(in_ch=args.nchannel, out_ch=args.nchannel, n1=args.nfeatures)   
    elif args.modelid == 6:
        model = DenseNet(model_depth=args.mdepth, n_input_channels=args.nchannel, num_classes=args.nchannel, drop_rate=args.dropprob)
    elif (args.modelid == 7) or (args.modelid == 8):
        model = ThisNewNet(in_channels=args.nchannel, n_classes=args.nchannel, depth=args.mdepth, batch_norm=args.batchnorm, up_mode=args.upmode, dropout=bool(args.dropprob), 
                            scale_factor=model_scale_factor, num_features=args.nfeatures, sliceup_first=True if args.modelid==8 else False, 
                            loss_slice_count=args.tnnlslc, loss_inplane=args.tnnlinp)
    elif args.modelid == 9:
        model=ResNet(n_channels=args.nchannel,is3D=True,res_blocks=14,starting_nfeatures=args.nfeatures,updown_blocks=2,is_relu_leaky=True, #TODO: put all params as args
                    do_batchnorm=args.batchnorm, res_drop_prob=0.2,out_act="sigmoid",forwardV=0, upinterp_algo=args.upmode, post_interp_convtrans=False)
    elif args.modelid == 10:
        model=ShuffleUNet(in_ch=args.nchannel, num_features=args.nfeatures, out_ch=args.nchannel)
    elif args.modelid == 11:
        model = UNetMSS(in_channels=args.nchannel, n_classes=args.nchannel, depth=args.mdepth, wf=round(math.log(args.nfeatures,2)), 
                        batch_norm=args.batchnorm, up_mode=args.upmode, dropout=bool(args.dropprob),
                        mss_level=args.msslevel, mss_fromlatent=args.msslatent, mss_up=args.mssup, mss_interpb4=args.mssinterpb4)
    else:
        sys.exit("Invalid Model ID")

    if args.modelid == 5:
        IsDeepSup = True
    else:
        IsDeepSup = False

    if args.profile:
        dummy = torch.randn(args.batchsize, args.nchannel, *args.inshape)
        with profiler.profile(profile_memory=True, record_shapes=True, use_cuda=True) as prof:
            model(dummy)
            prof.export_chrome_trace(os.path.join(save_path, 'model_trace'))

    args.lr = args.lr/args.lrdecrate
    optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    model.to(device)

    if args.lossid == 0:
        if args.nchannel != 1:
            sys.exit("Perceptual Loss used here only works for 1 channel images")
        loss_func = PerceptualLoss(device=device, loss_model="unet3Dds", resize=None, loss_type=args.plosstyp, n_level=args.plosslvl)
    elif args.lossid == 1:
        loss_func = nn.L1Loss(reduction='mean')
    elif args.lossid == 2:
        loss_func = MultiSSIM(data_range=1, n_channels=args.nchannel, reduction='mean').to(device)
    elif args.lossid == 3:
        loss_func = SSIM(data_range=1, channel=args.nchannel, spatial_dims=3).to(device)
    else:
        sys.exit("Invalid Loss ID")
    if (args.lossid == 0 and args.plosstyp == "L1") or (args.lossid == 1):
        IsNegLoss = False
    else:
        IsNegLoss = True

    if (args.modelid == 7) or (args.modelid == 8):
        model.loss_func = loss_func

    scaler = GradScaler(enabled=args.amp)

    if args.chkpoint:
        chk = torch.load(args.chkpoint, map_location=device)
    elif args.finetune:
        if args.chkpointft:
            chk = torch.load(args.chkpointft, map_location=device)
        else:
            sys.exit("Finetune can't be performed if chkpointft not supplied")
    else:
        chk = None
        start_epoch = 0
        best_loss = float('-inf') if IsNegLoss else float('inf')

    if chk is not None:
        model.load_state_dict(chk['state_dict'])
        optimizer.load_state_dict(chk['optimizer'])
        scaler.load_state_dict(chk['AMPScaler'])  
        best_loss = chk['best_loss']  
        start_epoch = chk['epoch'] + 1
        iterations = chk['iterations'] if 'iterations' in chk else 0 #TODO: hotfix for best models where iterations are not stored
        main_train_epcoh = (chk['main_train_epoch'] + 1) if 'main_train_epoch' in chk else start_epoch #only be used for finetune

    if args.finetune:
        if args.fteprt:
            args.epochs = int((main_train_epcoh*(1+args.fteprt)))
        else:
            args.iterations = int(iterations*args.ftitrt)
            n_ft_ep = int(args.iterations // len(train_loader))
            args.epochs = main_train_epcoh + n_ft_ep

    if args.epochs is None:
        args.epochs = int(args.iterations // len(train_loader) + 1)

    if start_epoch >= args.epochs:
        logging.error('Training should atleast be for one epoch. Adjusting to perform 1 epoch training')
        args.epochs = start_epoch+1

    if not args.wnbactive:
        os.environ["WANDB_MODE"] = "dryrun"

    with wandb.init(project=args.wnbproject, entity=args.wnbentity, group=args.wnbgroup, config=args, name=args.wnbprefix+trainID, id=args.wnbprefix+trainID, resume=True) as WnBRun:
        wandb.watch(model, log=args.wnbmodellog, log_freq=args.wnbmodelfreq)

        logging.info('Training Epochs: from {0} to {1}'.format(start_epoch, args.epochs-1))
        for epoch in range(start_epoch, args.epochs):
            #Train
            model.train()
            runningLoss = []
            train_loss = []
            print('Epoch '+ str(epoch)+ ': Train')
            for i, (images, gt) in enumerate(tqdm(train_loader)):
                images = images[:, None, ...].to(device)  
                gt = gt[:, None, ...].to(device)  

                with autocast(enabled=args.amp):
                    if type(model) is SRCNN3D:
                        output1, output2 = model(images)
                        loss1 = loss_func(output1, gt)
                        loss2 = loss_func(output2, gt)
                        loss = loss2 + loss1
                    elif type(model) is UNetVSeg:
                        if IsDeepSup:
                            sys.exit("Not Implimented yet")
                        else:
                            out, _, _ = model(images)
                            loss = loss_func(out, gt)
                    elif type(model) is ThisNewNet:
                        out, loss = model(images, gt=gt)
                    elif type(model) is UNetMSS:
                        out, mssout = model(images)
                        loss = loss_func(out, gt)
                        for mss in range(len(mssout)):
                            loss += model.mss_coeff[mss] * loss_func(mssout[mss], gt)
                    else:
                        out = model(images)
                        loss = loss_func(out, gt)

                if IsNegLoss:
                    loss = -loss

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                loss = round((-loss).data.item(),4) if IsNegLoss else round(loss.data.item(),4)
                train_loss.append(loss)
                runningLoss.append(loss)
                logging.info('[%d/%d][%d/%d] Train Loss: %.4f' % ((epoch+1), args.epochs, i, len(train_loader), loss))

                if i % args.logfreq == 0:
                    niter = epoch*len(train_loader)+i
                    tb_writer.add_scalar('Train/Loss', loss_reducer(runningLoss), niter)
                    wandb.log({"Epoch":epoch, "TrainLoss":loss_reducer(runningLoss)})#, step=niter)
                    # tensorboard_images(tb_writer, inp, out.detach(), gt, epoch, 'train')
                    runningLoss = []
            
            if args.finetune or (epoch % args.savefreq == 0):              
                checkpoint = {
                    'epoch': epoch,
                    'iterations': (epoch+1)*len(train_loader),
                    'best_loss': best_loss,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'AMPScaler': scaler.state_dict()         
                }
                torch.save(checkpoint, os.path.join(save_path, trainID+".pth.tar"))
                if args.modelid != 9:
                    torch.onnx.export(model, images, trainID+".onnx", input_names=["LRCurrTP"], output_names=["SuperResolvedCurrTP"])
                    wandb.save(trainID+".onnx")

            tb_writer.add_scalar('Train/EpochLoss', loss_reducer(train_loss), epoch)
            wandb.log({"TrainEpochLoss":loss_reducer(train_loss)})#, step=epoch)

            #Validate
            if val_loader:
                model.eval()
                with torch.no_grad():
                    runningLoss = []
                    val_loss = []
                    runningAcc = []
                    val_acc = []
                    print('Epoch '+ str(epoch)+ ': Val')
                    for i, (images, gt) in enumerate(tqdm(val_loader)):
                        images = images[:, None, ...].to(device)  
                        gt = gt[:, None, ...].to(device) 

                        with autocast(enabled=args.amp):
                            if type(model) is SRCNN3D:
                                output1, output2 = model(images)
                                loss1 = loss_func(output1, gt)
                                loss2 = loss_func(output2, gt)
                                loss = loss2 + loss1
                            elif type(model) is UNetVSeg:
                                if IsDeepSup:
                                    sys.exit("Not Implimented yet")
                                else:
                                    out, _, _ = model(images)
                                    loss = loss_func(out, gt)
                            elif type(model) is ThisNewNet:
                                out, loss = model(images, gt=gt)
                            else:
                                out = model(images)
                                loss = loss_func(out, gt)

                        ssim = getSSIM(gt.detach().cpu().numpy(), out.detach().cpu().numpy(), data_range=1)

                        loss = round((-loss).data.item(),4) if IsNegLoss else round(loss.data.item(),4)
                        val_loss.append(loss)
                        runningLoss.append(loss)
                        val_acc.append(ssim)
                        runningAcc.append(ssim)
                        logging.info('[%d/%d][%d/%d] Val Loss: %.4f' % ((epoch+1), args.epochs, i, len(val_loader), loss))
                        #For tensorboard
                        if i % args.logfreq == 0:
                            niter = epoch*len(val_loader)+i
                            tb_writer.add_scalar('Val/Loss', loss_reducer(runningLoss), niter)
                            wandb.log({"Epoch":epoch, "ValLoss":loss_reducer(runningLoss)})#, step=niter)
                            tb_writer.add_scalar('Val/SSIM', loss_reducer(runningAcc), niter)
                            wandb.log({"Epoch":epoch, "ValSSIM":loss_reducer(runningAcc)})#, step=niter)
                            # tensorboard_images(tb_writer, inp, out.detach(), gt, epoch, 'val')
                            runningLoss = []
                            runningAcc = []

                    if (loss_reducer(val_loss) < best_loss and not IsNegLoss) or (loss_reducer(val_loss) > best_loss and IsNegLoss):
                        best_loss = loss_reducer(val_loss)
                        WnBRun.summary["best_loss"] = best_loss
                        checkpoint = {
                            'epoch': epoch,
                            'best_loss': best_loss,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'AMPScaler': scaler.state_dict()         
                        }
                        torch.save(checkpoint, os.path.join(save_path, trainID+"_best.pth.tar"))
                        if args.modelid != 9:
                            torch.onnx.export(model, images, trainID+"_best.onnx", input_names=["LRCurrTP"], output_names=["SuperResolvedCurrTP"])
                            wandb.save(trainID+"_best.onnx")
                tb_writer.add_scalar('Val/EpochLoss', loss_reducer(val_loss), epoch)
                wandb.log({"ValEpochLoss":loss_reducer(val_loss)})#, step=epoch)
                tb_writer.add_scalar('Val/EpochSSIM', loss_reducer(val_acc), epoch)
                wandb.log({"ValEpochSSIM":loss_reducer(val_acc)})#, step=epoch)
