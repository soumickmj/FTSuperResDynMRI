import nibabel as nib
from skimage.metrics import structural_similarity as ssim
import os
import numpy as np
from glob import glob
from utils.utilities import calc_metircs
import pandas as pd
from tqdm import tqdm

def generateEnsb(dynMT, staticFT, GT, out):
    dynMT = nib.load(dynMT).get_fdata()
    staticFT = nib.load(staticFT).get_fdata()
    GT = nib.load(GT).get_fdata()

    GT = GT / GT.max()


    #let's ensamble without normalisation
    metrics_dynMT, ssimpMAP_dynMT = calc_metircs(GT, dynMT, "dynMT")
    nib.save(nib.Nifti1Image(ssimpMAP_dynMT, np.eye(4)), out+'_ssimMAP_dynMT.nii.gz')
    metrics_staticFT, ssimpMAP_staticFT = calc_metircs(GT, staticFT, "staticFT")
    nib.save(nib.Nifti1Image(ssimpMAP_staticFT, np.eye(4)), out+'_ssimMAP_staticFT.nii.gz')

    ensamble = (dynMT+staticFT) / 2
    metrics_ensamble, ssimpMAP_ensamble = calc_metircs(GT, ensamble, "ensamble")
    nib.save(nib.Nifti1Image(ssimpMAP_ensamble, np.eye(4)), out+'_ssimMAP_ensamble.nii.gz')
    nib.save(nib.Nifti1Image(ensamble, np.eye(4)), out+'_ensamble.nii.gz')  


    #let's ensamble with normalisation
    dynMT_normed = dynMT / dynMT.max()
    staticFT_normed = staticFT / staticFT.max()
    metrics_dynMT_normed, ssimpMAP_dynMT_normed = calc_metircs(GT, dynMT_normed, "dynMTNormed")
    nib.save(nib.Nifti1Image(ssimpMAP_dynMT_normed, np.eye(4)), out+'_ssimMAP_dynMTnormed.nii.gz')
    metrics_staticFT_normed, ssimpMAP_staticFT_normed = calc_metircs(GT, staticFT_normed, "staticFTNormed")
    nib.save(nib.Nifti1Image(ssimpMAP_staticFT_normed, np.eye(4)), out+'_ssimMAP_staticFTnormed.nii.gz')

    ensamble_normed = (dynMT_normed+staticFT_normed) / 2
    metrics_ensamble_normed, ssimpMAP_ensamble_normed = calc_metircs(GT, ensamble_normed, "ensambleNormed")
    nib.save(nib.Nifti1Image(ssimpMAP_ensamble_normed, np.eye(4)), out+'__ssimMAP_ensamblenormed.nii.gz')  
    nib.save(nib.Nifti1Image(ensamble_normed, np.eye(4)), out+'_outnormed.nii.gz')

    return {**metrics_dynMT, **metrics_staticFT, **metrics_ensamble, **metrics_dynMT_normed, **metrics_staticFT_normed, **metrics_ensamble_normed}
    
# #MickDynProtocol0
# dynMT = "/mnt/public/sarasaen/Data/CHAOSDynWoT2/dynDualChn/usTrain_UNETDSv1do0.0dp3upconv_Center6p25MaskWoPad_pLossL1lvl3_infstr3c3c3/Mick3DDyn0_woZpad/Results"
# staticFT = "/mnt/public/sarasaen/Data/StaticFT/MickAbdomen/Protocol1/staticTPSR/UNETValNewFTMickdo0.0dp3upconv_Center6p25MaskWoPad_pLossL1lvl3_FT_lrdec100_itrt0.1_infstr3c3c3/Mick3DDyn0_woZpad/Results"
# GT = "/mnt/public/sarasaen/Data/3DDynTest/MickAbdomen/DynProtocol0/hrTest"
# out = "/mnt/public/sarasaen/Data/CHAOSDynWoT2/dynDualChn/usTrain_UNETDSv1do0.0dp3upconv_Center6p25MaskWoPad_pLossL1lvl3_infstr3c3c3/NewEnsamble_Mick3DDyn0_woZpad"
# ignore_1stStatic = True

# #FatyDynProtocol0
# dynMT = "/mnt/public/sarasaen/Data/CHAOSDynWoT2/dynDualChn/usTrain_UNETDSv1do0.0dp3upconv_Center6p25MaskWoPad_pLossL1lvl3_infstr3c3c3/Faty3DDyn0_woZpad/Results"
# staticFT = "/mnt/public/sarasaen/Data/StaticFT/FatyAbdomen/Protocol1/staticTPSR/UNETValNewFTFatydo0.0dp3upconv_Center6p25MaskWoPad_pLossL1lvl3_FT_lrdec100_itrt0.1_infstr3c3c3/Faty3DDyn0_woZpad/Results"
# GT = "/mnt/public/sarasaen/Data/3DDynTest/FatyAbdomen/DynProtocol0/hrTest"
# out = "/mnt/public/sarasaen/Data/CHAOSDynWoT2/dynDualChn/usTrain_UNETDSv1do0.0dp3upconv_Center6p25MaskWoPad_pLossL1lvl3_infstr3c3c3/NewEnsamble_Faty3DDyn0_woZpad"
# ignore_1stStatic = True

#ChimpDynProtocol0
dynMT = "/mnt/public/sarasaen/Data/CHAOSDynWoT2/dynDualChn/usTrain_UNETDSv1do0.0dp3upconv_Center6p25MaskWoPad_pLossL1lvl3_infstr3c3c3/Chimp3DDyn0_woZpad/Results"
staticFT = "/mnt/public/sarasaen/Data/StaticFT/ChimpAbdomen/Protocol2/staticTPSR/usTest_UNETValNewFTdo0.0dp3upconv_Center6p25MaskWoPad_pLossL1lvl3_FT_lrdec100_eprt0.001/Chimp3DDyn0_woZpad/Results"
GT = "/mnt/public/sarasaen/Data/3DDynTest/ChimpAbdomen/DynProtocol0/hrTest"
out = "/mnt/public/sarasaen/Data/CHAOSDynWoT2/dynDualChn/usTrain_UNETDSv1do0.0dp3upconv_Center6p25MaskWoPad_pLossL1lvl3_infstr3c3c3/NewEnsamble_Chimp3DDyn0_woZpad"
ignore_1stStatic = True

dynMTs = sorted(glob(dynMT+"/*/"))
staticFTs = sorted(glob(staticFT+"/*/"))

if ignore_1stStatic:
    staticFTs = staticFTs[1:]

os.makedirs(os.path.join(out, "Results"), exist_ok=True)
metrics = []
for i in tqdm(range(len(dynMTs))):
    filename = os.path.basename(os.path.dirname(staticFTs[i])).replace(".nii.gz","")
    metrics.append(generateEnsb(dynMTs[i]+"out.nii.gz", staticFTs[i]+"out.nii.gz", os.path.join(GT, filename, filename+".nii.gz"), os.path.join(out, "Results", filename)))
df = pd.DataFrame.from_dict(metrics)
df.to_csv(os.path.join(out, "Results.csv"))