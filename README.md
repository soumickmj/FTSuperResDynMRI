# Fine-tuning deep learning model parameters for improved super-resolution of dynamic MRI with prior-knowledge
Official code of the paper "Fine-tuning deep learning model parameters for improved super-resolution of dynamic MRI with prior-knowledge" (https://doi.org/10.1016/j.artmed.2021.102196 and https://arxiv.org/abs/2102.02711)

The pre-print of this work is available on ArXiv: https://arxiv.org/abs/2102.02711


Abstract of this work was presented at ISMRM 2021 (Abstract available on RG: https://www.researchgate.net/publication/349588965_Fine-tuning_deep_learning_model_parameters_for_improved_super-resolution_of_dynamic_MRI_with_prior-knowledge)

Furher extension of this work, DDoS model, also incorporates the temporal information to improve the reconstruction further, was presented as an abstract at ESMRMB 2021 (Abstract availalble on RG: https://www.researchgate.net/publication/354888919_DDoS_Dynamic_Dual-Channel_U-Net_for_Improving_Deep_Learning_Based_Super-Resolution_of_Abdominal_Dynamic_MRI)

This repository is dedicated to the Fine-Tuning project (paper link below), is not regluarly maintained currently. This repo holds the codes used in the original work. 

The final DDoS model and further developments of the DDoS project are on the new GitHub repo: https://github.com/soumickmj/DDoS

## Model Weights
The weights of the main training for the highest undersampling evaluated here (i.e. 6.25% of the centre of the k-space), trained on the CHAOS dataset, has been made publicly available on Huggingface: [https://huggingface.co/soumickmj/FTSRDyn_UNet3D_CHAOS_MT_Centre6p25MaskWoPad](https://huggingface.co/soumickmj/FTSRDyn_UNet3D_CHAOS_MT_Centre6p25MaskWoPad). The fine-tuned models (after main training) are not shared as they are subject specific. 

Since these models have been uploaded to Hugging Face following their format, but the pipeline here is incompatible with Hugging Face directly, they cannot be used as-is. Instead, the weights must be downloaded using the AutoModel class from the transformers package, saved as a checkpoint, and then the path to this saved checkpoint must be specified using "--chkpoint" (to perform further training as the an extension of the main training stage) or "--chkpointft" (to perform fine-tuning) parameter.

```python
from transformers import AutoModel
modelHF = AutoModel.from_pretrained("soumickmj/FTSRDyn_UNet3D_CHAOS_MT_Centre6p25MaskWoPad", trust_remote_code=True)
torch.save({'state_dict': modelHF.model.state_dict()}, "/path/to/checkpoint/model.pth")
```
To run this pipeline with these weights, the path to the checkpoint must then be passed as chkpoint or chkpointft, as an additional parameter along with the other desired parameters:
```bash
--chkpoint /path/to/checkpoint/model.pth
```
or
```bash
--chkpointft /path/to/checkpoint/model.pth
```

## Credits

If you like this repository, please click on Star!

If you use this approach in your research or use codes from this repository, please cite the following in your publications:

> [Chompunuch Sarasaen, Soumick Chatterjee, Mario Breitkopf, Georg Rose, Andreas Nürnberger, Oliver Speck: Fine-tuning deep learning model parameters for improved super-resolution of dynamic MRI with prior-knowledge (Artificial Intelligence in Medicine, Nov 2021)](https://doi.org/10.1016/j.artmed.2021.102196)

BibTeX entry:

```bibtex
@article{sarasaen2021fine,
title = {Fine-tuning deep learning model parameters for improved super-resolution of dynamic MRI with prior-knowledge},
journal = {Artificial Intelligence in Medicine},
pages = {102196},
year = {2021},
issn = {0933-3657},
doi = {https://doi.org/10.1016/j.artmed.2021.102196},
author = {Sarasaen, Chompunuch and Chatterjee, Soumick and Breitkopf, Mario and Rose, Georg and N{\"u}rnberger, Andreas and Speck, Oliver},
}
```

The complete manuscript is also on ArXiv:-
> [Chompunuch Sarasaen, Soumick Chatterjee, Mario Breitkopf, Georg Rose, Andreas Nürnberger, Oliver Speck: Fine-tuning deep learning model parameters for improved super-resolution of dynamic MRI with prior-knowledge (arXiv:2102.02711, Feb 2021)](https://arxiv.org/abs/2102.02711)

Thank you so much for your support.
