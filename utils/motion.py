import math
import multiprocessing.dummy as multiprocessing
import random
from collections import defaultdict
from typing import List

import numpy as np
import SimpleITK as sitk
import torch
import torchio as tio
from scipy.ndimage import affine_transform
from torchio.transforms import Motion, RandomMotion
from torchio.transforms.interpolation import Interpolation

# import multiprocessing

__author__ = "Soumick Chatterjee, Alessandro Sciarra"
__copyright__ = "Copyright 2020, Faculty of Computer Science, Otto von Guericke University Magdeburg, Germany"
__credits__ = ["Soumick Chatterjee", "Alessandro Sciarra"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Soumick Chatterjee"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Production"

class CustomMotion(Motion):   
    def __init__(self, noise_dir=2, **kargs):
        super(CustomMotion, self).__init__(**kargs)
        self.noise_dir = noise_dir

    def add_artifact(
            self,
            image: sitk.Image,
            transforms: List[sitk.Euler3DTransform],
            times: np.ndarray,
            interpolation: Interpolation,
            ):
        images = self.resample_images(image, transforms, interpolation)
        arrays = [sitk.GetArrayViewFromImage(im) for im in images]
        arrays = [array.transpose() for array in arrays]  # ITK to NumPy
        spectra = [self.fourier_transform(array) for array in arrays]
        self.sort_spectra(spectra, times)
        result_spectrum = np.empty_like(spectra[0])
        noise_dir = self.noise_dir 
        if noise_dir == -1:
            noise_dir = random.randint(0,1)
        last_index = result_spectrum.shape[noise_dir] #it can be 0, 1 or 2
        indices = (last_index * times).astype(int).tolist()
        indices.append(last_index)
        ini = 0
        for spectrum, fin in zip(spectra, indices):
            if noise_dir == 0:
                result_spectrum[..., ini:fin,:,:] = spectrum[..., ini:fin,:,:] #depending upon last_index value, move ini:fin left or right [at the end :,: for 0, : for 1, none for 2]
            elif noise_dir == 1:
                result_spectrum[..., ini:fin,:] = spectrum[..., ini:fin,:] 
            else: #original
                result_spectrum[..., ini:fin] = spectrum[..., ini:fin] 
            ini = fin
        result_image = np.real(self.inv_fourier_transform(result_spectrum))
        return result_image.astype(np.float32)

class CustomRandomMotion(RandomMotion):
    def __init__(self, noise_dir=2, **kwargs):
        super(CustomRandomMotion, self).__init__(**kwargs)
        self.noise_dir = noise_dir

    def apply_transform(self, subject):
        arguments = defaultdict(dict)
        for name, image in self.get_images_dict(subject).items():
            params = self.get_params(
                self.degrees_range,
                self.translation_range,
                self.num_transforms,
                is_2d=image.is_2d(),
            )
            times_params, degrees_params, translation_params = params
            arguments['times'][name] = times_params
            arguments['degrees'][name] = degrees_params
            arguments['translation'][name] = translation_params
            arguments['image_interpolation'][name] = self.image_interpolation
        transform = CustomMotion(noise_dir=self.noise_dir,**self.add_include_exclude(arguments))
        transformed = transform(subject)
        return transformed

class RealityMotion():
    def __init__(self, n_threads = 4, mu = 0, sigma = 0.1, random_sigma=True):
        self.n_threads = n_threads
        self.mu = mu
        self.sigma = sigma
        self.sigma_limit = sigma
        self.random_sigma = random_sigma

    def __perform_singlePE(self, idx):
        rot_x = np.random.normal(self.mu, self.sigma, 1)*random.randint(-1,1)
        rot_y = np.random.normal(self.mu, self.sigma, 1)*random.randint(-1,1)
        rot_z = np.random.normal(self.mu, self.sigma, 1)*random.randint(-1,1)
        tran_x = int(np.random.normal(self.mu, self.sigma, 1)*random.randint(-1,1))
        tran_y = int(np.random.normal(self.mu, self.sigma, 1)*random.randint(-1,1))
        tran_z = int(np.random.normal(self.mu, self.sigma, 1)*random.randint(-1,1))

        temp_vol = self.__rot_tran_3d(self.in_vol, rot_x, rot_y, rot_z, tran_x, tran_y, tran_z)
        temp_k = np.fft.fftn(temp_vol)
        for slc in range(self.in_vol.shape[2]):
            self.out_k[idx,:,slc]=temp_k[idx,:,slc] 

    def __call__(self, vol):
        if self.random_sigma:
            self.sigma = random.uniform(0, self.sigma_limit)
        shape = vol.shape
        device = vol.device
        self.in_vol = vol.squeeze().cpu().numpy()
        self.in_vol = self.in_vol/self.in_vol.max()
        self.out_k = np.zeros((self.in_vol.shape)) + 0j
        if self.n_threads > 0:
            pool = multiprocessing.Pool(self.n_threads)
            pool.map(self.__perform_singlePE, range(self.in_vol.shape[0]))
        else:
            for idx in range(self.in_vol.shape[0]):
                self.__perform_singlePE(idx)
        vol = np.abs(np.fft.ifftn(self.out_k)) 
        vol = torch.from_numpy(vol).view(shape).to(device)
        del self.in_vol, self.out_k
        return vol

    def __x_rotmat(self, theta):
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        return np.array([[1, 0, 0],
                        [0, cos_t, -sin_t],
                        [0, sin_t, cos_t]])

    def __y_rotmat(self, theta):
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        return np.array([[cos_t, 0, sin_t],
                        [0, 1, 0],
                        [-sin_t, 0, cos_t]])

    def __z_rotmat(self, theta):
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        return np.array([[cos_t, -sin_t, 0],
                        [sin_t, cos_t, 0],
                        [0, 0, 1]])

    def __rot_tran_3d(self, J, rot_x, rot_y, rot_z, tran_x, tran_y, tran_z):  
        M = self.__x_rotmat(rot_x) * self.__y_rotmat(rot_y) * self.__z_rotmat(rot_z) 
        translation = ([tran_x, tran_y, tran_z])
        K = affine_transform(J, M, translation, order=1)
        return K/(K.max()+1e-16)

class MotionCorrupter():
    def __init__(self, mode=0, degrees=10, translation=10, num_transforms=2, image_interpolation='linear', norm_mode=0, noise_dir=2, mu=0, sigma=0.1, random_sigma=False, n_threads=4):
        self.mode = mode #0: TorchIO's version, 1: Custom direction specific motion
        self.degrees = degrees
        self.translation = translation
        self.num_transforms = num_transforms
        self.image_interpolation = image_interpolation
        self.norm_mode = norm_mode #0: No Norm, 1: Divide by Max, 2: MinMax
        self.noise_dir = noise_dir #0, 1 or 2 - which direction the motion is generated, only for custom random
        self.mu = mu #Only for Reality Motion
        self.sigma = sigma #Only for Reality Motion
        self.random_sigma = random_sigma  #Only for Reality Motion - to randomise the sigma value, treating the provided sigma as upper limit and 0 as lower
        self.n_threads = n_threads #Only for Reality Motion - to apply motion for each thread encoding line parallel, max thread controlled by this. Set to 0 to perform serially.

        if mode==0: #TorchIO's version
            self.corrupter = tio.transforms.RandomMotion(degrees=degrees, translation=translation, num_transforms=num_transforms, image_interpolation=image_interpolation)
        elif mode==1: #Custom Motion
            self.corrupter = CustomRandomMotion(degrees=degrees, translation=translation, num_transforms=num_transforms, image_interpolation=image_interpolation, noise_dir=noise_dir)
        elif mode==2: #Reality motion. 
            self.corrupter = RealityMotion(n_threads=n_threads, mu=mu, sigma=sigma, random_sigma=random_sigma)

    def perform(self, vol):
        vol = vol.float()
        transformed = self.corrupter(vol)
        if self.norm_mode==1:
            vol = vol/vol.max()
            transformed = transformed/transformed.max()
        elif self.norm_mode==2:
            vol = (vol-vol.min())/(vol.max()-vol.min())
            transformed = (transformed-transformed.min())/(transformed.max()-transformed.min())
        return torch.cat([vol,transformed], 0)
