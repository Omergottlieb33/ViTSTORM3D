import numpy as np
from src.data.generator.gaussian_center_model import ImModel, Sampling
import torch
import os
import pickle
import matplotlib
import shutil
from skimage import io
import scipy.io as sio
import time
import hydra
from omegaconf import DictConfig

import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

np.random.seed(66)
torch.manual_seed(88)

# Set the environment variable for full Hydra error messages
os.environ['HYDRA_FULL_ERROR'] = '1'

@hydra.main(config_path="../../../configs", config_name="gaussian_centers_generator_config.yaml")
def main(cfg: DictConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')

    im_param_dict = dict(
        device=device,
        M=cfg.imaging.M,
        NA=cfg.imaging.NA,
        n_immersion=cfg.imaging.n_immersion,
        lamda=cfg.optics.lamda,
        n_sample=cfg.optics.n_sample,
        f_4f=cfg.optics.f_4f,
        ps_camera=cfg.optics.ps_camera,
        ps_BFP=cfg.optics.ps_BFP,
        NFP=cfg.optics.NFP,
        phase_mask=None,
        Hmask=cfg.phase_mask.Hmask,
        Wmask=cfg.phase_mask.Wmask,
        H=cfg.image.H,
        W=cfg.image.W,
        g_size=cfg.image.g_size,
        g_sigma=cfg.image.g_sigma,
        bg=cfg.image.bg,
        baseline=cfg.image.baseline,
        read_std=cfg.image.read_std,
    )

    mask_dict = sio.loadmat(cfg.mask_path)
    mask_name = list(mask_dict.keys())[3]
    phase_mask = mask_dict[mask_name]
    im_param_dict['phase_mask'] = phase_mask

    sampling_param_dict = dict(
        zrange=cfg.sampling.zrange,
        D=cfg.sampling.D,
        psf_half_size=cfg.sampling.psf_half_size,
        Nsig_range=cfg.sampling.Nsig_range,
        num_particles_range=cfg.sampling.num_particles_range,
        blob_r=cfg.sampling.blob_r,
        blob_sigma=cfg.sampling.blob_sigma,
        blob_maxv=cfg.sampling.blob_maxv,
    )
    us_factor = cfg.sampling.us_factor
    sampling_param_dict['HH'] = int(im_param_dict['H'] * us_factor)
    sampling_param_dict['WW'] = int(im_param_dict['W'] * us_factor)
    sampling_param_dict['buffer_HH'] = int(sampling_param_dict['psf_half_size'] * us_factor)
    sampling_param_dict['buffer_WW'] = int(sampling_param_dict['psf_half_size'] * us_factor)
    sampling_param_dict['us_factor'] = us_factor

    vs_xy = im_param_dict['ps_camera'] / im_param_dict['M'] / us_factor
    vs_z = ((sampling_param_dict['zrange'][1] - sampling_param_dict['zrange'][0]) / sampling_param_dict['D'])
    sampling_param_dict['vs_xy'] = vs_xy
    sampling_param_dict['vs_z'] = vs_z

    print(f"vs_xy: {vs_xy} um, vs_z: {vs_z} um")

    param_dict = {**im_param_dict, **sampling_param_dict}
    param_dict['td_folder'] = cfg.training_data_folder
    os.makedirs(param_dict['td_folder'], exist_ok=True)
    param_dict['n_ims'] = cfg.n_ims
    param_dict['project_01'] = cfg.project_01

    device = param_dict['device']

    # imaging model
    model = ImModel(param_dict)
    psf_path = os.path.join(param_dict['td_folder'], 'psfs.jpg')
    model.model_demo(np.linspace(param_dict['zrange'][0], param_dict['zrange'][1], 5), path=psf_path)  # check PSF
    # sampling model
    sampling = Sampling(param_dict)

    # start
    td_folder = param_dict['td_folder']
    # if os.path.exists(td_folder):  # delete the directory if it exists
    #     shutil.rmtree(td_folder)
    x_folder = td_folder+'/x'
    os.makedirs(x_folder, exist_ok=True)  # make the folder for training data

    t0 = time.time()
    # labels_dict for training
    labels_dict = {}
    labels_dict['volume_size'] = (param_dict['D'], param_dict['HH'], param_dict['WW'])
    labels_dict['us_factor'] = param_dict['us_factor']
    labels_dict['blob_r'] = sampling.blob_r  # radius of each 3D blob representing an emitter in space
    labels_dict['blob_maxv'] = sampling.blob_maxv  # maximum value of blobs

    ntrain = param_dict['n_ims']
    for i in range(ntrain):
        xyzps, xyz_ids, blob3d = sampling.xyzp_batch()
        im = model(torch.from_numpy(xyzps).to(device)).cpu().numpy()
        if param_dict['project_01']:
            im = ((im-im.min())/(im.max()-im.min()))

        x_name = str(i).zfill(5) + '.tif'
        io.imsave(os.path.join(x_folder, x_name), im, check_contrast=False)
        labels_dict[x_name] = (xyz_ids, blob3d)

        if i % (ntrain//10) == 0:
            print('Training Example [%d / %d]' % (i + 1, ntrain))
    print('Training Example [%d / %d]' % (ntrain, ntrain))

    y_file = os.path.join(td_folder, r'y.pickle')
    with open(y_file, 'wb') as handle:
        pickle.dump(labels_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    param_file = os.path.join(td_folder, r'param.pickle')
    with open(param_file, 'wb') as handle:
        pickle.dump(param_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    t1 = time.time()
    print(f'finished generating training data in {t1-t0}s.')

if __name__ == "__main__":
    main()