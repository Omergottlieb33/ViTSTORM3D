import torch
from torch.utils.data import Dataset
from skimage.io import imread
import numpy as np

# converts continuous xyz locations to a boolean grid
def batch_xyz_to_boolean_grid(xyz_np, setup_params):
    
    # calculate upsampling factor
    pixel_size_FOV, pixel_size_rec = setup_params['pixel_size_FOV'], setup_params['pixel_size_rec']
    upsampling_factor = pixel_size_FOV / pixel_size_rec
    
    # axial pixel size
    pixel_size_axial = setup_params['pixel_size_axial']

    # current dimensions
    H, W, D = setup_params['H'], setup_params['W'], setup_params['D']
    
    # shift the z axis back to 0
    zshift = xyz_np[:, :, 2] - setup_params['zmin']
    
    # number of particles
    batch_size, num_particles = zshift.shape
    
    # project xyz locations on the grid and shift xy to the upper left corner
    xg = (np.floor(xyz_np[:, :, 0]/pixel_size_rec) + np.floor(W/2)*upsampling_factor).astype('int')
    yg = (np.floor(xyz_np[:, :, 1]/pixel_size_rec) + np.floor(H/2)*upsampling_factor).astype('int')
    zg = (np.floor(zshift/pixel_size_axial)).astype('int')
    
    # indices for sparse tensor
    indX, indY, indZ = (xg.flatten('F')).tolist(), (yg.flatten('F')).tolist(), (zg.flatten('F')).tolist()

    # update dimensions
    H, W = int(H * upsampling_factor), int(W * upsampling_factor)
    
    # if sampling a batch add a sample index
    if batch_size > 1:
        indS = (np.kron(np.ones(num_particles), np.arange(0, batch_size, 1)).astype('int')).tolist()
        ibool = torch.LongTensor([indS, indZ, indY, indX])
    else:
        ibool = torch.LongTensor([indZ, indY, indX])
    
    # spikes for sparse tensor
    vals = torch.ones(batch_size*num_particles)
    
    # resulting 3D boolean tensor
    if batch_size > 1:
        boolean_grid = torch.sparse_coo_tensor(ibool, vals, torch.Size([batch_size, D, H, W]), dtype=torch.float32).to_dense()
    else:
        boolean_grid = torch.sparse_coo_tensor(ibool, vals, torch.Size([D, H, W]), dtype=torch.float32).to_dense()
    
    return boolean_grid