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

# PSF images with corresponding xyz labels dataset
class ImagesDataset(Dataset):
    
    # initialization of the dataset
    def __init__(self, root_dir, list_IDs, labels, setup_params):
        self.root_dir = root_dir
        self.list_IDs = list_IDs
        self.labels = labels
        self.setup_params = setup_params
        self.train_stats = setup_params['train_stats']
    
    # total number of samples in the dataset
    def __len__(self):
        return len(self.list_IDs)
    
    # sampling one example from the data
    def __getitem__(self, index):
        
        # select sample
        ID = self.list_IDs[index]
        
        # load tiff image
        im_name = self.root_dir + '/im' + ID + '.tiff'
        im_np = imread(im_name)

        # turn image into torch tensor with 1 channel
        im_np = np.expand_dims(im_np, 0)
        im_tensor = torch.from_numpy(im_np)
        
        # corresponding xyz labels turned to a boolean tensor
        xyz_np = self.labels[ID]
        bool_grid = batch_xyz_to_boolean_grid(xyz_np, self.setup_params)
        
        return im_tensor, bool_grid