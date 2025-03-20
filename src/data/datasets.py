import torch
import numpy as np
from skimage.io import imread
from torch.utils.data import Dataset

from src.utils.data_utils import batch_xyz_to_boolean_grid
# PSF images with corresponding xyz labels dataset


class TiffBooleanGridDataset(Dataset):
    """
    PyTorch Dataset class for loading tiff images and corresponding xyz labels 
    and turning them into boolean grids."""

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
