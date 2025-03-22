import os
import torch
import numpy as np
from skimage import io
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


class GaussianCentersDataset(Dataset):
    # initialization of the dataset
    def __init__(self, root_dir, list_IDs, labels):
        self.root_dir = root_dir
        self.list_IDs = list_IDs
        self.labels = labels

        self.r = labels['blob_r']
        self.maxv = labels['blob_maxv']
        self.volume_size = labels['volume_size']

    # total number of samples in the dataset
    def __len__(self):
        return len(self.list_IDs)

    # sampling one example from the data
    def __getitem__(self, index):
        # select sample
        ID = self.list_IDs[index]
        # load tiff image
        tiff_file = os.path.join(self.root_dir, ID)
        image = imread(tiff_file)

        image = image[np.newaxis, :, :].astype(np.float32)

        y = np.zeros(self.volume_size)
        y = np.pad(y, self.r)
        xyz_ids, blob3d = self.labels[ID][0], self.labels[ID][1]
        for i in range(xyz_ids.shape[0]):
            xidx, yidx, zidx = xyz_ids[i, 0], xyz_ids[i, 1], xyz_ids[i, 2]
            y[zidx:zidx + 2 * self.r + 1, yidx:yidx + 2 * self.r + 1, xidx:xidx + 2 * self.r + 1] += blob3d[i]
        y = (y[self.r:-self.r, self.r:-self.r, self.r:-self.r]).astype(np.float32)

        return image, y
