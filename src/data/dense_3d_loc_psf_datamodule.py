import os
import pickle
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.data.datasets import TiffBooleanGridDataset, GaussianCentersDataset

class Dense3DLocPSFDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        setup_params_path: str,
        labels_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = False
    ):
        """
        PyTorch Lightning DataModule that organizes your ImagesDataset for 
        training, validation, and testing.
        """
        super().__init__()
        self.data_dir = data_dir
        with open(setup_params_path, 'rb') as f:
            self.setup_params = pickle.load(f)
        with open(labels_path, 'rb') as f:
            self.labels = pickle.load(f)
        
        # Dataloader hyperparameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Placeholders for the actual Dataset objects
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str = None):
        """
        Called by Lightning at the beginning of fit/test. Used to set up
        Dataset objects.
        """
        # If we're in the 'fit' stage or setting up for both
        if stage == "fit" or stage is None:
            x_list = os.listdir(self.data_dir)
            num_x = len(x_list)
            partition = {'train': x_list[:int(num_x*0.9)], 'validation': x_list[int(num_x*0.9):]}
            self.train_dataset = GaussianCentersDataset(
                root_dir=self.data_dir,
                list_IDs=partition['train'],
                labels=self.labels
            )
            self.val_dataset = GaussianCentersDataset(
                root_dir=self.data_dir,
                list_IDs=partition['validation'],
                labels=self.labels
            )

        # If we're in the 'test' stage or setting up for both
        # if stage == "test" or stage is None:
        #     self.test_dataset = ImagesDataset(
        #         root_dir=self.root_dir,
        #         list_IDs=self.test_list_IDs,
        #         labels=self.test_labels,
        #         setup_params=self.setup_params
        #     )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,         
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,       
            num_workers=self.num_workers,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,        
            num_workers=self.num_workers
        )
