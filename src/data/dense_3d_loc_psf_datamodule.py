import pickle
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.utils.data_utils import ImagesDataset

class Dense3DLocPSFDataModule(pl.LightningDataModule):
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
        self.partition = self.setup_params['partition']
        
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
            self.train_dataset = ImagesDataset(
                root_dir=self.data_dir,
                list_IDs=self.partition['train'],
                labels=self.labels,
                setup_params=self.setup_params
            )
            self.val_dataset = ImagesDataset(
                root_dir=self.data_dir,
                list_IDs=self.partition['valid'],
                labels=self.labels,
                setup_params=self.setup_params
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
            shuffle=True,         # Usually shuffle for training
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,        # Typically no shuffle for validation
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,        # Typically no shuffle for testing
            num_workers=self.num_workers
        )
