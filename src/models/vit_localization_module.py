import torch
import pickle


from torch import nn
from torch.nn import MSELoss
from torch.nn import functional as F
from lightning import LightningModule
from transformers import ViTConfig, ViTModel
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.utils.loss_utils import KDE_loss3D
from src.utils.evaluation_utils import calc_jaccard_index
from src.models.decoder import CNNDecoder


class ViTLocalizationLightningModel(LightningModule):
    def __init__(self, setup_params_path, lr=1e-4, T_max=50, pretrained=False, device='cuda:0'):
        super(ViTLocalizationLightningModel, self).__init__()
        with open(setup_params_path, 'rb') as f:
            self.setup_params = pickle.load(f)
        self.lr = lr
        self.T_max = T_max
        self.D = self.setup_params['D']
        self.H = self.setup_params['HH']
        self.W = self.setup_params['WW']
        self.blob_maxv = self.setup_params['blob_maxv']
        self._device = device if torch.cuda.is_available() else 'cpu'

        # Load ViT Tiny from Hugging Face
        if pretrained:
            self.encoder = ViTModel.from_pretrained(
                'google/vit-base-patch16-224-in21k')
        else:
            # ViT base config for grayscale images
            config = ViTConfig(
                image_size=224,  # 224 for RGB
                patch_size=16,  # 16 for RGB
                num_channels=1,  # 3 for RGB
                hidden_size=256,  # 16*16*3=768 for RGB
                num_hidden_layers=4,  # 12 for RGB
                num_attention_heads=4,  # 12 for RGB
                intermediate_size=3072)
            self.encoder = ViTModel(config)
        self.decoder = CNNDecoder(input_patch_size=16, embedding_dim=256,
                                  output_depth=self.D, output_height=self.H, output_width=self.W)
        
        self.loss_fn = MSELoss() # Loss function for Gaussian centers data representation

    def forward(self, x):
        vit_input = nn.functional.interpolate(
            x, size=(224, 224), mode='bilinear', align_corners=False)
        encoder_features = self.encoder(vit_input)
        # Shape: (batch_size, num_patches + 1, hidden_size)
        last_hidden_state = encoder_features.last_hidden_state
        # Remove the class token
        patch_embeddings = last_hidden_state[:, 1:]
        out = self.decoder(patch_embeddings)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.loss_fn(y_hat, y)
        self.log('val_loss', val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.T_max)
        return [optimizer], [scheduler]


# Checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='checkpoints/',
    filename='vit-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    mode='min',
)
