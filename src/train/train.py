from typing import Any, Dict, List, Optional, Tuple

import time
import torch
import hydra
import logging
import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningDataModule, LightningModule, Callback
from lightning.pytorch.loggers import Logger
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger

from src.data.dense_3d_loc_psf_datamodule import Dense3DLocPSFDataModule
from src.models.vit_localization_module import ViTLocalizationLightningModel
from src.utils.pylogger import RankedLogger
from src.utils.instantiators import instantiate_callbacks, instantiate_loggers

log = RankedLogger(__name__, rank_zero_only=True)


def train(cfg: DictConfig):
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    #log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule = Dense3DLocPSFDataModule(**cfg.data)
    #datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    #log.info(f"Instantiating model <{cfg.model._target_}>")
    #model: LightningModule = hydra.utils.instantiate(cfg.model)
    model = ViTLocalizationLightningModel(**cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger)

    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule,
                ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics
    metric_dict = {**train_metrics}

    return metric_dict


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # train the model
    train(cfg)


if __name__ == "__main__":
    main()
