import torch
import pytorch_lightning as pl
from datasets import CrossViewiNATBirds
from models import MAE, CVEMAEMeta, CVMMAEMeta
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
from pytorch_lightning.loggers import WandbLogger
import json
import pandas as pd
from config import cfg


def pretrain():
    torch.cuda.empty_cache()
    logger = WandbLogger(project="BirdSAT", name=cfg.pretrain.train.expt_name)
    train_json = json.load(open("data/train_birds.json"))
    train_labels = pd.read_csv("data/train_birds_labels.csv")
    train_dataset = CrossViewiNATBirds(train_json, train_labels)
    val_json = json.load(open("data/val_birds.json"))
    val_labels = pd.read_csv("data/val_birds_labels.csv")
    val_dataset = CrossViewiNATBirds(val_json, val_labels, val=True)

    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints",
        filename="{cfg.pretrain.train.expt_name}-{epoch:02d}-{val_loss:.2f}",
        mode="min",
    )

    if cfg.pretrain.train.model_type == "MAE":
        model = MAE(train_dataset, val_dataset)
    elif cfg.pretrain.train.model_type == "CVEMAE":
        model = CVEMAEMeta(train_dataset, val_dataset)
    elif cfg.pretrain.train.model_type == "CVMMAE":
        model = CVMMAEMeta(train_dataset, val_dataset)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=cfg.pretrain.train.devices,
        strategy="ddp_find_unused_parameters_true",
        max_epochs=cfg.pretrain.train.num_epochs,
        num_nodes=1,
        callbacks=[checkpoint],
        logger=logger,
    )
    trainer.fit(model)


if __name__ == "__main__":
    pretrain()
