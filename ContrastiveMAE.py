from .MAEPretrain_SceneClassification.models_mae_vitae import mae_vitae_base_patch16_dec512d8b, MaskedAutoencoderViTAE
import torch 
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, datasets
import torch.nn.functional as F
from einops import repeat, rearrange
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
from vit_pytorch.vit import Transformer
import wandb
from contextlib import redirect_stdout, redirect_stderr
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from PIL import Image
import json
import pandas as pd
from torchmetrics import Accuracy
from datetime import datetime
import copy
import os
from functools import partial


class MaeBirds(LightningModule):
    def __init__(self, train_dataset, val_dataset, **kwargs):
        super().__init__()
        self.sat_encoder = mae_vitae_base_patch16_dec512d8b()
        self.sat_encoder.load_state_dict(torch.load('/storage1/fs1/jacobsn/Active/user_s.sastry/Remote-Sensing-RVSA/vitae-b-checkpoint-1599-transform-no-average.pth')['model'])
        self.sat_encoder.requires_grad_(False)
        self.ground_encoder = MaskedAutoencoderViTAE(img_size=384, patch_size=32, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_pix_loss=False, kernel=3, mlp_hidden_dim=None)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = kwargs.get('batch_size', 77)
        self.num_workers = kwargs.get('num_workers', 16)
        self.lr = kwargs.get('lr', 0.02)

    def forward(self, img_ground, img_overhead):
        ground_embeddings, *_ = self.ground_encoder.forward_encoder(img_ground, 0)
        sat_embeddings, *_ = self.sat_encoder.forward_encoder(img_overhead, 0)
        norm_ground_features = F.normalize(ground_embeddings[:, 0], dim=-1)
        norm_overhead_features = F.normalize(sat_embeddings[:, 0], dim=-1)
        similarity = torch.einsum('ij,kj->ik', norm_ground_features, norm_overhead_features)
        labels_clip = torch.arange(similarity.shape[0], device=self.device).long()
        loss_clip = (F.cross_entropy(similarity, labels_clip) + F.cross_entropy(similarity.T, labels_clip)) / 2
        loss_recon, *_ = self.ground_encoder(img_ground)
        return loss_clip, loss_recon

    def shared_step(self, batch, batch_idx):
        img_ground, img_overhead = batch[0], batch[1]
        #import code; code.interact(local=locals());
        loss_clip, loss_recon  = self(img_ground, img_overhead)
        loss = 0.3*loss_clip + loss_recon
        return loss, loss_clip, loss_recon

    def training_step(self, batch, batch_idx):
        loss, loss_clip, loss_recon = self.shared_step(batch, batch_idx)
        self.log('train_loss_recon', loss_recon, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('train_loss_clip', loss_clip, prog_bar=True, on_epoch=True, sync_dist=True)
        return {"loss": loss, "loss_clip": loss_clip, "loss_recon":loss_recon}

    def validation_step(self, batch, batch_idx):
        loss, loss_clip, loss_recon = self.shared_step(batch, batch_idx)
        self.log('val_loss_recon', loss_recon, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('val_loss_clip', loss_clip, prog_bar=True, on_epoch=True, sync_dist=True)
        return {"loss": loss, "loss_clip": loss_clip, "loss_recon":loss_recon}

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                        shuffle=True,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        persistent_workers=True,
                        pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                        shuffle=False,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        persistent_workers=True,
                        pin_memory=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = CosineAnnealingWarmRestarts(optimizer, 40)
        return optimizer

class Birds(Dataset):
    def __init__(self, dataset, label, val=False):
        self.dataset = dataset
        self.images = np.array(self.dataset['images'])
        self.idx = np.array(label.iloc[:, 1]).astype(int)
        self.images = self.images[self.idx]
        self.val = val
        if not val:
            self.transform_ground = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.TrivialAugmentWide(),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.transform_overhead = transforms.Compose([
                transforms.RandomCrop(224),
                transforms.ColorJitter(),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform_ground = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.transform_overhead = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]['file_name']
        img_ground = Image.open(os.path.join('/storage1/fs1/jacobsn/Active/user_s.sastry/metaformer/', img_path))
        img_ground = self.transform_ground(img_ground)
        if not self.val:
            img_overhead = Image.open(f"/scratch1/fs1/jacobsn/s.sastry/metaformer/train_overhead/images_sentinel/{idx}.jpeg")
        else:
            img_overhead = Image.open(f"/scratch1/fs1/jacobsn/s.sastry/metaformer/val_overhead/images_sentinel/{idx}.jpeg")
        img_overhead = self.transform_overhead(img_overhead)
        return img_ground, img_overhead

if __name__=='__main__':
    torch.cuda.empty_cache()
    logger = WandbLogger(project="Cross-View-MAE", name="Pretrained Cont MAE")
    train_dataset = json.load(open("/storage1/fs1/jacobsn/Active/user_s.sastry/metaformer/train_birds.json"))
    train_labels = pd.read_csv('/storage1/fs1/jacobsn/Active/user_s.sastry/metaformer/train_birds_labels.csv')
    train_dataset = Birds(train_dataset, train_labels)
    val_dataset = json.load(open("/storage1/fs1/jacobsn/Active/user_s.sastry/metaformer/val_birds.json"))
    val_labels = pd.read_csv('/storage1/fs1/jacobsn/Active/user_s.sastry/metaformer/val_birds_labels.csv')
    val_dataset = Birds(val_dataset, val_labels, val=True)

    checkpoint = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='ContrastiveMAEv5-{epoch:02d}-{val_loss:.2f}',
        mode='min'
    )

    model = MaeBirds(train_dataset, val_dataset)
    #model = model.load_from_checkpoint('/storage1/fs1/jacobsn/Active/user_s.sastry/metaformer/checkpoints/ContrastiveMAE-epoch=39-val_loss=1.22.ckpt', train_dataset=train_dataset, val_dataset=val_dataset)
    
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=2,
        strategy='ddp_find_unused_parameters_true',
        max_epochs=1500,
        num_nodes=1,
        callbacks=[checkpoint],
        logger=logger)
    trainer.fit(model)

