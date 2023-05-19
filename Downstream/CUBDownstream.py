from .MAEPretrain_SceneClassification.models_mae_vitae import mae_vitae_base_patch16_dec512d8b, MaskedAutoencoderViTAE
import torch 
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
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
from timm.data import Mixup
from timm.data import create_transform
from timm.loss import SoftTargetCrossEntropy
from timm.utils import accuracy

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
        self.geo_encode = nn.Linear(4, 768)
        self.date_encode = nn.Linear(4, 768)

    def forward(self, img_ground, val=False):
        if not val:
            ground_embeddings, *_ = self.ground_encoder.forward_encoder(img_ground, 0.3055)
            return F.normalize(ground_embeddings[:, 0], dim=-1)
        else:
            ground_embeddings, *_ = self.ground_encoder.forward_encoder(img_ground, 0)
            return F.normalize(ground_embeddings[:, 0], dim=-1)

class MaeBirdsDownstream(LightningModule):
    def __init__(self, train_dataset, val_dataset, **kwargs):
        super().__init__()
        self.model = MaeBirds.load_from_checkpoint('/storage1/fs1/jacobsn/Active/user_s.sastry/Remote-Sensing-RVSA/ContrastiveGeoDateMAEv5-epoch=28-val_loss=1.53.ckpt', train_dataset=train_dataset, val_dataset=val_dataset)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = kwargs.get('batch_size', 32)
        self.num_workers = kwargs.get('num_workers', 16)
        self.lr = kwargs.get('lr', 0.02)
        self.classify = nn.Linear(768, 1486)
        #self.criterion = SoftTargetCrossEntropy()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.acc = Accuracy(task='multiclass', num_classes=1486)
        self.mixup_fn = Mixup(
            mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
            prob=1.0, switch_prob=0.5, mode='batch',
            label_smoothing=0.1, num_classes=1486)

    def forward(self, img_ground, val):
        return self.model(img_ground, val)

class CUBDownstream(LightningModule):
    def __init__(self, train_dataset, val_dataset, **kwargs):
        super().__init__()
        self.model = MaeBirdsDownstream.load_from_checkpoint('/storage1/fs1/jacobsn/Active/user_s.sastry/checkpoints/ContrastiveDownstreamGeoMAEv10-epoch=05-val_loss=1.71.ckpt', train_dataset=train_dataset, val_dataset=val_dataset)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = kwargs.get('batch_size', 32)
        self.num_workers = kwargs.get('num_workers', 16)
        self.lr = kwargs.get('lr', 0.02)
        self.classify = nn.Linear(768, 200)
        self.criterion = nn.CrossEntropyLoss()
        self.acc = Accuracy(task='multiclass', num_classes=200)

    def forward(self, img_ground, val):
        return self.classify(self.model(img_ground, val))

    def shared_step(self, batch, batch_idx, val=False):
        img_ground, labels = batch[0], batch[1]
        preds = self(img_ground, val)
        loss = self.criterion(preds, labels)
        acc = self.acc(preds, labels)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, batch_idx)
        self.log('train_acc', acc, on_epoch=True, prog_bar=True)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return {"loss": loss, "acc": acc}

    def validation_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, batch_idx, True)
        self.log('val_acc', acc, prog_bar=True, on_epoch=True)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        return {"loss": loss, "acc":acc}
    
    def predict_step(self, batch, batch_idx):
        acc = self.shared_step(batch, batch_idx)
        return acc

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                        shuffle=True,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        persistent_workers=False,
                        pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                        shuffle=False,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        persistent_workers=True,
                        pin_memory=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-4, weight_decay=0.02)
        scheduler = CosineAnnealingWarmRestarts(optimizer, 5)
        return [optimizer], [scheduler]

class CUBBirds(Dataset):
    def __init__(self, path, val=False):
        self.path = path
        self.images = np.loadtxt(os.path.join(self.path, "train_test_split.txt"))
        if not val:
            self.images = self.images[self.images[:, 1] == 1]
        else:
            self.images = self.images[self.images[:, 1] == 0]
        self.img_paths = np.genfromtxt(os.path.join(self.path, 'images.txt'),dtype='str')
        if not val:
            self.transform = transforms.Compose([
                transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandAugment(12, 12, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.TrivialAugmentWide(num_magnitude_bins=50, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.AugMix(9, 9, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.path, 'images/'+ self.img_paths[int(self.images[idx, 0])-1, 1])
        label = int(self.img_paths[int(self.images[idx, 0])-1, 1][:3]) - 1
        img = Image.open(img_path)
        if len(np.array(img).shape)==2:
            img_path = os.path.join(self.path, 'images/'+ self.img_paths[int(self.images[idx-1, 0])-1, 1])
            label = int(self.img_paths[int(self.images[idx-1, 0])-1, 1][:3]) - 1
            img = Image.open(img_path)
            #img = Image.fromarray(np.stack(np.array(img), np.array(img), np.array(img)), axis=-1)
        img = self.transform(img)
        return img, torch.tensor(label)

if __name__=='__main__':
    f = open("log.txt", "w")
    #with redirect_stdout(f), redirect_stderr(f):
    if True:
        torch.cuda.empty_cache()
        logger = WandbLogger(project="Fine Grained", name="CUB")
        path = '/scratch1/fs1/jacobsn/s.sastry/CUB_200_2011'
        train_dataset = CUBBirds(path)
        val_dataset = CUBBirds(path, val=True)

        checkpoint = ModelCheckpoint(
            monitor='val_loss',
            dirpath='checkpoints',
            filename='CUBv1-{epoch:02d}-{val_loss:.2f}',
            mode='min'
        )

    
        model = CUBDownstream(train_dataset, val_dataset)
        #model = model.load_from_checkpoint("/storage1/fs1/jacobsn/Active/user_s.sastry/checkpoints/ContrastiveDownstreamGeoMAEv7-epoch=94-val_loss=2.77.ckpt", train_dataset=train_dataset, val_dataset=val_dataset)
        trainer = pl.Trainer(
            accelerator='gpu',
            devices=2,
            strategy='ddp_find_unused_parameters_true',
            max_epochs=1500,
            num_nodes=1,
            callbacks=[checkpoint],
            logger=logger
            )
        trainer.fit(model)
        """predloader = DataLoader(train_dataset,
                        shuffle=False,
                        batch_size=64,
                        num_workers=8,
                        persistent_workers=False,
                        pin_memory=True,
                        drop_last=True)
        acc = trainer.predict(model, predloader)
        print(sum(acc)/len(acc))"""
