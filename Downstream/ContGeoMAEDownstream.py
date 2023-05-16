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

    def forward(self, img_ground, geoloc, date):
        geo_token = self.geo_encode(geoloc)
        date_token = self.date_encode(date)
        ground_embeddings, *_ = self.ground_encoder.forward_encoder(img_ground, 0)
        return F.normalize(ground_embeddings[:, 0] + geo_token + date_token, dim=-1)

class MaeBirdsDownstream(LightningModule):
    def __init__(self, train_dataset, val_dataset, **kwargs):
        super().__init__()
        self.model = MaeBirds.load_from_checkpoint('/storage1/fs1/jacobsn/Active/user_s.sastry/Remote-Sensing-RVSA/ContrastiveGeoDateMAEv5-epoch=28-val_loss=1.53.ckpt', train_dataset=train_dataset, val_dataset=val_dataset)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = kwargs.get('batch_size', 64)
        self.num_workers = kwargs.get('num_workers', 8)
        self.lr = kwargs.get('lr', 0.02)
        self.classify = nn.Linear(768, 1486)
        self.criterion = SoftTargetCrossEntropy()
        #self.acc = Accuracy(task='multiclass', num_classes=1486)
        self.mixup_fn = Mixup(
            mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
            prob=1.0, switch_prob=0.5, mode='batch',
            label_smoothing=0.1, num_classes=1486)

    def forward(self, img_ground, geoloc, date):
        return self.classify(self.model(img_ground, geoloc, date))

    def shared_step(self, batch, batch_idx):
        img_ground, geoloc, date, labels = batch[0], batch[1], batch[2], batch[3]
        img_ground, labels_mix = self.mixup_fn(img_ground, labels)
        #import code; code.interact(local=locals());
        preds = self(img_ground, geoloc, date)
        #import code; code.interact(local=locals());
        loss = self.criterion(preds, labels_mix)
        #acc = self.acc(preds, labels)
        acc = sum(accuracy(preds, labels)) / preds.shape[0]
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, batch_idx)
        self.log('train_acc', acc, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return {"loss": loss, "acc": acc}

    def validation_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, batch_idx)
        self.log('val_acc', acc, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return {"loss": loss, "acc":acc}

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
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.001)
        scheduler = CosineAnnealingWarmRestarts(optimizer, 40)
        return [optimizer], [scheduler]

class Birds(Dataset):
    def __init__(self, dataset, label, val=False):
        self.dataset = dataset
        self.images = np.array(self.dataset['images'])
        self.labels = np.array(self.dataset['categories'])
        self.species = {}
        for i in range(len(self.labels)):
            self.species[self.labels[i]['id']] = i
        self.categories = np.array(self.dataset['annotations'])
        self.idx = np.array(label.iloc[:, 1]).astype(int)
        self.images = self.images[self.idx]
        self.categories = self.categories[self.idx]
        self.val = val
        if not val:
            self.transform_ground = create_transform(
            input_size=384,
            is_training=True,
            color_jitter=0.4,
            auto_augment='rand-m9-mstd0.5-inc1',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
            interpolation='bicubic',
        )
            # self.transform_ground = transforms.Compose([
            #     transforms.Resize((384, 384)),
            #     transforms.AutoAugment(),
            #     transforms.AugMix(5, 5),
            #     transforms.RandomHorizontalFlip(0.5),
            #     transforms.RandomVerticalFlip(0.5),
            #     transforms.ToTensor(),
            #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # ])
        else:
            self.transform_ground = transforms.Compose([
                transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]['file_name']
        label = self.species[self.categories[idx]['category_id']]
        img_ground = Image.open(os.path.join('/storage1/fs1/jacobsn/Active/user_s.sastry/metaformer/', img_path))
        img_ground = self.transform_ground(img_ground)
        lat = self.images[idx]['latitude']
        lon = self.images[idx]['longitude']
        date = self.images[idx]['date'].split(" ")[0]
        month = int(datetime.strptime(date, '%Y-%m-%d').date().strftime('%m'))
        day = int(datetime.strptime(date, '%Y-%m-%d').date().strftime('%d'))
        date_encode = torch.tensor([np.sin(2*np.pi*month/12), np.cos(2*np.pi*month/12), np.sin(2*np.pi*day/31), np.cos(2*np.pi*day/31)])
        return img_ground, torch.tensor([np.sin(np.pi*lat/90), np.cos(np.pi*lat/90), np.sin(np.pi*lon/180), np.cos(np.pi*lon/180)]).float(), date_encode.float(), torch.tensor(label)

if __name__=='__main__':
    f = open("log.txt", "w")
    #with redirect_stdout(f), redirect_stderr(f):
    if True:
        torch.cuda.empty_cache()
        logger = WandbLogger(project="Cross-View-MAE", name="Downstram Cont MAE")
        train_dataset = json.load(open("/storage1/fs1/jacobsn/Active/user_s.sastry/metaformer/train_birds.json"))
        train_labels = pd.read_csv('/storage1/fs1/jacobsn/Active/user_s.sastry/metaformer/train_birds_labels.csv')
        train_dataset = Birds(train_dataset, train_labels)
        val_dataset = json.load(open("/storage1/fs1/jacobsn/Active/user_s.sastry/metaformer/val_birds.json"))
        val_labels = pd.read_csv('/storage1/fs1/jacobsn/Active/user_s.sastry/metaformer/val_birds_labels.csv')
        val_dataset = Birds(val_dataset, val_labels, val=True)

        checkpoint = ModelCheckpoint(
            monitor='val_loss',
            dirpath='checkpoints',
            filename='ContrastiveDownstreamGeoMAEv7-{epoch:02d}-{val_loss:.2f}',
            mode='min'
        )

    
        model = MaeBirdsDownstream(train_dataset, val_dataset)
        #model = model.load_from_checkpoint("/storage1/fs1/jacobsn/Active/user_s.sastry/checkpoints/ContrastiveDownstreamGeoMAEv7-epoch=94-val_loss=2.77.ckpt", train_dataset=train_dataset, val_dataset=val_dataset)
        trainer = pl.Trainer(
            accelerator='gpu',
            devices=4,
            strategy='ddp_find_unused_parameters_true',
            max_epochs=1500,
            num_nodes=1,
            callbacks=[checkpoint],
            logger=logger
            )
        trainer.fit(model)
