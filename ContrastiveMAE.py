from vit_pytorch import ViT
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

class MAE(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        decoder_dim,
        masking_ratio = 0.75,
        decoder_depth = 1,
        decoder_heads = 8,
        decoder_dim_head = 64
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        self.encoder_ground = encoder
        self.encoder_overhead = copy.deepcopy(self.encoder_ground)
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]
        self.to_patch_ground, self.patch_to_emb_ground = encoder.to_patch_embedding[:2]
        self.to_patch_overhead, self.patch_to_emb_overhead = copy.deepcopy(self.to_patch_ground), copy.deepcopy(self.patch_to_emb_ground)
        pixel_values_per_patch = self.patch_to_emb_ground.weight.shape[-1]
        #self.iim_token = nn.Parameter(torch.randn(1, encoder_dim))
        self.enc_embed = nn.Parameter(torch.randn(num_patches, encoder_dim))
        #self.match = nn.Linear(encoder_dim, 1)
        #self.geo_encode = nn.Linear(4, encoder_dim)
        self.ground_token = nn.Parameter(torch.randn(1, encoder_dim))
        self.overhead_token = nn.Parameter(torch.randn(1, encoder_dim))

        # decoder parameters
        self.decoder_dim = decoder_dim
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.dec_to_feat = nn.Linear(decoder_dim, encoder_dim)
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder_ground = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4)
        self.decoder_features = Transformer(dim = encoder_dim, depth = 4, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4)
        #self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.decoder_pos_emb_ground = nn.Parameter(torch.randn(num_patches-1, decoder_dim))
        self.to_pixels_ground = nn.Linear(decoder_dim, pixel_values_per_patch)
        self.out = nn.Sigmoid()

    def forward(self, img_ground, img_overhead):
        device = img_ground.device

        patches_ground = self.to_patch_ground(img_ground)
        patches_overhead = self.to_patch_overhead(img_overhead)
        #patches_all = torch.cat((patches_ground, patches_overhead), dim=1)
        batch, num_patches, *_ = patches_ground.shape

        # patch to encoder tokens and add positions
        tokens_ground = self.patch_to_emb_ground(patches_ground)
        tokens_overhead = self.patch_to_emb_overhead(patches_overhead)

        ground_token = repeat(self.ground_token, 'n d -> b n d', b = batch)
        tokens_ground = torch.cat((ground_token, tokens_ground), dim=1) + self.enc_embed

        overhead_token = repeat(self.overhead_token, 'n d -> b n d', b = batch)
        tokens_overhead = torch.cat((overhead_token, tokens_overhead), dim=1) + self.enc_embed

        encoded_overhead = self.encoder_overhead.transformer(tokens_overhead)
        
        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device = device).argsort(dim = -1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        # get the unmasked tokens to be encoded

        batch_range = torch.arange(batch, device = device)[:, None]
        tokens_unmasked = tokens_ground[:, 1:][batch_range, unmasked_indices]

        tokens_unmasked = torch.cat((ground_token, tokens_unmasked), dim=1)

        encoded_ground = self.encoder_ground.transformer(tokens_unmasked)

        decoder_tokens = self.enc_to_dec(encoded_ground)

        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_masked)

        decoder_tokens_all = torch.zeros(batch, num_patches, self.decoder_dim, device = device)
        decoder_tokens_all[batch_range, unmasked_indices] = decoder_tokens[:, 1:]
        decoder_tokens_all[batch_range, masked_indices] = mask_tokens

        decoded_tokens = self.decoder_ground(decoder_tokens_all + self.decoder_pos_emb_ground)

        masked_patches = patches_ground[batch_range, masked_indices]

        pred_pixel_values_ground = self.out(self.to_pixels_ground(decoded_tokens))

        masked_pixel_preds = pred_pixel_values_ground[batch_range, masked_indices]

        loss_recon = F.mse_loss(masked_pixel_preds, masked_patches)

        decoder_tokens_all = torch.cat((decoder_tokens[:, 0:1], decoder_tokens_all), dim=1)

        decoded_features = self.decoder_features(self.dec_to_feat(decoder_tokens_all) +  self.enc_embed)

        norm_ground_features = F.normalize(decoded_features[:, 0], dim=-1)
        norm_overhead_features = F.normalize(encoded_overhead[:, 0], dim=-1)
        similarity = torch.einsum('ij,kj->ik', norm_ground_features, norm_overhead_features)
        #similarity = torch.matmul(norm_ground_features, norm_overhead_features.T)

        labels_clip = torch.arange(batch, device=device).long()
        loss_clip = (F.cross_entropy(similarity, labels_clip) + F.cross_entropy(similarity.T, labels_clip)) / 2
        return loss_clip, loss_recon

class MaeBirds(LightningModule):
    def __init__(self, train_dataset, val_dataset, **kwargs):
        super().__init__()
        self.vit = ViT(
            image_size = 224,
            patch_size = 16,
            num_classes = 10,
            dim = 1024,
            depth = 12,
            heads = 12,
            mlp_dim = 2048,
            dropout=0.2
        )
        self.model = MAE(
            encoder=self.vit,
            masking_ratio=0.75,
            decoder_dim=256,
            decoder_depth=8,
        )
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = kwargs.get('batch_size', 64)
        self.num_workers = kwargs.get('num_workers', 16)
        self.lr = kwargs.get('lr', 0.02)

    def forward(self, img_ground, img_overhead):
        return self.model(img_ground, img_overhead)

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
        return [optimizer], [scheduler]

class Birds(Dataset):
    def __init__(self, dataset, label, val=False):
        self.dataset = dataset
        self.images = np.array(self.dataset['images'])
        self.idx = np.array(label.iloc[:, 1]).astype(int)
        self.images = self.images[self.idx]
        self.val = val
        if not val:
            self.transform_ground = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor()
            ])
            self.transform_overhead = transforms.Compose([
                transforms.RandomCrop(224),
                transforms.ToTensor()
            ])
        else:
            self.transform_ground = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor()
            ])
            self.transform_overhead = transforms.Compose([
                transforms.RandomCrop(224),
                transforms.ToTensor()
            ])
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]['file_name']
        img_ground = Image.open(img_path)
        img_ground = self.transform_ground(img_ground)
        if not self.val:
            img_overhead = Image.open(f"/scratch1/fs1/jacobsn/s.sastry/metaformer/train_overhead/images_sentinel/{idx}.jpeg")
        else:
            img_overhead = Image.open(f"/scratch1/fs1/jacobsn/s.sastry/metaformer/val_overhead/images_sentinel/{idx}.jpeg")
        img_overhead = self.transform_overhead(img_overhead)
        return img_ground, img_overhead

if __name__=='__main__':
    torch.cuda.empty_cache()
    logger = WandbLogger(project="Meta-MAE", name="Cross View Cont MAE")
    train_dataset = json.load(open("train_birds.json"))
    train_labels = pd.read_csv('train_birds_labels.csv')
    train_dataset = Birds(train_dataset, train_labels)
    val_dataset = json.load(open("val_birds.json"))
    val_labels = pd.read_csv('val_birds_labels.csv')
    val_dataset = Birds(val_dataset, val_labels, val=True)

    checkpoint = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='ContrastiveMAE-{epoch:02d}-{val_loss:.2f}',
        mode='min'
    )

    model = MaeBirds(train_dataset, val_dataset)
    
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=4,
        strategy='ddp_find_unused_parameters_true',
        max_epochs=1500,
        num_nodes=1,
        callbacks=[checkpoint],
        logger=logger)
    trainer.fit(model)
