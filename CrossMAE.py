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

        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]
        self.to_patch_ground, self.patch_to_emb_ground = encoder.to_patch_embedding[:2]
        self.to_patch_overhead, self.patch_to_emb_overhead = copy.deepcopy(self.to_patch_ground), copy.deepcopy(self.patch_to_emb_ground)
        pixel_values_per_patch = self.patch_to_emb.weight.shape[-1]
        self.iim_token = nn.Parameter(torch.randn(1, encoder_dim))
        self.enc_embed = nn.Parameter(torch.randn(2*num_patches-1, encoder_dim))
        self.match = nn.Linear(encoder_dim, 1)


        # decoder parameters
        self.decoder_dim = decoder_dim
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder_ground = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4)
        self.decoder_overhead = copy.deepcopy(self.decoder_ground)
        #self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.decoder_pos_emb_ground = nn.Parameter(torch.randn(num_patches-1, decoder_dim))
        self.decoder_pos_emb_overhead = copy.deepcopy(self.decoder_pos_emb_ground)
        self.to_pixels_ground = nn.Linear(decoder_dim, pixel_values_per_patch)
        self.to_pixels_overhead = copy.deepcopy(self.to_pixels_ground)
        self.out = nn.Sigmoid()

    def forward(self, img_ground, img_overhead):
        device = img_ground.device

        patches_ground = self.to_patch_ground(img_ground)
        patches_overhead = self.to_patch_overhead(img_overhead)
        patches_all = torch.cat((patches_ground, patches_overhead), dim=1)
        batch, num_patches, *_ = patches_all.shape

        # patch to encoder tokens and add positions
        tokens_ground = self.patch_to_emb_ground(patches_ground)
        tokens_overhead = self.patch_to_emb_overhead(patches_overhead)

        tokens_reconstruct = torch.cat((tokens_ground, tokens_overhead), dim=1)
        tokens_reconstruct = tokens_reconstruct + self.enc_embed[1:]

        tokens_ground_ex = torch.cat((tokens_ground, tokens_ground), dim=0)
        tokens_overhead_ex = torch.cat((tokens_overhead, torch.roll(tokens_overhead, 1, 0)), dim=0)
        iim_token = repeat(self.iim_token, 'n d -> b n d', b = 2*batch)
        tokens_matching = torch.cat((iim_token, tokens_ground_ex, tokens_overhead_ex), dim=1)
        tokens_matching = tokens + self.enc_embed

        enc_tokens_matching = self.encoder.transformer(tokens_matching)
        matching_probs = self.match(enc_tokens_matching[:, 0, :])
        
        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device = device).argsort(dim = -1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        # get the unmasked tokens to be encoded

        batch_range = torch.arange(batch, device = device)[:, None]
        tokens_unmasked = tokens_reconstruct[batch_range, unmasked_indices]

        # get the patches to be masked for the final reconstruction loss

        masked_patches = patches_all[batch_range, masked_indices]

        # attend with vision transformer

        encoded_tokens = self.encoder.transformer(tokens_unmasked)

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder

        decoder_tokens = self.enc_to_dec(encoded_tokens)
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_masked)

        decoder_tokens_all = torch.zeros(batch, num_patches, self.decoder_dim, device = device)
        decoder_tokens_all[batch_range, unmasked_indices] = decoder_tokens
        decoder_tokens_all[batch_range, masked_indices] = mask_tokens

        decoder_tokens_ground = decoder_tokens_all[:, :num_patches//2] + self.decoder_pos_emb_ground
        decoder_tokens_overhead = decoder_tokens_all[:, num_patches//2:] + self.decoder_pos_emb_overhead

        decoded_tokens_ground = self.decoder_ground(decoder_tokens_ground)
        decoded_tokens_overhead = self.decoder_overhead(decoder_tokens_overhead)

        # splice out the mask tokens and project to pixel values
        pred_pixel_values_ground = self.out(self.to_pixels_ground(decoded_tokens_ground))

        pred_pixel_values_overhead = self.out(self.to_pixels_overhead(decoded_tokens_overhead))

        pred_pixel_values_all = torch.cat((pred_pixel_values_ground, pred_pixel_values_overhead), dim=1)
        masked_pixel_preds = pred_pixel_values_all[batch_range, masked_indices]

        loss_recon = F.mse_loss(masked_pixel_preds, masked_patches)
        matching_labels = torch.cat(torch.ones(batch), torch.zeros(batch), device=device)
        loss_matching = F.cross_entropy(matching_probs, matching_labels)
        return 

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
        self.batch_size = kwargs.get('batch_size', 32)
        self.num_workers = kwargs.get('num_workers', 32)
        self.lr = kwargs.get('lr', 0.02)
        self.loss = MSE()
        self.loc_loss = nn.CrossEntropyLoss()
        self.acc = Accuracy(task='multiclass', num_classes=6)

    def forward(self, x, date):
        return self.model(x, date)

    def shared_step(self, batch, batch_idx):
        x, gt_loc, date = batch[0], batch[1], batch[2].float()
        #import code; code.interact(local=locals());
        pred, pm, gt, viz, loc = self(x, date)
        geo_loss = self.loc_loss(loc, gt_loc) 
        loss = self.loss(pm, gt) + 0.1*geo_loss
        #loss = geo_loss
        acc = self.acc(loc, gt_loc)
        return loss, viz, pred, x, geo_loss, acc

    def training_step(self, batch, batch_idx):
        loss, viz, pred, gt, geo_loss, acc = self.shared_step(batch, batch_idx)
        return {"loss": loss, "geo_loss": geo_loss, "acc":acc}

    def validation_step(self, batch, batch_idx):
        loss, viz, pred, gt, geo_loss, acc = self.shared_step(batch, batch_idx)
        return {"loss": loss, "pred": pred, "gt":gt, "viz":viz, "geo_loss":geo_loss, "acc": acc}

    def training_step_end(self, outputs):
        loss = outputs['loss'].mean()
        geo_loss = outputs['geo_loss'].mean()
        acc = outputs['acc'].mean()
        return {"loss": loss, 'geo_loss':geo_loss, 'acc':acc}

    def validation_step_end(self, outputs):
        loss = outputs['loss'].mean()
        geo_loss = outputs['geo_loss'].mean()
        pred = outputs['pred']
        gt_masked = outputs['viz']
        gt = outputs['gt']
        acc = outputs['acc'].mean()
        return {"loss": loss, "pred": pred, "gt":gt, "gt_masked":gt_masked, "geo_loss":geo_loss, 'acc':acc}

    def training_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        geo_loss = torch.stack([x['geo_loss'] for x in outputs]).mean()
        acc = torch.stack([x['acc'] for x in outputs]).mean()
        self.log('train_geo_loss', geo_loss, prog_bar=True, sync_dist=True)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        self.log('train_geo_acc', acc, prog_bar=True, sync_dist=True)

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        geo_loss = torch.stack([x['geo_loss'] for x in outputs]).mean()
        acc = torch.stack([x['acc'] for x in outputs]).mean()
        if self.current_epoch%50==0:
            ind = np.random.randint(0, len(outputs), size=5)
            for k in ind:
                pred = rearrange(outputs[k]['pred'][0], '(h w) (p1 p2 c) -> (h p1) (w p2) c', p1=16, p2=16, h=14)
                gt = rearrange(outputs[k]['gt'][0], 'c h w -> h w c')
                gtm = rearrange(outputs[k]['gt_masked'][0], '(h w) (p1 p2 c) -> (h p1) (w p2) c', p1=16, p2=16, h=14)
                pred = pred.cpu().detach().numpy()
                gt = gt.cpu().detach().numpy()
                gtm = gtm.cpu().detach().numpy()
                gtm = (gtm*255).astype(np.uint8)
                pred = (pred*255).astype(np.uint8)
                gt = (gt*255).astype(np.uint8)
                self.logger.experiment.log({
            "samples": [wandb.Image(img) for img in [gt, gtm, pred]]})

        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_geo_loss', geo_loss, prog_bar=True, sync_dist=True)
        self.log('val_geo_acc', acc, prog_bar=True, sync_dist=True)

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
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.05)
        scheduler = CosineAnnealingWarmRestarts(optimizer, 40)
        return [optimizer], [scheduler]

class Birds(Dataset):
    def __init__(self, dataset, label, val=False):
        self.dataset = dataset
        self.images = np.array(self.dataset['images'])
        self.idx = np.array(label.iloc[:, 1]).astype(int)
        self.labels = np.array(label.iloc[:, 2])
        self.images = self.images[self.idx]
        if not val:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.ColorJitter(),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]['file_name']
        img = Image.open(img_path)
        img = self.transform(img)
        date = self.images[idx]['date'].split(" ")[0]
        year = int(datetime.strptime(date, '%Y-%m-%d').date().strftime('%Y'))
        month = int(datetime.strptime(date, '%Y-%m-%d').date().strftime('%m'))
        day = int(datetime.strptime(date, '%Y-%m-%d').date().strftime('%d'))
        date_encode = torch.tensor([np.sin(2*np.pi*year/2023), np.cos(2*np.pi*year/2023), np.sin(2*np.pi*month/12), np.cos(2*np.pi*month/12), np.sin(2*np.pi*day/31), np.cos(2*np.pi*day/31)])
        return img, self.labels[idx], date_encode

if __name__=='__main__':
    f = open("eur_train.txt", "w")
    with redirect_stderr(f), redirect_stdout(f):
        torch.cuda.empty_cache()
        logger = WandbLogger(project="Meta-MAE", name="full train")
        train_dataset = json.load(open("train_birds.json"))
        train_labels = pd.read_csv('train_birds_labels_small.csv')
        train_dataset = Birds(train_dataset, train_labels)
        val_dataset = json.load(open("val_birds.json"))
        val_labels = pd.read_csv('val_birds_labels_small.csv')
        val_dataset = Birds(val_dataset, val_labels, val=True)

        checkpoint = ModelCheckpoint(
            monitor='val_loss',
            dirpath='checkpoints',
            filename='Try-{epoch:02d}-{val_loss:.2f}',
            mode='min'
        )

        model = MaeBirds(train_dataset, val_dataset)
       
        trainer = pl.Trainer(
            accelerator='gpu',
            devices=4,
            strategy='ddp',
            max_epochs=1500,
            num_nodes=1,
            callbacks=[checkpoint],
            logger=logger)
        trainer.fit(model)
