from vit_pytorch import ViT
import torch
import torch.nn as nn
from torch import einsum
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
from einops.layers.torch import Rearrange

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# positional embedding

def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)

# feedforward

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# (cross)attention

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None):
        b, n, _, h = *x.shape, self.heads

        x = self.norm(x)

        context = self.norm(context) if exists(context) else x

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x, context = None):
        for attn, ff in self.layers:
            x = attn(x, context = context) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, num_classes, image_size, patch_size, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.dim = dim
        self.num_patches = num_patches

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, img):
        *_, h, w, dtype = *img.shape, img.dtype

        x = self.to_patch_embedding(img)
        pe = posemb_sincos_2d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        x = self.transformer(x)

        return x

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
        self.to_patch_ground = encoder.to_patch_embedding[0]
        self.patch_to_emb_ground = nn.Sequential(*encoder.to_patch_embedding[1:])
        self.to_patch_overhead, self.patch_to_emb_overhead = copy.deepcopy(self.to_patch_ground), copy.deepcopy(self.patch_to_emb_ground)
        pixel_values_per_patch = encoder.to_patch_embedding[2].weight.shape[-1]
        self.iim_token = nn.Parameter(torch.randn(1, encoder_dim))
        self.enc_embed = nn.Parameter(torch.randn(num_patches, encoder_dim))
        self.match = nn.Linear(encoder_dim, 1)

        # decoder parameters
        self.decoder_dim = decoder_dim
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder_ground = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4)
        #self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.decoder_pos_emb_ground = nn.Parameter(torch.randn(num_patches-1, decoder_dim))
        self.to_pixels_ground = nn.Linear(decoder_dim, pixel_values_per_patch)
        self.out = nn.Sigmoid()

    def forward(self, img_ground, img_overhead):
        device = img_ground.device

        patches_ground = self.to_patch_ground(img_ground)
        #patches_overhead = self.to_patch_overhead(img_overhead)
        #patches_all = torch.cat((patches_ground, patches_overhead), dim=1)
        batch, num_patches, *_ = patches_ground.shape

        # patch to encoder tokens and add positions
        tokens_ground = self.patch_to_emb_ground(patches_ground)
        #tokens_overhead = self.patch_to_emb_overhead(patches_overhead)

        iim_token = repeat(self.iim_token, 'n d -> b n d', b = batch)
        tokens_matching = torch.cat((iim_token, tokens_ground), dim=1)
        tokens_matching = tokens_matching + self.enc_embed

        enc_tokens_matching = self.encoder.transformer(tokens_matching)
        return enc_tokens_matching[:, 0]

class MaeBirds(LightningModule):
    def __init__(self, train_dataset, val_dataset, **kwargs):
        super().__init__()
        self.vit = ViT(
            image_size = 384,
            patch_size = 32,
            num_classes = 1,
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
        self.num_workers = kwargs.get('num_workers', 16)
        self.lr = kwargs.get('lr', 0.02)

    def forward(self, img_ground, img_overhead):
        return self.model(img_ground, img_overhead)

class MaeBirdsDownstream(LightningModule):
    def __init__(self, train_dataset, val_dataset, **kwargs):
        super().__init__()
        self.model = MaeBirds.load_from_checkpoint('/storage1/fs1/jacobsn/Active/user_s.sastry/metaformer/checkpoints/CrossMAEv2-epoch=00-val_loss=0.70.ckpt', train_dataset=train_dataset, val_dataset=val_dataset)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = kwargs.get('batch_size', 16)
        self.num_workers = kwargs.get('num_workers', 16)
        self.lr = kwargs.get('lr', 0.02)
        self.classify = nn.Linear(1024, 1486)
        self.criterion = nn.CrossEntropyLoss()
        self.acc = Accuracy(task='multiclass', num_classes=1486)

    def forward(self, img_ground, img_overhead):
        return self.classify(self.model(img_ground, img_overhead))

    def shared_step(self, batch, batch_idx):
        img_ground, img_overhead, labels = batch[0], batch[1], batch[2]
        #import code; code.interact(local=locals());
        preds = self(img_ground, img_overhead)
        loss = self.criterion(preds, labels)
        acc = self.acc(preds, labels)
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
            self.transform_ground = transforms.Compose([
                transforms.Resize((384, 384)),
                #transforms.CenterCrop((224, 224)),
                transforms.ToTensor()
            ])
            self.transform_overhead = transforms.Compose([
                transforms.Resize(384),
                transforms.ToTensor()
            ])
        else:
            self.transform_ground = transforms.Compose([
                transforms.Resize((384, 384)),
                #transforms.CenterCrop((224, 224)),
                transforms.ToTensor()
            ])
            self.transform_overhead = transforms.Compose([
                transforms.Resize(384),
                transforms.ToTensor()
            ])
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]['file_name']
        label = self.species[self.categories[idx]['category_id']]
        img_ground = Image.open(img_path)
        img_ground = self.transform_ground(img_ground)
        if not self.val:
            img_overhead = Image.open(f"/scratch1/fs1/jacobsn/s.sastry/metaformer/train_overhead/images_sentinel/{idx}.jpeg")
        else:
            img_overhead = Image.open(f"/scratch1/fs1/jacobsn/s.sastry/metaformer/val_overhead/images_sentinel/{idx}.jpeg")
        img_overhead = self.transform_overhead(img_overhead)
        return img_ground, img_overhead, torch.tensor(label)

if __name__=='__main__':
    torch.cuda.empty_cache()
    logger = WandbLogger(project="Cross-View-MAE", name="Cross View Downstream")
    train_dataset = json.load(open("train_birds.json"))
    train_labels = pd.read_csv('train_birds_labels.csv')
    train_dataset = Birds(train_dataset, train_labels)
    val_dataset = json.load(open("val_birds.json"))
    val_labels = pd.read_csv('val_birds_labels.csv')
    val_dataset = Birds(val_dataset, val_labels, val=True)

    checkpoint = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='CrossMAE_Downstreamv2-{epoch:02d}-{val_loss:.2f}',
        mode='min'
    )

    model = MaeBirdsDownstream(train_dataset, val_dataset)
    
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=2,
        strategy='ddp_find_unused_parameters_true',
        max_epochs=1500,
        num_nodes=1,
        callbacks=[checkpoint],
        logger=logger)
    trainer.fit(model)
