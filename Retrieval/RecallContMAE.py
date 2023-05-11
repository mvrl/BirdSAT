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
from tqdm import tqdm
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
        return ground_embeddings[:, 0], sat_embeddings[:, 0]

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
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.transform_overhead = transforms.Compose([
                transforms.Resize(224),
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
        label = self.species[self.categories[idx]['category_id']]
        img_ground = Image.open(os.path.join('/storage1/fs1/jacobsn/Active/user_s.sastry/metaformer/', img_path))
        img_ground = self.transform_ground(img_ground)
        if not self.val:
            img_overhead = Image.open(f"/scratch1/fs1/jacobsn/s.sastry/metaformer/train_overhead/images_sentinel/{idx}.jpeg")
        else:
            img_overhead = Image.open(f"/scratch1/fs1/jacobsn/s.sastry/metaformer/val_overhead/images_sentinel/{idx}.jpeg")
        img_overhead = self.transform_overhead(img_overhead)
        return img_ground, img_overhead, torch.tensor(label)

if __name__=='__main__':
    torch.cuda.empty_cache()
    val_dataset = json.load(open("/storage1/fs1/jacobsn/Active/user_s.sastry/metaformer/val_birds.json"))
    val_labels = pd.read_csv('/storage1/fs1/jacobsn/Active/user_s.sastry/metaformer/val_birds_labels.csv')
    val_dataset = Birds(val_dataset, val_labels, val=True)

    model = MaeBirds.load_from_checkpoint('/storage1/fs1/jacobsn/Active/user_s.sastry/Remote-Sensing-RVSA/ContrastiveMAEv5-epoch=44-val_loss=1.60.ckpt', train_dataset=val_dataset, val_dataset=val_dataset)
    model = model.eval()
    val_overhead = DataLoader(val_dataset,
                        shuffle=False,
                        batch_size=77,
                        num_workers=8,
                        persistent_workers=False,
                        pin_memory=True,
                        drop_last=True
                        )
    
    recall = 0
    for batch in tqdm(val_overhead):
        #for batch2 in tqdm(val_overhead):
        img_ground, img_overhead, label = batch
        z = 0 
        running_val = 0
        running_label = 0
        for batch2 in tqdm(val_overhead):
            img_ground2, img_overhead2, label2 = batch2
            ground_embeddings, overhead_embeddings = model(img_ground2.cuda(), img_overhead.cuda())
            norm_ground_features = F.normalize(ground_embeddings, dim=-1)
            norm_overhead_features = F.normalize(overhead_embeddings, dim=-1)
            similarity = torch.einsum('ij,kj->ik', norm_ground_features, norm_overhead_features)
            vals, ind = torch.topk(similarity.detach().cpu(), 10, dim=0)
            if z==0:
                running_val = vals
                running_label = label2[ind]
                z+=1
            else:
                running_val = torch.cat((running_val, vals), dim=0)
                running_label = torch.cat((running_label, label2[ind]), dim=0)
        _, ind = torch.topk(running_val, 10, dim=0)

        #import code; code.interact(local=locals())
        preds = running_label[ind]
        recall+=sum([1 if label[i] in preds[:, i] else 0 for i in range(label.shape[0])])
        #import code; code.interact(local=locals())
        print(f"Current Recall Score: {recall}")
