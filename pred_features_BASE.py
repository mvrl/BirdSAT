from vit_pytorch import ViT
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from torch.nn import L1Loss as MSE
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, datasets
import torch.nn.functional as F
from einops import repeat, rearrange
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
from vit_pytorch.vit import Transformer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from PIL import Image
import json

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
        self.to_patch, self.patch_to_emb = encoder.to_patch_embedding[:2]
        pixel_values_per_patch = self.patch_to_emb.weight.shape[-1]

        # decoder parameters
        self.decoder_dim = decoder_dim
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4)
        #self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.decoder_pos_emb = nn.Parameter(torch.randn(num_patches-1, decoder_dim))
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)
        self.out = nn.Sigmoid()

    def forward(self, img):
        device = img.device

        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape

        # patch to encoder tokens and add positions

        tokens = self.patch_to_emb(patches)
        tokens = tokens + self.encoder.pos_embedding[:, 1:(num_patches + 1)]

        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device = device).argsort(dim = -1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        # get the unmasked tokens to be encoded

        batch_range = torch.arange(batch, device = device)[:, None]
        tokens_unmasked = tokens[batch_range, unmasked_indices]

        # get the patches to be masked for the final reconstruction loss

        masked_patches = patches[batch_range, masked_indices]

        # attend with vision transformer

        encoded_tokens = self.encoder.transformer(tokens_unmasked)

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder

        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # reapply decoder position embedding to unmasked tokens

        decoder_tokens = decoder_tokens + self.decoder_pos_emb[unmasked_indices]

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above

        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb[masked_indices]

        # concat the masked tokens to the decoder tokens and attend with decoder
        decoder_tokens_all = torch.zeros(batch, num_patches, self.decoder_dim, device = device)

        decoder_tokens_all[batch_range, unmasked_indices] = decoder_tokens
        decoder_tokens_all[batch_range, masked_indices] = mask_tokens

        decoded_tokens = self.decoder(decoder_tokens_all)

        pred = self.out(self.to_pixels(decoded_tokens))

        # splice out the mask tokens and project to pixel values
        pred_mask_tokens = decoded_tokens[batch_range, masked_indices]
        pred_pixel_values = self.out(self.to_pixels(pred_mask_tokens))

        # calculate reconstruction loss
        viz = torch.ones(batch, num_patches, pred.shape[-1], device=device)*0.5
        viz[batch_range, unmasked_indices] = patches[batch_range, unmasked_indices]
        #recon_loss = F.mse_loss(pred_pixel_values, masked_patches)
        return pred, pred_pixel_values, masked_patches, viz

class MaeBirds(LightningModule):
    def __init__(self, train_dataset, val_dataset, **kwargs):
        super().__init__()
        self.vit = ViT(
            image_size = 224,
            patch_size = 8,
            num_classes = 10,
            dim = 1024,
            depth = 12,
            heads = 12,
            mlp_dim = 2048,
            dropout=0.1
        )
        self.model = MAE(
            encoder=self.vit,
            masking_ratio=0.75,
            decoder_dim=256,
            decoder_depth=8,
        )
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = kwargs.get('batch_size', 16)
        self.num_workers = kwargs.get('num_workers', 15)
        self.lr = kwargs.get('lr', 0.02)
        self.loss = MSE()

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch, batch_idx):
        x = batch
        pred, pm, gt, viz = self(x)
        loss = self.loss(pm, gt)
        return loss, viz, pred, x

    def training_step(self, batch, batch_idx):
        loss, viz, pred, gt = self.shared_step(batch, batch_idx)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss, viz, pred, gt = self.shared_step(batch, batch_idx)
        return {"loss": loss, "pred": pred, "gt":gt, "viz":viz}

    def training_step_end(self, outputs):
        loss = outputs['loss'].mean()
        return {"loss": loss}

    def validation_step_end(self, outputs):
        loss = outputs['loss'].mean()
        pred = outputs['pred']
        gt_masked = outputs['viz']
        gt = outputs['gt']
        return {"loss": loss, "pred": pred, "gt":gt, "gt_masked":gt_masked}

    def training_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        if self.current_epoch%50==0:
            ind = np.random.randint(0, len(outputs), size=5)
            for k in ind:
                pred = rearrange(outputs[k]['pred'][0], '(h w) (p1 p2 c) -> (h p1) (w p2) c', p1=8, p2=8, h=28)
                gt = rearrange(outputs[k]['gt'][0], 'c h w -> h w c')
                gtm = rearrange(outputs[k]['gt_masked'][0], '(h w) (p1 p2 c) -> (h p1) (w p2) c', p1=8, p2=8, h=28)
                pred = pred.cpu().detach().numpy()
                gt = gt.cpu().detach().numpy()
                gtm = gtm.cpu().detach().numpy()
                gtm = (gtm*255).astype(np.uint8)
                pred = (pred*255).astype(np.uint8)
                gt = (gt*255).astype(np.uint8)
                self.logger.experiment.log({
            "samples": [wandb.Image(img) for img in [gt, gtm, pred]]})

        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
    
    def predict_step(self, batch, batch_idx):
        x = batch
        device = self.device

        patches = self.model.to_patch(x)
        batch, num_patches, *_ = patches.shape

        # patch to encoder tokens and add positions

        tokens = self.model.patch_to_emb(patches)
        tokens = tokens + self.model.encoder.pos_embedding[:, 1:(num_patches + 1)]
        encoded_tokens = torch.mean(self.model.encoder.transformer(tokens), dim=1)
        return encoded_tokens

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
        optimizer = torch.optim.AdamW(self.parameters(), lr=1.5e-4, weight_decay=0.05)
        scheduler = CosineAnnealingWarmRestarts(optimizer, 40)
        return [optimizer], [scheduler]

class Birds(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.images = self.dataset['images']
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        if idx>=10000:
            raise IndexError
        img_path = self.images[idx]['file_name']
        img = Image.open(img_path)
        img = self.transform(img)
        return img


if __name__=='__main__':
    
    torch.cuda.empty_cache()
    train_dataset = json.load(open("train_mini_birds.json"))
    train_dataset = Birds(train_dataset)
    val_dataset = json.load(open("val_birds.json"))
    val_dataset = Birds(val_dataset)

    model = MaeBirds(train_dataset, val_dataset)
    model = model.load_from_checkpoint("./checkpoints/BaseMaePaper-epoch=118-val_loss=0.05.ckpt", train_dataset=None, val_dataset=None)

    predloader = DataLoader(train_dataset,
                            shuffle=False,
                            batch_size=32,
                            num_workers=16,
                            pin_memory=True)
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=1,
        num_nodes=1)
    
    preds = torch.cat(trainer.predict(model, predloader), dim=0).detach().cpu().numpy()
    #from sklearn.decomposition import PCA
    #pca = PCA(n_components=3)
    #rgb = pca.fit_transform(preds)
    #import pandas as pd
    #pd.DataFrame(rgb).to_csv("rgb_features.csv")
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=3)
    rgb = tsne.fit_transform(preds)
    import pandas as pd
    pd.DataFrame(rgb).to_csv("tsne_features.csv")
    
        
