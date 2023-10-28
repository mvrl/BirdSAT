from .MAEPretrain_SceneClassification.models_mae_vitae import (
    mae_vitae_base_patch16_dec512d8b,
    MaskedAutoencoderViTAE,
    CrossViewMaskedAutoencoder,
)
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
import torch.nn.functional as F
from torchvision.transforms import v2
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchmetrics import Accuracy
from functools import partial
from config import cfg
from tqdm import tqdm


class MAE(LightningModule):
    def __init__(self, train_dataset, val_dataset, **kwargs):
        super().__init__()
        assert not (cfg.pretrain.train.enabled and cfg.finetune.train.enabled)
        self.model = MaskedAutoencoderViTAE(
            img_size=cfg.pretrain.ground.img_size,
            patch_size=cfg.pretrain.ground.model.patch_size,
            in_chans=cfg.pretrain.ground.model.in_chans,
            embed_dim=cfg.pretrain.ground.model.embed_dim,
            depth=cfg.pretrain.ground.model.depth,
            num_heads=cfg.pretrain.ground.model.num_heads,
            decoder_embed_dim=cfg.pretrain.ground.model.decoder_embed_dim,
            decoder_depth=cfg.pretrain.ground.model.decoder_depth,
            decoder_num_heads=cfg.pretrain.ground.model.decoder_num_heads,
            mlp_ratio=cfg.pretrain.ground.model.mlp_ratio,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            norm_pix_loss=False,
            kernel=3,
            mlp_hidden_dim=None,
        )
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def setup_finetune(self):
        if cfg.finetune.train.enabled:
            if cfg.finetune.train.dataset == "iNAT":
                self.classify = nn.Linear(cfg.pretrain.ground.model.embed_dim, 1486)
                self.cutmix = v2.CutMix(num_classes=1486)
                self.mixup = v2.MixUp(num_classes=1486)
                self.acc = Accuracy(task="multiclass", num_classes=1486)
            elif cfg.finetune.train.dataset == "CUB":
                self.classify = nn.Linear(cfg.pretrain.ground.model.embed_dim, 200)
                self.cutmix = v2.CutMix(num_classes=200)
                self.mixup = v2.MixUp(num_classes=200)
                self.acc = Accuracy(task="multiclass", num_classes=200)
            elif cfg.finetune.train.dataset == "NABirds":
                self.classify = nn.Linear(cfg.pretrain.ground.model.embed_dim, 555)
                self.cutmix = v2.CutMix(num_classes=555)
                self.mixup = v2.MixUp(num_classes=555)
                self.acc = Accuracy(task="multiclass", num_classes=555)
            self.criterion = nn.CrossEntropyLoss(
                label_smoothing=cfg.finetune.train.label_smoothing
            )

    def forward(self, img_ground):
        loss_recon, *_ = self.model(img_ground)
        return loss_recon

    def forward_features(self, img_ground):
        embeddings, *_ = self.model.forward_encoder(img_ground, 0)
        return embeddings

    def forward_finetune(self, img_ground, label):
        img_ground, label = self.mixup(img_ground, label)
        img_ground, label = self.cutmix(img_ground, label)
        embeddings, *_ = self.model.forward_encoder(img_ground, 0)
        out = self.classify(embeddings[:, 0])
        return out

    def shared_step_pretrain(self, batch, batch_idx):
        img_ground = batch
        loss_recon = self(img_ground)
        return loss_recon

    def shared_step_finetune(self, batch, batch_idx):
        img_ground, labels = batch[0], batch[1]
        preds = self.forward_finetune(img_ground, labels)
        loss = self.criterion(preds, labels)
        acc = self.acc(preds, labels)
        return loss, acc

    def training_step(self, batch, batch_idx):
        if cfg.pretrain.train.enabled:
            loss_recon = self.shared_step_pretrain(batch, batch_idx)
            self.log(
                "train_loss", loss_recon, prog_bar=True, on_epoch=True, sync_dist=True
            )
            return {"loss": loss_recon}
        else:
            loss, acc = self.shared_step_finetune(batch, batch_idx)
            self.log("train_acc", acc, on_epoch=True, prog_bar=True)
            self.log("train_loss", loss, prog_bar=True, on_epoch=True)
            return {"loss": loss, "acc": acc}

    def validation_step(self, batch, batch_idx):
        if cfg.pretrain.train.enabled:
            loss_recon = self.shared_step_pretrain(batch, batch_idx)
            self.log(
                "val_loss", loss_recon, prog_bar=True, on_epoch=True, sync_dist=True
            )
            return {"loss": loss_recon}
        else:
            loss, acc = self.shared_step_finetune(batch, batch_idx)
            self.log("val_acc", acc, prog_bar=True, on_epoch=True)
            self.log("val_loss", loss, prog_bar=True, on_epoch=True)
            return {"loss": loss, "acc": acc}

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=cfg.pretrain.train.batch_size,
            num_workers=cfg.pretrain.train.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=cfg.pretrain.train.batch_size,
            num_workers=cfg.pretrain.train.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=cfg.pretrain.train.lr,
            weight_decay=cfg.pretrain.train.weight_decay,
        )
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, cfg.pretrain.train.warmup_epochs
        )
        return [optimizer, scheduler]


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class MoCoGeo(pl.LightningModule):
    def __init__(self, trainset, validset, queueset):
        super().__init__()
        print("Implementing MoCov2 Geo")
        self.train_dataset = trainset
        self.val_dataset = validset
        self.queue_dataset = queueset

        self.sat_encoder = mae_vitae_base_patch16_dec512d8b()
        self.sat_encoder.load_state_dict(
            torch.load(
                "/storage1/fs1/jacobsn/Active/user_s.sastry/Remote-Sensing-RVSA/vitae-b-checkpoint-1599-transform-no-average.pth"
            )["model"]
        )
        self.sat_encoder.requires_grad_(False)
        self.ground_encoder = MaskedAutoencoderViTAE(
            img_size=cfg.pretrain.ground.img_size,
            patch_size=cfg.pretrain.ground.model.patch_size,
            in_chans=cfg.pretrain.ground.model.in_chans,
            embed_dim=cfg.pretrain.ground.model.embed_dim,
            depth=cfg.pretrain.ground.model.depth,
            num_heads=cfg.pretrain.ground.model.num_heads,
            decoder_embed_dim=cfg.pretrain.ground.model.decoder_embed_dim,
            decoder_depth=cfg.pretrain.ground.model.decoder_depth,
            decoder_num_heads=cfg.pretrain.ground.model.decoder_num_heads,
            mlp_ratio=cfg.pretrain.ground.model.mlp_ratio,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            norm_pix_loss=False,
            kernel=3,
            mlp_hidden_dim=None,
        )
        self.geo_classify = nn.Linear(cfg.pretrain.ground.model.embed_dim, 20)

        # instantiate the learnable temperature parameter and queue size
        self.K = 10000

        # initialize queue
        self.register_buffer(
            "queue", torch.randn(cfg.pretrain.ground.model.embed_dim, self.K)
        )
        self.queue = nn.functional.normalize(self.queue, dim=0)
        # initialize pointer for queue
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    # fill the queue with initial values
    def on_fit_start(self):
        if cfg.pretrain.train.enabled:
            self.fill_queue(self.queue_dataloader())
        pass

    # forward function that runs during inference
    def forward(self, ground_img, sat_img):
        ground_embeddings, *_ = self.ground_encoder.forward_encoder(ground_img, 0.0)
        sat_embeddings, *_ = self.sat_encoder.forward_encoder(sat_img, 0.0)

        ground_embeddings = F.normalize(ground_embeddings[:, 0], dim=-1)
        sat_embeddings = F.normalize(sat_embeddings[:, 0], dim=-1)

        geo_cls = self.geo_classify(ground_embeddings)

        return ground_embeddings, sat_embeddings, geo_cls

    def forward_features(self, ground_img, sat_img):
        ground_embeddings, *_ = self.ground_encoder.forward_encoder(ground_img, 0.0)
        sat_embeddings, *_ = self.sat_encoder.forward_encoder(sat_img, 0.0)

        ground_embeddings = F.normalize(ground_embeddings[:, 0], dim=-1)
        sat_embeddings = F.normalize(sat_embeddings[:, 0], dim=-1)

        return ground_embeddings, sat_embeddings

    def forward_finetune(self, ground_img):
        ground_embeddings, *_ = self.ground_encoder.forward_encoder(ground_img, 0.0)
        ground_embeddings = F.normalize(ground_embeddings[:, 0], dim=-1)
        return self.classify(ground_embeddings)

    # shred step for test and validation that returns embeddings and clip loss
    def shared_step_pretrain(self, batch):
        # get embeddings from forward function
        ground_img, sat_img, geo_label = batch
        ground_embeddings, sat_embeddings, geo_cls = self(ground_img, sat_img)

        # positive logits:Nx1
        l_pos = torch.einsum("nc,nc->n", [ground_embeddings, sat_embeddings]).unsqueeze(
            -1
        )

        # negative logits: NxK
        l_neg = torch.einsum(
            "nc,ck->nk", [ground_embeddings, self.queue.clone().detach()]
        )

        logits = torch.cat([l_pos, l_neg], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
        loss_cont = F.cross_entropy(logits, labels)

        loss_geo = F.cross_entropy(geo_cls, geo_label)

        loss = loss_cont + loss_geo

        # dequeue and enqueue
        self._dequeue_and_enqueue(sat_embeddings)

        return loss, loss_cont, loss_geo

    def shared_step_finetune(self, batch, batch_idx):
        ground_img, labels = batch[0], batch[1]
        preds = self.forward_finetune(ground_img)
        loss = self.criterion(preds, labels)
        acc = self.acc(preds, labels)
        return loss, acc

    def setup_finetune(self):
        if cfg.finetune.train.enabled:
            if cfg.finetune.train.dataset == "iNAT":
                self.classify = nn.Linear(cfg.pretrain.ground.model.embed_dim, 1486)
                self.cutmix = v2.CutMix(num_classes=1486)
                self.mixup = v2.MixUp(num_classes=1486)
                self.acc = Accuracy(task="multiclass", num_classes=1486)
            elif cfg.finetune.train.dataset == "CUB":
                self.classify = nn.Linear(cfg.pretrain.ground.model.embed_dim, 200)
                self.cutmix = v2.CutMix(num_classes=200)
                self.mixup = v2.MixUp(num_classes=200)
                self.acc = Accuracy(task="multiclass", num_classes=200)
            elif cfg.finetune.train.dataset == "NABirds":
                self.classify = nn.Linear(cfg.pretrain.ground.model.embed_dim, 555)
                self.cutmix = v2.CutMix(num_classes=555)
                self.mixup = v2.MixUp(num_classes=555)
                self.acc = Accuracy(task="multiclass", num_classes=555)
            self.criterion = nn.CrossEntropyLoss(
                label_smoothing=cfg.finetune.train.label_smoothing
            )

    # forward pass for each batch in training
    def training_step(self, batch, batch_idx):
        if cfg.pretrain.train.enabled:
            outputs = self.shared_step_pretrain(batch)
            train_loss = outputs[0]
            train_contrastive_loss = outputs[1]
            train_geo_loss = outputs[2]
            self.log("train_loss", train_loss, sync_dist=True)
            self.log(
                "contrastive_loss",
                train_contrastive_loss,
                prog_bar=True,
                sync_dist=True,
            )
            self.log("train_geo_loss", train_geo_loss, prog_bar=True, sync_dist=True)
        else:
            loss, acc = self.shared_step_finetune(batch, batch_idx)
            self.log("train_acc", acc, on_epoch=True, prog_bar=True)
            self.log("train_loss", loss, prog_bar=True, on_epoch=True)
            return {"loss": loss, "acc": acc}
        return train_loss

    # forward pass for each batch in validation
    def validation_step(self, batch, batch_idx):
        if cfg.pretrain.train.enabled:
            outputs = self.shared_step_pretrain(batch)
            val_loss = outputs[0]
            val_contrastive_loss = outputs[1]
            val_geo_loss = outputs[2]
            self.log("val_loss", val_loss, sync_dist=True, prog_bar=True)
            self.log(
                "val_contrastive_loss",
                val_contrastive_loss,
                prog_bar=True,
                sync_dist=True,
            )
            self.log("val_geo_loss", val_geo_loss, sync_dist=True, prog_bar=True)
        else:
            loss, acc = self.shared_step_finetune(batch, batch_idx)
            self.log("val_acc", acc, prog_bar=True, on_epoch=True)
            self.log("val_loss", loss, prog_bar=True, on_epoch=True)
            return {"loss": loss, "acc": acc}
        return val_loss

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=cfg.pretrain.train.batch_size,
            num_workers=cfg.pretrain.train.num_workers,
            persistent_workers=False,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=cfg.pretrain.train.batch_size,
            num_workers=cfg.pretrain.train.num_workers,
            persistent_workers=False,
            pin_memory=True,
            drop_last=True,
        )

    def queue_dataloader(self):
        return DataLoader(
            self.queue_dataset,
            shuffle=False,
            batch_size=cfg.pretrain.train.batch_size,
            num_workers=cfg.pretrain.train.num_workers,
            persistent_workers=False,
            pin_memory=False,
        )

    def configure_optimizers(self):
        self.optim = torch.optim.AdamW(
            self.parameters(),
            lr=cfg.pretrain.train.lr,
            weight_decay=cfg.pretrain.train.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-6,
        )

        self.warm_up_iterations = 300
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=self.optim, T_0=self.warm_up_iterations
        )

        return [self.optim], [self.scheduler]

    @torch.no_grad()
    def fill_queue(self, dataset):
        num_steps_to_fill = 100
        # fill_dataset = MultiData(self.hparams).get_ds(mode='queue')
        print("Filling Queue with initial values")
        for i, batch in tqdm(enumerate(dataset)):
            ground_img, sat_img, geo_label = batch
            # get the ground img encodings and detach
            with torch.set_grad_enabled(False):  # equivalent to torch.no_grad()
                sat_embeddings, *_ = self.sat_encoder.forward_encoder(
                    sat_img.to(self.device), 0.0
                )  # keys: NxC
                sat_embeddings = F.normalize(sat_embeddings[:, 0], dim=-1)
                self._dequeue_and_enqueue(sat_embeddings)
            if i == num_steps_to_fill:
                break

        print("Queue Fill Successful")

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert (
            self.K % batch_size == 0
        ), f"Size Mismatch: Queue Size is {self.K} and Batch size is {batch_size}"  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr


class CVEMAEMeta(LightningModule):
    def __init__(self, train_dataset, val_dataset, **kwargs):
        super().__init__()
        assert not (cfg.pretrain.train.enabled and cfg.finetune.train.enabled)
        self.sat_encoder = mae_vitae_base_patch16_dec512d8b()
        self.sat_encoder.load_state_dict(
            torch.load(
                "pretrained_models/vitae-b-checkpoint-1599-transform-no-average.pth"
            )["model"]
        )
        self.sat_encoder.requires_grad_(False)
        self.ground_encoder = MaskedAutoencoderViTAE(
            img_size=cfg.pretrain.ground.img_size,
            patch_size=cfg.pretrain.ground.model.patch_size,
            in_chans=cfg.pretrain.ground.model.in_chans,
            embed_dim=cfg.pretrain.ground.model.embed_dim,
            depth=cfg.pretrain.ground.model.depth,
            num_heads=cfg.pretrain.ground.model.num_heads,
            decoder_embed_dim=cfg.pretrain.ground.model.decoder_embed_dim,
            decoder_depth=cfg.pretrain.ground.model.decoder_depth,
            decoder_num_heads=cfg.pretrain.ground.model.decoder_num_heads,
            mlp_ratio=cfg.pretrain.ground.model.mlp_ratio,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            norm_pix_loss=False,
            kernel=3,
            mlp_hidden_dim=None,
        )
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        if cfg.pretrain.train.mode == "full_metadata":
            self.geo_encode = nn.Linear(4, cfg.pretrain.ground.model.embed_dim)
            self.date_encode = nn.Linear(4, cfg.pretrain.ground.model.embed_dim)

    def setup_finetune(self):
        if cfg.finetune.train.enabled:
            if cfg.finetune.train.dataset == "iNAT":
                self.classify = nn.Linear(cfg.pretrain.ground.model.embed_dim, 1486)
                self.cutmix = v2.CutMix(num_classes=1486)
                self.mixup = v2.MixUp(num_classes=1486)
                self.acc = Accuracy(task="multiclass", num_classes=1486)
            elif cfg.finetune.train.dataset == "CUB":
                self.classify = nn.Linear(cfg.pretrain.ground.model.embed_dim, 200)
                self.cutmix = v2.CutMix(num_classes=200)
                self.mixup = v2.MixUp(num_classes=200)
                self.acc = Accuracy(task="multiclass", num_classes=200)
            elif cfg.finetune.train.dataset == "NABirds":
                self.classify = nn.Linear(cfg.pretrain.ground.model.embed_dim, 555)
                self.cutmix = v2.CutMix(num_classes=555)
                self.mixup = v2.MixUp(num_classes=555)
                self.acc = Accuracy(task="multiclass", num_classes=555)
            self.criterion = nn.CrossEntropyLoss(
                label_smoothing=cfg.finetune.train.label_smoothing
            )

    def forward(self, img_ground, img_overhead, geoloc=None, date=None):
        norm_ground_features, norm_overhead_features = self.forward_features(
            img_ground, img_overhead, geoloc, date
        )
        similarity = torch.einsum(
            "ij,kj->ik", norm_ground_features, norm_overhead_features
        )
        labels_clip = torch.arange(similarity.shape[0], device=self.device).long()
        loss_clip = (
            F.cross_entropy(similarity, labels_clip)
            + F.cross_entropy(similarity.T, labels_clip)
        ) / 2
        loss_recon, *_ = self.ground_encoder(img_ground)
        return loss_clip, loss_recon

    def forward_features(self, img_ground, img_overhead, geoloc=None, date=None):
        ground_embeddings, *_ = self.ground_encoder.forward_encoder(img_ground, 0)
        sat_embeddings, *_ = self.sat_encoder.forward_encoder(img_overhead, 0)
        norm_ground_features = F.normalize(ground_embeddings[:, 0], dim=-1)
        if geoloc is None or date is None:
            norm_overhead_features = F.normalize(sat_embeddings[:, 0], dim=-1)
        else:
            if torch.rand(1) < cfg.pretrain.train.meta_dropout_prob:
                norm_overhead_features = F.normalize(sat_embeddings[:, 0], dim=-1)
            else:
                geo_token = self.geo_encode(geoloc)
                date_token = self.date_encode(date)
                norm_overhead_features = F.normalize(
                    sat_embeddings[:, 0] + geo_token + date_token, dim=-1
                )
        return norm_ground_features, norm_overhead_features

    def forward_finetune(self, img_ground, label, geoloc=None, date=None):
        img_ground, label = self.mixup(img_ground, label)
        img_ground, label = self.cutmix(img_ground, label)
        ground_embeddings, *_ = self.ground_encoder.forward_encoder(img_ground, 0)
        if geoloc is None or date is None:
            norm_ground_features = F.normalize(ground_embeddings[:, 0], dim=-1)
        else:
            if torch.rand(1) < cfg.pretrain.train.meta_dropout_prob:
                norm_ground_features = F.normalize(ground_embeddings[:, 0], dim=-1)
            else:
                geo_token = self.geo_encode(geoloc)
                date_token = self.date_encode(date)
                norm_ground_features = F.normalize(
                    ground_embeddings[:, 0] + geo_token + date_token, dim=-1
                )
        out = self.classify(norm_ground_features)
        return out

    def shared_step_pretrain(self, batch, batch_idx):
        if cfg.pretrain.train.mode == "no_metadata":
            img_ground, img_overhead = batch[0], batch[1]
            loss_clip, loss_recon = self(img_ground, img_overhead)
        else:
            img_ground, img_overhead, geoloc, date = (
                batch[0],
                batch[1],
                batch[2],
                batch[3],
            )
            loss_clip, loss_recon = self(img_ground, img_overhead, geoloc, date)
        loss = 0.3 * loss_clip + loss_recon
        return loss, loss_clip, loss_recon

    def shared_step_finetune(self, batch, batch_idx):
        if cfg.pretrain.train.mode == "no_metadata":
            img_ground, labels = batch[0], batch[1]
            preds = self.forward_finetune(img_ground, labels)
        else:
            img_ground, labels, geoloc, date = batch[0], batch[1], batch[2], batch[3]
            preds = self.forward_finetune(img_ground, labels, geoloc, date)
        loss = self.criterion(preds, labels)
        acc = self.acc(preds, labels)
        return loss, acc

    def training_step(self, batch, batch_idx):
        if cfg.pretrain.train.enabled:
            loss, loss_clip, loss_recon = self.shared_step_pretrain(batch, batch_idx)
            self.log(
                "train_loss_recon",
                loss_recon,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            self.log("train_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
            self.log(
                "train_loss_clip",
                loss_clip,
                prog_bar=True,
                on_epoch=True,
                sync_dist=True,
            )
            return {"loss": loss, "loss_clip": loss_clip, "loss_recon": loss_recon}
        else:
            loss, acc = self.shared_step_finetune(batch, batch_idx)
            self.log("train_acc", acc, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
            return {"loss": loss, "acc": acc}

    def validation_step(self, batch, batch_idx):
        if cfg.pretrain.train.enabled:
            loss, loss_clip, loss_recon = self.shared_step_pretrain(batch, batch_idx)
            self.log(
                "val_loss_recon",
                loss_recon,
                prog_bar=True,
                on_epoch=True,
                sync_dist=True,
            )
            self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
            self.log(
                "val_loss_clip", loss_clip, prog_bar=True, on_epoch=True, sync_dist=True
            )
            return {"loss": loss, "loss_clip": loss_clip, "loss_recon": loss_recon}
        else:
            loss, acc = self.shared_step_finetune(batch, batch_idx)
            self.log("val_acc", acc, prog_bar=True, on_epoch=True, sync_dist=True)
            self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
            return {"loss": loss, "acc": acc}

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=cfg.pretrain.train.batch_size,
            num_workers=cfg.pretrain.train.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=cfg.pretrain.train.batch_size,
            num_workers=cfg.pretrain.train.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=cfg.pretrain.train.lr,
            weight_decay=cfg.pretrain.train.weight_decay,
        )
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, cfg.pretrain.train.warmup_epochs
        )
        return [optimizer, scheduler]


class CVMMAEMeta(LightningModule):
    def __init__(self, train_dataset, val_dataset, **kwargs):
        super().__init__()
        assert not (cfg.pretrain.train.enabled and cfg.finetune.train.enabled)
        self.model = CrossViewMaskedAutoencoder(
            img_size1=cfg.pretrain.ground.img_size,
            img_size2=cfg.pretrain.overhead.img_size,
            patch_size1=cfg.pretrain.ground.model.patch_size,
            patch_size2=cfg.pretrain.overhead.patch_size,
            in_chans=cfg.pretrain.ground.model.in_chans,
            embed_dim=cfg.pretrain.ground.model.embed_dim,
            depth=cfg.pretrain.ground.model.depth,
            num_heads=cfg.pretrain.ground.model.num_heads,
            decoder_embed_dim=cfg.pretrain.ground.model.decoder_embed_dim,
            decoder_depth=cfg.pretrain.ground.model.decoder_depth,
            decoder_num_heads=cfg.pretrain.ground.model.decoder_num_heads,
            mlp_ratio=cfg.pretrain.ground.model.mlp_ratio,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            norm_pix_loss=False,
            kernel=3,
            mlp_hidden_dim=None,
        )
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.match = nn.Linear(cfg.pretrain.ground.model.embed_dim, 1)
        if cfg.pretrain.train.mode == "full_metadata":
            self.geo_encode = nn.Linear(4, cfg.pretrain.ground.model.embed_dim)
            self.date_encode = nn.Linear(4, cfg.pretrain.ground.model.embed_dim)

    def setup_finetune(self):
        if cfg.finetune.train.enabled:
            if cfg.finetune.train.dataset == "iNAT":
                self.classify = nn.Linear(cfg.pretrain.ground.model.embed_dim, 1486)
                self.cutmix = v2.CutMix(num_classes=1486)
                self.mixup = v2.MixUp(num_classes=1486)
                self.acc = Accuracy(task="multiclass", num_classes=1486)
            elif cfg.finetune.train.dataset == "CUB":
                self.classify = nn.Linear(cfg.pretrain.ground.model.embed_dim, 200)
                self.cutmix = v2.CutMix(num_classes=200)
                self.mixup = v2.MixUp(num_classes=200)
                self.acc = Accuracy(task="multiclass", num_classes=200)
            elif cfg.finetune.train.dataset == "NABirds":
                self.classify = nn.Linear(cfg.pretrain.ground.model.embed_dim, 555)
                self.cutmix = v2.CutMix(num_classes=555)
                self.mixup = v2.MixUp(num_classes=555)
                self.acc = Accuracy(task="multiclass", num_classes=555)

    def forward(self, img_ground, img_overhead, geoloc, date):
        img_ground_ex = torch.cat((img_ground, img_ground), dim=0)
        img_overhead_ex = torch.cat(
            (img_overhead, torch.roll(img_overhead, 1, 0)), dim=0
        )
        embeddings, *_ = self.model.forward_encoder(img_ground_ex, img_overhead_ex, 0)
        if cfg.pretrain.train.mode == "no_metadata":
            matching_probs = self.match(embeddings[:, 0])
        else:
            if torch.rand(1) < cfg.pretrain.train.meta_dropout_prob:
                matching_probs = self.match(embeddings[:, 0])
            else:
                geo_token = self.geo_encode(geoloc)
                date_token = self.date_encode(date)
                geo_ex = torch.cat((geo_token, torch.roll(geo_token, 1, 0)), dim=0)
                date_ex = torch.cat((date_token, torch.roll(date_token, 1, 0)), dim=0)
                matching_probs = self.match(embeddings[:, 0] + geo_ex + date_ex)
        matching_labels = torch.cat(
            (
                torch.ones(img_ground.shape[0], device=self.device),
                torch.zeros(img_ground.shape[0], device=self.device),
            )
        )
        loss_matching = F.binary_cross_entropy_with_logits(
            matching_probs.squeeze(-1), matching_labels
        )
        loss_recon, *_ = self.model(img_ground, img_overhead)
        return loss_matching, loss_recon

    def forward_features(self, img_ground, img_overhead, geoloc=None, date=None):
        embeddings, *_ = self.model.forward_encoder(img_ground, img_overhead, 0)
        if cfg.pretrain.train.mode == "no_metadata":
            norm_embeddings = F.normalize(embeddings[:, 0], dim=-1)
        else:
            if torch.rand(1) < cfg.pretrain.train.meta_dropout_prob:
                norm_embeddings = F.normalize(embeddings[:, 0], dim=-1)
            else:
                geo_token = self.geo_encode(geoloc)
                date_token = self.date_encode(date)
                norm_embeddings = F.normalize(
                    embeddings[:, 0] + geo_token + date_token, dim=-1
                )
        return norm_embeddings, self.match(norm_embeddings)

    def forward_finetune(self, img_ground, img_overhead, label, geoloc=None, date=None):
        img_ground, label = self.mixup(img_ground, label)
        img_ground, label = self.cutmix(img_ground, label)
        embeddings, *_ = self.model.forward_encoder(img_ground, img_overhead, 0)
        if cfg.pretrain.train.mode == "no_metadata":
            norm_embeddings = F.normalize(embeddings[:, 0], dim=-1)
        else:
            if torch.rand(1) < cfg.pretrain.train.meta_dropout_prob:
                norm_embeddings = F.normalize(embeddings[:, 0], dim=-1)
            else:
                geo_token = self.geo_encode(geoloc)
                date_token = self.date_encode(date)
                norm_embeddings = F.normalize(
                    embeddings[:, 0] + geo_token + date_token, dim=-1
                )
        out = self.classify(norm_embeddings)
        return out

    def shared_step_pretrain(self, batch, batch_idx):
        if cfg.pretrain.train.mode == "no_metadata":
            img_ground, img_overhead = batch[0], batch[1]
            loss_matching, loss_recon = self(img_ground, img_overhead)
        else:
            img_ground, img_overhead, geoloc, date = (
                batch[0],
                batch[1],
                batch[2],
                batch[3],
            )
            loss_matching, loss_recon = self(img_ground, img_overhead, geoloc, date)
        loss = loss_matching + loss_recon
        return loss, loss_matching, loss_recon

    def shared_step_finetune(self, batch, batch_idx):
        if cfg.pretrain.train.mode == "no_metadata":
            img_ground, img_overhead, labels = batch[0], batch[1], batch[2]
            preds = self.forward_finetune(img_ground, img_overhead, labels)
        else:
            img_ground, img_overhead, labels, geoloc, date = (
                batch[0],
                batch[1],
                batch[2],
                batch[3],
                batch[4],
            )
            preds = self.forward_finetune(
                img_ground, img_overhead, labels, geoloc, date
            )
        loss = self.criterion(preds, labels)
        acc = self.acc(preds, labels)
        return loss, acc

    def training_step(self, batch, batch_idx):
        if cfg.pretrain.train.enabled:
            loss, loss_clip, loss_recon = self.shared_step_pretrain(batch, batch_idx)
            self.log(
                "train_loss_recon",
                loss_recon,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            self.log("train_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
            self.log(
                "train_loss_clip",
                loss_clip,
                prog_bar=True,
                on_epoch=True,
                sync_dist=True,
            )
            return {"loss": loss, "loss_clip": loss_clip, "loss_recon": loss_recon}
        else:
            loss, acc = self.shared_step_finetune(batch, batch_idx)
            self.log("train_acc", acc, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
            return {"loss": loss, "acc": acc}

    def validation_step(self, batch, batch_idx):
        if cfg.pretrain.train.enabled:
            loss, loss_clip, loss_recon = self.shared_step_pretrain(batch, batch_idx)
            self.log(
                "val_loss_recon",
                loss_recon,
                prog_bar=True,
                on_epoch=True,
                sync_dist=True,
            )
            self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
            self.log(
                "val_loss_clip", loss_clip, prog_bar=True, on_epoch=True, sync_dist=True
            )
            return {"loss": loss, "loss_clip": loss_clip, "loss_recon": loss_recon}
        else:
            loss, acc = self.shared_step_finetune(batch, batch_idx)
            self.log("val_acc", acc, prog_bar=True, on_epoch=True, sync_dist=True)
            self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
            return {"loss": loss, "acc": acc}

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=cfg.pretrain.train.batch_size,
            num_workers=cfg.pretrain.train.num_workers,
            persistent_workers=False,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=cfg.pretrain.train.batch_size,
            num_workers=cfg.pretrain.train.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=cfg.pretrain.train.lr,
            weight_decay=cfg.pretrain.train.weight_decay,
        )
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, cfg.pretrain.train.warmup_epochs
        )
        return [optimizer, scheduler]
