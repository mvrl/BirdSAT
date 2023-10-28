import torch
import pytorch_lightning as pl
from datasets import CrossViewiNATBirdsFineTune, HeirarchicalRet
from models import MAE, CVEMAEMeta, CVMMAEMeta
from torch.utils.data import random_split
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
from pytorch_lightning.loggers import WandbLogger
import json
import pandas as pd
from config import cfg
from utils import seed_everything
from tqdm import tqdm


def retrieval_eval():
    torch.cuda.empty_cache()
    seed_everything()

    test_json = json.load(open("data/val_birds.json"))
    test_labels = pd.read_csv("data/val_birds_labels.csv")
    test_dataset = CrossViewiNATBirdsFineTune(test_json, test_labels, val=True)

    if cfg.retrieval.model_type == "MAE":
        model = MAE.load_from_checkpoint(
            cfg.retrieval.ckpt,
            train_dataset=None,
            val_dataset=test_dataset,
        )
    elif cfg.retrieval.model_type == "CVEMAE":
        model = CVEMAEMeta.load_from_checkpoint(
            cfg.retrieval.ckpt,
            train_dataset=None,
            val_dataset=test_dataset,
        )
    elif cfg.retrieval.model_type == "CVMMAE":
        model_filter = CVMMAEMeta.load_from_checkpoint(
            cfg.retrieval.ckpt,
            train_dataset=None,
            val_dataset=test_dataset,
        )
        model_filter.eval()
        model_filter = model_filter.cuda()
        model = CVEMAEMeta.load_from_checkpoint(
            cfg.retrieval.cve_ckpt,
            train_dataset=None,
            val_dataset=test_dataset,
        )

    model.eval()
    model = model.cuda()

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.retrieval.batch_size,
        shuffle=False,
        num_workers=cfg.retrieval.num_workers,
    )

    recall = 0

    for batch in tqdm(test_loader):
        if cfg.retrieval.mode == "full_metadata":
            _, img_overhead, label, *_ = batch
        else:
            _, img_overhead, label = batch
        z = 0
        running_val = 0
        running_label = 0
        for batch2 in tqdm(test_loader):
            if cfg.retrieval.mode == "full_metadata":
                img_ground, _, label2, geoloc, date = batch2
                ground_embeddings, overhead_embeddings = model.forward_features(
                    img_ground.cuda(), img_overhead.cuda(), geoloc.cuda(), date.cuda()
                )
            else:
                img_ground, _, label2 = batch2
                ground_embeddings, overhead_embeddings = model.forward_features(
                    img_ground.cuda(), img_overhead.cuda()
                )

            similarity = torch.einsum(
                "ij,kj->ik", ground_embeddings, overhead_embeddings
            )
            if z == 0:
                running_val = similarity.detach().cpu()
                running_label = label2
                z += 1
            else:
                running_val = torch.cat((running_val, similarity.detach().cpu()), dim=0)
                running_label = torch.cat((running_label, label2), dim=0)
        if cfg.retrieval.model_type == "CVEMAE":
            _, ind = torch.topk(running_val, 10, dim=0)

        # Hierarchical Retrieval
        elif cfg.retrieval.model_type == "CVMMAE":
            _, ind = torch.topk(running_val, 50, dim=0)
            if cfg.retrieval.mode == "full_metadata":
                img_ground, _, label2, geoloc, date = test_dataset[ind]
            else:
                img_ground, _, label2 = test_dataset[ind]
            similarity = torch.zeros((50, 50))
            idx = torch.arange(50)
            for i in range(50):
                img_ground_rolled = torch.roll(img_ground, i, 0)
                idx_rolled = torch.roll(idx, i, 0)
                if cfg.retrieval.mode == "full_metadata":
                    _, scores = model_filter.forward_features(
                        img_ground_rolled.cuda(),
                        img_overhead.cuda(),
                        geoloc.cuda(),
                        date.cuda(),
                    )
                else:
                    _, scores = model_filter.forward_features(
                        img_ground_rolled.cuda(), img_overhead.cuda()
                    )
                similarity[idx_rolled, torch.arange(50)] = (
                    scores.squeeze(0).detach().cpu()
                )
            _, ind = torch.topk(similarity, 10, dim=0)
            running_label = label2

        preds = running_label[ind]
        recall += sum(
            [1 if label[i] in preds[:, i] else 0 for i in range(label.shape[0])]
        )
        print(f"Current Recall Score: {recall/len(test_dataset)}")
