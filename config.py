from easydict import EasyDict as edict

cfg = edict()

cfg.pretrain = edict()

cfg.pretrain.ground = edict()
cfg.pretrain.ground.img_size = (384, 384)  # (height, width)
cfg.pretrain.ground.hori_flip_prob = 0.5
cfg.pretrain.ground.vert_flip_prob = 0.5

cfg.pretrain.ground.model = edict()
cfg.pretrain.ground.model.in_chans = 3
cfg.pretrain.ground.model.patch_size = 32
cfg.pretrain.ground.model.embed_dim = 768
cfg.pretrain.ground.model.depth = 12
cfg.pretrain.ground.model.num_heads = 12
cfg.pretrain.ground.model.decoder_embed_dim = 512
cfg.pretrain.ground.model.decoder_depth = 8
cfg.pretrain.ground.model.decoder_num_heads = 16
cfg.pretrain.ground.model.mlp_ratio = 4.0


cfg.pretrain.overhead = edict()
cfg.pretrain.overhead.img_size = (224, 224)
cfg.pretrain.overhead.hori_flip_prob = 0.5


cfg.pretrain.train = edict()
cfg.pretrain.train.enabled = True
cfg.pretrain.train.mode = "full_metadata"  # one of 'no_metadata', 'full_metadata'
cfg.pretrain.train.batch_size = 77
cfg.pretrain.train.devices = 4
cfg.pretrain.train.num_workers = 12
cfg.pretrain.train.meta_dropout_prob = 0.25
cfg.pretrain.train.num_epochs = 100
cfg.pretrain.train.lr = 1e-4
cfg.pretrain.train.weight_decay = 1e-2
cfg.pretrain.train.accumulate_grad_batches = 1
cfg.pretrain.train.warmup_epochs = 40
cfg.pretrain.train.model_type = "CVEMAE"  # one of 'CVEMAE', 'CVMMAE', 'MAE', 'MOCOGEO'
cfg.pretrain.train.expt_name = "CVEMAE_v1"


cfg.finetune = edict()
cfg.finetune.ground = edict()
cfg.finetune.ground.img_size = (384, 384)  # (height, width)
cfg.finetune.ground.hori_flip_prob = 0.5
cfg.finetune.ground.vert_flip_prob = 0.5
cfg.finetune.ground.randaugment = 12
cfg.finetune.ground.augmix = 9
cfg.finetune.ground.cutmix = 1
cfg.finetune.ground.mixup = 0.8


cfg.finetune.overhead = edict()
cfg.finetune.overhead.img_size = (224, 224)
cfg.finetune.overhead.hori_flip_prob = 0.5


cfg.finetune.train = edict()
cfg.finetune.train.enabled = False
cfg.finetune.train.batch_size = 77
cfg.finetune.train.devices = 4
cfg.finetune.train.num_workers = 12
cfg.finetune.train.meta_dropout_prob = 0.25
cfg.finetune.train.num_epochs = 100
cfg.finetune.train.lr = 1e-4
cfg.finetune.train.weight_decay = 1e-1
cfg.finetune.train.accumulate_grad_batches = 1
cfg.finetune.train.warmup_epochs = 40
cfg.finetune.train.label_smoothing = 0.05
cfg.finetune.train.model_type = "CVEMAE"  # one of 'CVEMAE', 'CVMMAE', 'MAE', 'MOCOGEO'
cfg.finetune.train.expt_name = "CVEMAE_finetune_v1"
cfg.finetune.train.dataset = "CUB"  # one of 'iNAT', 'CUB', 'NABirds'
cfg.finetune.train.linear_probe = False
cfg.finetune.train.ckpt = "checkpoints/CVEMAE_v1-epoch=99-val_loss=0.00.ckpt"


cfg.retrieval = edict()
cfg.retrieval.enabled = False
cfg.retrieval.model_type = "CVEMAE"  # one of 'CVEMAE', 'CVMMAE' 'MOCOGEO'
cfg.retrieval.mode = "full_metadata"  # one of 'no_metadata', 'full_metadata'
cfg.retrieval.topk = 10
cfg.retrieval.hierarchical_filter = 50
cfg.retrieval.batch_size = cfg.retrieval.hierarchical_filter
cfg.retrieval.devices = 1
cfg.retrieval.num_workers = 12
cfg.retrieval.ckpt = "checkpoints/CVEMAE_v1-epoch=99-val_loss=0.00.ckpt"
cfg.retrieval.cve_ckpt = "checkpoints/CVEMAE_v1-epoch=99-val_loss=0.00.ckpt"
