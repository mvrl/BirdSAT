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
cfg.pretrain.train.mode = "full_metadata"  # one of 'no_metadata', 'full_metadata'
cfg.pretrain.train.batch_size = 64
cfg.pretrain.train.devices = 4
cfg.pretrain.train.num_workers = 12
cfg.pretrain.train.meta_dropout_prob = 0.25
cfg.pretrain.train.num_epochs = 100
cfg.pretrain.train.lr = 1e-4
cfg.pretrain.train.weight_decay = 1e-2
cfg.pretrain.train.accumulate_grad_batches = 1
cfg.pretrain.train.warmup_epochs = 40
cfg.pretrain.train.model_type = "CVEMAE"  # one of 'CVEMAE', 'CVMMAE', 'MAE'
cfg.pretrain.train.expt_name = "CVEMAE_v1"
