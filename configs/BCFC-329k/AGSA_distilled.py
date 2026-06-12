# Model settings
pretrained = 'https://download.openmmlab.com/mmclassification/v0/tinyvit/tinyvit-21m_in21k-distill-pre_3rdparty_in1k_20221021-3d9b30a2.pth'
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='TinyViT',
        arch='21m',
        img_size=(224, 224),
        window_size=[7, 7, 14, 7],
        out_indices=(3, ),
        drop_path_rate=0.2,
        gap_before_final_norm=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=pretrained,
            prefix='backbone.'),
        ),
    head=dict(
        type='AGSAdistilledClsHead',
        num_classes=47,
        in_channels=576,
        area_mean = 154543.77,
        area_std = 239624.67,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        loss_aux=dict(type='CrossEntropyLoss', loss_weight=0.7),
        loss_supcon=dict(type='SupConLoss', loss_weight=0.3, base_temperature=0.07, temperature=0.07),

        all_img_path = 'xxx/all_img_embeddings.pt',
        mapper_weight_path = 'xxx/MFA.pth',
        root_prefix_path = 'xxx/',

        init_cfg=[
            dict(
                type='TruncNormal',
                layer=['Conv2d', 'Linear'],
                std=.02,
                bias=0.),
            dict(type='Constant', layer=['LayerNorm'], val=1., bias=0.),
        ]
    )
    )

work_dir = 'work_dirs/BCFC-329k/AGSA_distilled'

bs = 64
num_gpu = 2
# bs = 128
# num_gpu = 1
# bs = 8 # debug
# num_gpu = 1

find_unused_parameters=False

# dataset settings
dataset_type = 'BCFC_329k'
data_preprocessor = dict(
    num_classes=47,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=256,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=bs,
    num_workers=5,
    dataset = dict(
        type=dataset_type,
        data_root='xxx/data/BCFC-329k',
        split='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True), # False is not allowed
)

val_dataloader = dict(
    batch_size=bs,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='xxx/data/BCFC-329k',
        split='val',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

evaluator = [dict(type='SingleLabelMetric', average='macro'), dict(type='SingleLabelMetric', average=None)]
val_evaluator = evaluator

# If you want standard test, please manually configure the test dataset
test_dataloader = dict(
    batch_size=bs,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='xxx/data/BCFC-329k',
        split='test',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
test_evaluator = val_evaluator


# for batch in each gpu is 128, 8 gpu
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=5e-4 * bs * num_gpu / 512,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        flat_decay_mult=0.0,
        custom_keys={
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0)
        }),
)

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=5e-5,
        by_epoch=True,
        end=7,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(type='CosineAnnealingLR', eta_min=1e-5, by_epoch=True, begin=7)
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=bs * num_gpu)


# defaults to use registries in mmpretrain
default_scope = 'mmpretrain'

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type='IterTimerHook'),

    # print log every 100 iterations.
    logger=dict(type='LoggerHook', interval=100),
    # logger=dict(type='LoggerHook', interval=1), # debug

    # enable the parameter scheduler.
    param_scheduler=dict(type='ParamSchedulerHook'),

    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=5, save_best='single-label/f1-score',rule='greater',),

    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type='DistSamplerSeedHook'),

    # validation results visualization, set True to enable it.
    visualization=dict(type='VisualizationHook', enable=False),
)

custom_imports = dict(
    imports=['hooks.iter_test_hook'],
    allow_failed_imports=False
)

custom_hooks = [
    dict(
        type='IterTestHook',
    )
]

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,

    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),

    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends)

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=3407, deterministic=False)