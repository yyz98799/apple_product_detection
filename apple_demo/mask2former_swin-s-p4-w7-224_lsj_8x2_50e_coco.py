dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Resize',
        img_scale=(1024, 1024),
        ratio_range=(0.1, 2.0),
        multiscale_mode='range',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=(1024, 1024),
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(
        type='FilterAnnotations', min_gt_bbox_wh=(1e-05, 1e-05), by_mask=True),
    dict(
        type='Pad',
        size=(1024, 1024),
        pad_val=dict(img=(128, 128, 128), masks=0, seg=255)),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='DefaultFormatBundle', img_to_float=True),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Pad',
                size_divisor=32,
                pad_val=dict(img=(128, 128, 128), masks=0, seg=255)),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='CocoDataset',
        ann_file='data/coco/annotations/instances_train2017.json',
        img_prefix='data/coco/train2017/',
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Resize',
                img_scale=(1024, 1024),
                ratio_range=(0.1, 2.0),
                multiscale_mode='range',
                keep_ratio=True),
            dict(
                type='RandomCrop',
                crop_size=(1024, 1024),
                crop_type='absolute',
                recompute_bbox=True,
                allow_negative_crop=True),
            dict(
                type='FilterAnnotations',
                min_gt_bbox_wh=(1e-05, 1e-05),
                by_mask=True),
            dict(
                type='Pad',
                size=(1024, 1024),
                pad_val=dict(img=(128, 128, 128), masks=0, seg=255)),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle', img_to_float=True),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
        ]),
    val=dict(
        type='CocoDataset',
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Pad',
                        size_divisor=32,
                        pad_val=dict(img=(128, 128, 128), masks=0, seg=255)),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CocoDataset',
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Pad',
                        size_divisor=32,
                        pad_val=dict(img=(128, 128, 128), masks=0, seg=255)),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(
    interval=5000,
    metric=['bbox', 'segm'],
    dynamic_intervals=[(365001, 368750)])
checkpoint_config = dict(
    interval=5000, by_epoch=False, save_last=True, max_keep_ckpts=3)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook', by_epoch=False)
    ])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 5000)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=16)
num_things_classes = 80
num_stuff_classes = 0
num_classes = 80
model = dict(
    type='Mask2Former',
    backbone=dict(
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        frozen_stages=-1,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth'
        )),
    panoptic_head=dict(
        type='Mask2FormerHead',
        in_channels=[96, 192, 384, 768],
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_things_classes=80,
        num_stuff_classes=0,
        num_queries=100,
        num_transformer_feat_level=3,
        pixel_decoder=dict(
            type='MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=False,
                        norm_cfg=None,
                        init_cfg=None),
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True)),
                    operation_order=('self_attn', 'norm', 'ffn', 'norm')),
                init_cfg=None),
            positional_encoding=dict(
                type='SinePositionalEncoding', num_feats=128, normalize=True),
            init_cfg=None),
        enforce_decoder_input_project=False,
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=128, normalize=True),
        transformer_decoder=dict(
            type='DetrTransformerDecoder',
            return_intermediate=True,
            num_layers=9,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=False),
                ffn_cfgs=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True),
                feedforward_channels=2048,
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                 'ffn', 'norm')),
            init_cfg=None),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1
            ]),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0)),
    panoptic_fusion_head=dict(
        type='MaskFormerFusionHead',
        num_things_classes=80,
        num_stuff_classes=0,
        loss_panoptic=None,
        init_cfg=None),
    train_cfg=dict(
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        assigner=dict(
            type='MaskHungarianAssigner',
            cls_cost=dict(type='ClassificationCost', weight=2.0),
            mask_cost=dict(
                type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
            dice_cost=dict(
                type='DiceCost', weight=5.0, pred_act=True, eps=1.0)),
        sampler=dict(type='MaskPseudoSampler')),
    test_cfg=dict(
        panoptic_on=False,
        semantic_on=False,
        instance_on=True,
        max_per_image=100,
        iou_thr=0.8,
        filter_low_score=True),
    init_cfg=None)
image_size = (1024, 1024)
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.05,
    eps=1e-08,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        custom_keys=dict({
            'backbone':
            dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed':
            dict(lr_mult=1.0, decay_mult=0.0),
            'query_feat':
            dict(lr_mult=1.0, decay_mult=0.0),
            'level_embed':
            dict(lr_mult=1.0, decay_mult=0.0),
            'backbone.patch_embed.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'absolute_pos_embed':
            dict(lr_mult=0.1, decay_mult=0.0),
            'relative_position_bias_table':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.0.blocks.0.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.0.blocks.1.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.1.blocks.0.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.1.blocks.1.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.0.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.1.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.2.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.3.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.4.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.5.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.3.blocks.0.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.3.blocks.1.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.0.downsample.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.1.downsample.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.downsample.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.6.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.7.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.8.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.9.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.10.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.11.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.12.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.13.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.14.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.15.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.16.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.17.norm':
            dict(lr_mult=0.1, decay_mult=0.0)
        }),
        norm_decay_mult=0.0))
optimizer_config = dict(grad_clip=dict(max_norm=0.01, norm_type=2))
lr_config = dict(
    policy='step',
    gamma=0.1,
    by_epoch=False,
    step=[327778, 355092],
    warmup='linear',
    warmup_by_epoch=False,
    warmup_ratio=1.0,
    warmup_iters=10)
max_iters = 368750
runner = dict(type='IterBasedRunner', max_iters=368750)
interval = 5000
dynamic_intervals = [(365001, 368750)]
pad_cfg = dict(img=(128, 128, 128), masks=0, seg=255)
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth'
depths = [2, 2, 18, 2]
backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
backbone_embed_multi = dict(lr_mult=0.1, decay_mult=0.0)
custom_keys = dict({
    'backbone':
    dict(lr_mult=0.1, decay_mult=1.0),
    'backbone.patch_embed.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'absolute_pos_embed':
    dict(lr_mult=0.1, decay_mult=0.0),
    'relative_position_bias_table':
    dict(lr_mult=0.1, decay_mult=0.0),
    'query_embed':
    dict(lr_mult=1.0, decay_mult=0.0),
    'query_feat':
    dict(lr_mult=1.0, decay_mult=0.0),
    'level_embed':
    dict(lr_mult=1.0, decay_mult=0.0),
    'backbone.stages.0.blocks.0.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.0.blocks.1.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.1.blocks.0.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.1.blocks.1.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.0.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.1.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.2.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.3.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.4.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.5.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.3.blocks.0.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.3.blocks.1.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.0.downsample.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.1.downsample.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.downsample.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.6.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.7.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.8.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.9.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.10.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.11.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.12.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.13.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.14.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.15.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.16.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.17.norm':
    dict(lr_mult=0.1, decay_mult=0.0)
})
work_dir = 'mask2former_work_dirs'
auto_resume = False
gpu_ids = [0]
