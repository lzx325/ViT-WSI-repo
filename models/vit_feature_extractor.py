from pprint import pprint

import torch
import torch.nn as nn

import timm.models.vision_transformer as vit
import timm.data.transforms_factory as transforms_factory
import timm.data

class ViTFeatureExtractor(nn.Module):
    def __init__(self,vit):
        super(ViTFeatureExtractor,self).__init__()
        self.vit=vit
    def forward(self,val):
        return self.vit.forward_features(val)

def vit_large_patch16_384():
    vit_large_patch16_384_config={
        'classifier': 'head',
        'crop_pct': 1.0,
        'first_conv': 'patch_embed.proj',
        'fixed_input_size': True,
        'input_size': (3, 384, 384),
        'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5),
        'num_classes': 1000,
        'pool_size': None,
        'std': (0.5, 0.5, 0.5),
        'url': 'https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz'
    }
    model=vit.vit_large_patch16_384(pretrained=True,default_cfg=vit_large_patch16_384_config)
    feature_extractor=ViTFeatureExtractor(model)
    data_config = timm.data.resolve_data_config({}, model=model)
    pprint(data_config)
    custom_transforms = transforms_factory.create_transform(**data_config)
    return feature_extractor,custom_transforms