import math
import pdb
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lseg_blocks import FeatureFusionBlock, Interpolate, _make_encoder, FeatureFusionBlock_custom, forward_vit
import clip
import numpy as np
import pandas as pd
import os
from transformers import Dinov2Model, Dinov2PreTrainedModel, DPTModel, Dinov2Config, DPTConfig, DPTForDepthEstimation
from transformers.modeling_outputs import SemanticSegmenterOutput
from transformers import  AutoProcessor

class depthwise_clipseg_conv(nn.Module):
    def __init__(self):
        super(depthwise_clipseg_conv, self).__init__()
        self.depthwise = nn.Conv2d(1, 1, kernel_size=3, padding=1)
    
    def depthwise_clipseg(self, x, channels):
        x = torch.cat([self.depthwise(x[:, i].unsqueeze(1)) for i in range(channels)], dim=1)
        return x

    def forward(self, x):
        channels = x.shape[1]
        out = self.depthwise_clipseg(x, channels)
        return out


class depthwise_conv(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1):
        super(depthwise_conv, self).__init__()
        self.depthwise = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        # support for 4D tensor with NCHW
        C, H, W = x.shape[1:]
        x = x.reshape(-1, 1, H, W)
        x = self.depthwise(x)
        x = x.view(-1, C, H, W)
        return x


class depthwise_block(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, activation='relu'):
        super(depthwise_block, self).__init__()
        self.depthwise = depthwise_conv(kernel_size=3, stride=1, padding=1)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()

    def forward(self, x, act=True):
        x = self.depthwise(x)
        if act:
            x = self.activation(x)
        return x


class bottleneck_block(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, activation='relu'):
        super(bottleneck_block, self).__init__()
        self.depthwise = depthwise_conv(kernel_size=3, stride=1, padding=1)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()


    def forward(self, x, act=True):
        sum_layer = x.max(dim=1, keepdim=True)[0]
        x = self.depthwise(x)
        x = x + sum_layer
        if act:
            x = self.activation(x)
        return x

class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.
        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location=torch.device("cpu"))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)

def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        activation=nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )

class LSeg(BaseModel):
    def __init__(
        self,
        head,
        features=256,
        backbone="clip_vitl16_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        **kwargs,
    ):
        super(LSeg, self).__init__()

        self.channels_last = channels_last

        hooks = {
            "clip_vitl16_384": [5, 11, 17, 23],
            "clipRN50x16_vitl16_384": [5, 11, 17, 23],
            "clip_vitb32_384": [2, 5, 8, 11],
        }

        # Instantiate backbone and reassemble blocks
        self.clip_pretrained, self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
        )

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
        if backbone in ["clipRN50x16_vitl16_384"]:
            self.out_c = 768
        else:
            self.out_c = 512
        self.scratch.head1 = nn.Conv2d(features, self.out_c, kernel_size=1)

        self.arch_option = kwargs["arch_option"]
        if self.arch_option == 1:
            self.scratch.head_block = bottleneck_block(activation=kwargs["activation"])
            self.block_depth = kwargs['block_depth']
        elif self.arch_option == 2:
            self.scratch.head_block = depthwise_block(activation=kwargs["activation"])
            self.block_depth = kwargs['block_depth']

        self.scratch.output_conv = head

        self.text = clip.tokenize(self.labels)    
        
    def forward(self, x, labelset=''):
        if labelset == '':
            text = self.text
        else:
            text = clip.tokenize(labelset)    
        
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        text = text.to(x.device)
        self.logit_scale = self.logit_scale.to(x.device)
        text_features = self.clip_pretrained.encode_text(text)

        image_features = self.scratch.head1(path_1)

        imshape = image_features.shape
        image_features = image_features.permute(0,2,3,1).reshape(-1, self.out_c)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        pixel_encoding = self.logit_scale * image_features.half() 
        
        logits_per_image = pixel_encoding @ text_features.t()


        out = logits_per_image.float().view(imshape[0], imshape[2], imshape[3], -1).permute(0,3,1,2)

        if self.arch_option in [1, 2]:
            for _ in range(self.block_depth - 1):
                out = self.scratch.head_block(out)
            out = self.scratch.head_block(out, False)

        out = self.scratch.output_conv(out)
            
        return out


class LSegNet(LSeg):
    """Network for semantic segmentation."""
    def __init__(self, labels, path=None, scale_factor=0.5, crop_size=480, **kwargs):

        features = kwargs["features"] if "features" in kwargs else 256
        kwargs["use_bn"] = True

        self.crop_size = crop_size
        self.scale_factor = scale_factor
        self.labels = labels

        head = nn.Sequential(
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )

        super().__init__(head, **kwargs)

        if path is not None:
            self.load(path)

class LSegEnc(BaseModel):
    def __init__(
        self,
        head,
        features=256,
        backbone="siglip_vitl16_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        **kwargs,
    ):
        super(LSegEnc, self).__init__()
        #TODO 开始更改我的VLMAPS的视觉backbone的模型
        self.not_changed = kwargs['not_changed']
        self.channels_last = channels_last
        self.backbone = backbone
        
        if self.not_changed:
            hooks = {
                "clip_vitl16_384": [5, 11, 17, 23],
                "clipRN50x16_vitl16_384": [5, 11, 17, 23],
                "clip_vitb32_384": [2, 5, 8, 11],
            }

            # Instantiate backbone and reassemble blocks
            self.clip_pretrained, self.pretrained, self.scratch = _make_encoder(
                backbone,
                features,
                groups=1,
                expand=False,
                exportable=False,
                hooks=hooks[backbone],
                use_readout=readout,
            )

            self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
            self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
            self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
            self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
            if backbone in ["clipRN50x16_vitl16_384"]:
                self.out_c = 768
            else:
                self.out_c = 512
            self.scratch.head1 = nn.Conv2d(features, self.out_c, kernel_size=1)

            self.arch_option = kwargs["arch_option"]
            if self.arch_option == 1:
                self.scratch.head_block = bottleneck_block(activation=kwargs["activation"])
                self.block_depth = kwargs['block_depth']
            elif self.arch_option == 2:
                self.scratch.head_block = depthwise_block(activation=kwargs["activation"])
                self.block_depth = kwargs['block_depth']

            self.scratch.output_conv = head

            self.text = clip.tokenize(self.labels)
            #Dinov2 config
            backbone_config = Dinov2Config.from_pretrained("facebook/dinov2-large", out_features=["stage5", "stage11", "stage17", "stage23"], reshape_hidden_states=False)
            config = DPTConfig(backbone_config=backbone_config, image_size=480)
            self.dinov2  = DPTForDepthEstimation(config=config)
            self.fusion_conv = nn.Conv2d(in_channels=768, out_channels=512, kernel_size=1)

        else:
            hooks = {
                "clip_vitl16_384": [5, 11, 17, 23],
                "clipRN50x16_vitl16_384": [5, 11, 17, 23],
                "clip_vitb32_384": [2, 5, 8, 11],
                "siglip_vitl16_384":[5, 11, 17, 23]
            }

            # # Instantiate backbone and reassemble blocks
            self.clip_pretrained, self.pretrained, self.scratch = _make_encoder(
                backbone,
                features,
                groups=1,
                expand=False,
                exportable=False,
                hooks=hooks[backbone],
                use_readout=readout,
            )
            
            self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
            self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
            self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
            self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
            if backbone in ["clipRN50x16_vitl16_384"]:
                self.out_c = 768
            elif backbone in ["siglip_vitl16_384"]:
                self.out_c = 1024
            else:
                self.out_c = 512
            self.scratch.head1 = nn.Conv2d(features, self.out_c, kernel_size=1)

            self.arch_option = kwargs["arch_option"]
            if self.arch_option == 1:
                self.scratch.head_block = bottleneck_block(activation=kwargs["activation"])
                self.block_depth = kwargs['block_depth']
            elif self.arch_option == 2:
                self.scratch.head_block = depthwise_block(activation=kwargs["activation"])
                self.block_depth = kwargs['block_depth']

            self.scratch.output_conv = head

            if self.backbone == "siglip_vitl16_384":
                processor = AutoProcessor.from_pretrained("google/siglip-large-patch16-384")
                tokenizer = processor.tokenizer
                self.tokenizer = tokenizer
                self.text = tokenizer(self.labels, padding="max_length")
                self.image_processor = processor.image_processor
    
            else:
                self.text = clip.tokenize(self.labels)  
        
    def forward(self, x, labelset):
        if labelset == '':
            text = self.text
        elif self.backbone == "siglip_vitl16_384":
            text = self.tokenizer(labelset, padding="max_length")
        else:
            text = clip.tokenize(labelset) 

        if not torch.is_tensor(text): 
            text = torch.tensor(text.input_ids)

        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        if self.not_changed:
            layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

            layer_1_rn = self.scratch.layer1_rn(layer_1)
            layer_2_rn = self.scratch.layer2_rn(layer_2)
            layer_3_rn = self.scratch.layer3_rn(layer_3)
            layer_4_rn = self.scratch.layer4_rn(layer_4)

            path_4 = self.scratch.refinenet4(layer_4_rn)
            path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
            path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
            path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

            text = text.to(x.device)
            self.logit_scale = self.logit_scale.to(x.device)
            text_features = self.clip_pretrained.encode_text(text)

            image_features = self.scratch.head1(path_1)
            origin_features = image_features
            #TODO 进行特征的融合试试forward
            patch_embeddings = self.dinov2(x, output_hidden_states=True, return_dict=True)
            dinov2_hidden_states = patch_embeddings.hidden_states
            last_fused_feature = dinov2_hidden_states[-1]
            # last_fused_feature = self.up_d(last_fused_feature) # 1 256 272 272->1 512 272 272
            last_fused_feature =  F.interpolate(last_fused_feature, size=(240, 240), mode='bilinear', align_corners=False)
            combined_features = torch.cat((last_fused_feature, origin_features), dim=1)
            fusion_features = self.fusion_conv(combined_features)
            image_features = fusion_features 

            imshape = image_features.shape
            image_features = image_features.permute(0,2,3,1).reshape(-1, self.out_c)

            # normalized features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            pixel_encoding = self.logit_scale * image_features.half() 
            
            logits_per_image = pixel_encoding @ text_features.t()
            pixel_encoding = pixel_encoding.float().view(imshape[0], imshape[2], imshape[3], -1).permute(0,3,1,2)

            out = logits_per_image.float().view(imshape[0], imshape[2], imshape[3], -1).permute(0,3,1,2)

            # if self.arch_option in [1, 2]:
            #     for _ in range(self.block_depth - 1):
            #         out = self.scratch.head_block(out)
            #     out = self.scratch.head_block(out, False)

            pixel_encoding = self.scratch.output_conv(pixel_encoding)
            out = self.scratch.output_conv(out)

        else:
            layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x, self.backbone)

            layer_1_rn = self.scratch.layer1_rn(layer_1)
            layer_2_rn = self.scratch.layer2_rn(layer_2)
            layer_3_rn = self.scratch.layer3_rn(layer_3)
            layer_4_rn = self.scratch.layer4_rn(layer_4)

            path_4 = self.scratch.refinenet4(layer_4_rn)
            path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
            path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
            path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

            text = text.to(x.device)

            self.logit_scale = self.logit_scale.to(x.device)


            if self.backbone == "siglip_vitl16_384":
                text_features = self.clip_pretrained(text).pooler_output
            else:
                text_features = self.clip_pretrained.encode_text(text)

            image_features = self.scratch.head1(path_1)

            imshape = image_features.shape
            image_features = image_features.permute(0,2,3,1).reshape(-1, self.out_c)

            # normalized features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            pixel_encoding = self.logit_scale * image_features.half() 
            
            if image_features.dtype == text_features.t().dtype:
                logits_per_image = self.logit_scale * image_features @ text_features.t()
            else:
                logits_per_image = self.logit_scale * image_features.half() @ text_features.t()

            
            pixel_encoding = pixel_encoding.float().view(imshape[0], imshape[2], imshape[3], -1).permute(0,3,1,2)

            out = logits_per_image.float().view(imshape[0], imshape[2], imshape[3], -1).permute(0,3,1,2)

            # pdb.set_trace()
            pixel_encoding = self.scratch.output_conv(pixel_encoding)
            out = self.scratch.output_conv(out)

            
        return pixel_encoding, out


class LSegEncNet(LSegEnc):
    """Network for semantic segmentation."""
    def __init__(self, labels, path=None, scale_factor=0.5, crop_size=480, **kwargs):

        features = kwargs["features"] if "features" in kwargs else 256
        kwargs["use_bn"] = True
        # kwargs["not_changed"] = True  
        self.crop_size = crop_size
        self.scale_factor = scale_factor
        self.labels = labels

        head = nn.Sequential(
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )

        super().__init__(head, **kwargs)

        if path is not None:
            self.load(path)