import math

import torch
from torch import nn
from torch.nn import functional as F

from .models import (register_multimodal_meta_arch, make_multimodal_backbone, 
                    make_dependency_block)
from .multimodal_backbones import Alignment
from .blocks import MaskedConv1D, Scale, LayerNorm
from .losses import ctr_diou_loss_1d, sigmoid_focal_loss

from ..utils import batched_nms

import numpy as np

 

class NCE(nn.Module):
    def __init__(self):
        super(NCE, self).__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def forward(self, q, k, neg, device='cuda:0'):
        q = F.normalize(q, dim=1) #[1, C]
        k = F.normalize(k, dim=1) #[1, C]
        neg = F.normalize(neg, dim=1) #[T, C]
        l_pos = q @ k.T #[1, 1]
        l_neg = q @ neg.T #[1, T]
        logits = torch.cat([l_pos, l_neg], dim=1) #[1, 1 + T]
        logits *= self.logit_scale #[1, 1 + T]
        
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
        loss = F.cross_entropy(logits, labels)
        return loss

class Dual_Contrastive_Loss(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.logit_scale_inter = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.NCE_video = NCE()
        self.NCE_text = NCE()
        
    def forward(self, contrastive_pairs):
        if len(contrastive_pairs) == 0:
            return torch.zeros(1).cuda(), torch.zeros(1).cuda()
        cls_video = contrastive_pairs['cls_video']
        cls_text = contrastive_pairs['cls_text']
        key_video_list = contrastive_pairs['key_video_list']
        nonkey_video_list = contrastive_pairs['nonkey_video_list']
        key_text_list = contrastive_pairs['key_text_list']
        nonkey_text_list = contrastive_pairs['nonkey_text_list']
            
        B = cls_video.shape[0]
        device = cls_video.device
        
        ########## Inter-Sample Contrastive Loss ##########
        cls_video = F.normalize(cls_video.squeeze(1), dim=1) #[B, C]
        cls_text = F.normalize(cls_text.squeeze(1), dim=1) #[B, C]

        # cosine similarity as logits
        logits_per_video = self.logit_scale_inter.exp() * cls_video @ cls_text.t() #[B, B]
        logits_per_text = logits_per_video.t() #[B, B]
        
        target = torch.arange(B).to(device)
        inter_contrastive_loss_video = F.cross_entropy(logits_per_video, target)
        inter_contrastive_loss_text = F.cross_entropy(logits_per_text, target)
        inter_contrastive_loss = (inter_contrastive_loss_video + inter_contrastive_loss_text) / 2
        
        ########## Intra-Sample Contrastive Loss ##########
        intra_contrastive_loss = 0
        for i in range(B):
            intra_contrastive_loss_video = self.NCE_video(
                torch.mean(key_video_list[i], dim=0, keepdim=True),
                torch.mean(key_text_list[i], dim=0, keepdim=True),
                nonkey_video_list[i],
                device
            )
            intra_contrastive_loss_text = self.NCE_text(
                torch.mean(key_text_list[i], dim=0, keepdim=True),
                torch.mean(key_video_list[i], dim=0, keepdim=True),
                nonkey_text_list[i],
                device
            )
            intra_contrastive_loss += (intra_contrastive_loss_video + intra_contrastive_loss_text) / 2
        intra_contrastive_loss /= B
        
        return inter_contrastive_loss, intra_contrastive_loss
    

class PtTransformerClsHead(nn.Module):
    """
    1D Conv heads for classification
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        num_classes,
        prior_prob=0.01,
        num_layers=3,
        kernel_size=3,
        act_layer=nn.ReLU,
        with_ln=False,
        empty_cls = [],
    ):
        super().__init__()
        self.act = act_layer()

        # build the head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()

        for idx in range(num_layers-1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size//2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(
                    LayerNorm(out_dim)
                )
            else:
                self.norm.append(nn.Identity())

        # classifier
        self.cls_head = MaskedConv1D(
                feat_dim, num_classes, kernel_size,
                stride=1, padding=kernel_size//2
            )

        # use prior in model initialization to improve stability
        # this will overwrite other weight init
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.cls_head.conv.bias, bias_value)

        # a quick fix to empty categories:
        # the weights assocaited with these categories will remain unchanged
        # we set their bias to a large negative value to prevent their outputs
        if len(empty_cls) > 0:
            bias_value = -(math.log((1 - 1e-6) / 1e-6))
            for idx in empty_cls:
                torch.nn.init.constant_(self.cls_head.conv.bias[idx], bias_value)

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)
        # apply the classifier for each pyramid level
        out_logits = tuple()
        for _, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_logits, _ = self.cls_head(cur_out, cur_mask)
            out_logits += (cur_logits, )

        return out_logits


class PtTransformerRegHead(nn.Module):
    """
    Shared 1D Conv heads for regression
    Simlar logic as PtTransformerClsHead with separated implementation for clarity
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        num_classes,
        fpn_levels,
        num_layers=3,
        kernel_size=3,
        act_layer=nn.ReLU,
        with_ln=False,
        class_aware=False
    ):
        super().__init__()
        self.fpn_levels = fpn_levels
        self.act = act_layer()
        self.num_classes = num_classes

        # build the conv head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers-1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim

            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size//2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(
                    LayerNorm(out_dim)
                )
            else:
                self.norm.append(nn.Identity())

        self.scale = nn.ModuleList()
        for idx in range(fpn_levels):
            self.scale.append(Scale())

        if class_aware:
            self.offset_head = MaskedConv1D(
                    feat_dim, 2*num_classes, kernel_size,
                    stride=1, padding=kernel_size//2
                )
        else:
            self.offset_head = MaskedConv1D( 
                feat_dim, 2, kernel_size,
                stride=1, padding=kernel_size//2
            )


    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)
        assert len(fpn_feats) == self.fpn_levels

        # apply the classifier for each pyramid level
        out_offsets = tuple()
        for l, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_offsets, _ = self.offset_head(cur_out, cur_mask)
            out_offsets += (F.relu(self.scale[l](cur_offsets)), ) 

        return out_offsets


@register_multimodal_meta_arch("LocPointTransformer")
class PtTransformer(nn.Module):
    """
        Transformer based model for single stage action localization
    """
    def __init__(
        self,
        backbone_type,         # a string defines backbone type
        dependency_type,       # a string defines dependency block
        backbone_arch,         # a tuple defines # layers in embed / stem / branch
        scale_factor,          # scale factor between branch layers
        input_dim_V,           # input visual feat dim
        input_dim_A,           # input audio feat dim
        max_seq_len,           # max sequence length (used for training)
        n_head,                # number of heads for self-attention in transformer
        embd_kernel_size,      # kernel size of the embedding network
        embd_dim,              # output feat channel of the embedding network
        embd_with_ln,          # attach layernorm to embedding network
        head_dim,              # feature dim for head
        regression_range,      # regression range on each level of FPN
        head_num_layers,       # number of layers in the head (including the classifier)
        head_kernel_size,      # kernel size for reg/cls heads
        head_with_ln,          # attache layernorm to reg/cls heads
        use_abs_pe,            # if to use abs position encoding
        num_classes,           # number of action classes
        train_cfg,             # other cfg for training
        test_cfg,              # other cfg for testing
        class_aware,           # if to use class-aware regression
        use_dependency,        # if to use dependency block
        intra_contr_weight,    # intra-contrastive loss weight
        inter_contr_weight,    # inter-contrastive loss weight
        score_V_weight,        # video score loss weight
        score_A_weight,        # audio score loss weight
    ):
        super().__init__()
        self.fpn_strides = [scale_factor**i for i in range(backbone_arch[-1]+1)]
        self.reg_range = regression_range
        assert len(self.fpn_strides) == len(self.reg_range)
        self.scale_factor = scale_factor
        self.num_classes = num_classes
        self.class_aware = class_aware
        self.use_dependency = use_dependency
        # self.use_dependency = False

        self.max_seq_len = max_seq_len
        max_div_factor = 1
        for l, stride in enumerate(self.fpn_strides):
            assert max_seq_len % stride == 0, "max_seq_len must be divisible by fpn stride"
            if max_div_factor < stride:
                max_div_factor = stride
        self.max_div_factor = max_div_factor

        # training time config
        self.train_loss_weight = train_cfg['loss_weight']
        self.inter_contr_weight = inter_contr_weight#0.02
        self.intra_contr_weight = intra_contr_weight#0
        self.score_V_weight = score_V_weight#0.0001
        self.score_T_weight = score_A_weight#0.0001
        self.train_cls_prior_prob = train_cfg['cls_prior_prob']
        self.train_dropout = train_cfg['dropout']
        self.train_droppath = train_cfg['droppath']
        self.train_label_smoothing = train_cfg['label_smoothing']

        # test time config
        self.test_pre_nms_thresh = test_cfg['pre_nms_thresh']
        self.test_pre_nms_topk = test_cfg['pre_nms_topk']
        self.test_iou_threshold = test_cfg['iou_threshold']
        self.test_min_score = test_cfg['min_score']
        self.test_max_seg_num = test_cfg['max_seg_num']
        self.test_nms_method = test_cfg['nms_method']
        assert self.test_nms_method in ['soft', 'hard', 'none']
        self.test_duration_thresh = test_cfg['duration_thresh']
        self.test_multiclass_nms = test_cfg['multiclass_nms']
        self.test_nms_sigma = test_cfg['nms_sigma']
        self.test_voting_thresh = test_cfg['voting_thresh']

        # backbone network: conv + transformer
        assert backbone_type in ['convTransformer']
        self.backbone = make_multimodal_backbone(
            'convTransformer',
            **{
                'n_in_V' : input_dim_V,
                'n_in_A' : input_dim_A,
                'n_embd' : embd_dim,
                'n_head': n_head,
                'n_embd_ks': embd_kernel_size,
                'max_len': max_seq_len,
                'arch' : backbone_arch,
                'scale_factor' : scale_factor,
                'with_ln' : embd_with_ln,
                'attn_pdrop' : 0.0,
                'proj_pdrop' : self.train_dropout,
                'path_pdrop' : self.train_droppath,
                'use_abs_pe' : use_abs_pe,
            }
        )

        assert dependency_type in ['DependencyBlock']
        if self.use_dependency:
            self.dependency_block = make_dependency_block(
                'DependencyBlock',
                **{
                    'in_channel' : embd_dim*2,
                    'n_embd' : 128,  
                    'n_embd_ks' : embd_kernel_size,
                    'num_classes' : self.num_classes,
                    'path_pdrop' : self.train_droppath,
                }
            )
        
        # classfication and regerssion heads
        self.cls_head = PtTransformerClsHead(
            embd_dim*2,
            head_dim, self.num_classes,  
            kernel_size=head_kernel_size,
            prior_prob=self.train_cls_prior_prob,
            with_ln=head_with_ln,
            num_layers=head_num_layers,
            empty_cls=train_cfg['head_empty_cls']
        )
        self.reg_head = PtTransformerRegHead(
            embd_dim*2,
            head_dim, self.num_classes, 
            len(self.fpn_strides),
            kernel_size=head_kernel_size,
            num_layers=head_num_layers,
            with_ln=head_with_ln,
            class_aware=self.class_aware
        )

        # maintain an EMA of #foreground to stabilize the loss normalizer
        # useful for small mini-batch training
        self.loss_normalizer = train_cfg['init_loss_norm']
        self.loss_normalizer_momentum = 0.9

        ### m:
        # self.alignment_list = nn.ModuleList()
        # for l, level_dim in enumerate([224, 112, 56, 28, 14, 7]):
        #     self.alignment_list.append(
        #         Alignment(
        #             video_dim=512,
        #             audio_dim=512,
        #         )
        #     )
        self.alignment = Alignment(
                    video_dim=2048,#512
                    audio_dim=128,#512
                )
        
        self.contrastive_losses = Dual_Contrastive_Loss()

        ###
    @property
    def device(self):
        # a hacky way to get the device type
        # will throw an error if parameters are on different devices
        return list(set(p.device for name, p in self.named_parameters()))[0]
    
    ###m:
    @staticmethod
    def param_count(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    ###

    def forward(self, video_list):
        # batch the video list into feats (B, C, T) and masks (B, 1, T)
        # batched_inputs_V, batched_inputs_A, batched_masks, batched_scores, batched_start_end_idx, batched_m_labels = self.preprocessing(video_list) #m:[32, 2048, 224], [32, 128, 224], [32, 1, 224]
        batched_inputs_V = video_list['visual']
        batched_inputs_A = video_list['audio']
        batched_masks = video_list['mask']
        batched_scores = video_list['scores']
        batched_start_end_idx = video_list['start_end']
        batched_m_labels = video_list['m_labels']

      ######m: Implementing alignment guided fusion
        feats_V_aligned, feats_A_aligned, contrastive_pairs = self.alignment(
            video=[batched_inputs_V], 
            text=[batched_inputs_A], 
            mask_video=[batched_masks], 
            mask_text=[batched_masks],
            m_start_end = batched_start_end_idx,
            m_scores_gt = batched_scores,
            m_labels = batched_m_labels,
        )


        #########
        # forward the backbone
        # feats_V, feats_A, masks = self.backbone(batched_inputs_V, batched_inputs_A, batched_masks) 
        #######m:input: tensor:[B,2048,224],[B,128,224],[B,1,224] 
        # out feats_V(tuple): [B, 512, 224], ..., [B, 512, 7]; feats_A(tuple): [B, 512, 224], ..., [B, 512, 7]; masks: [B, 1, 224], ..., [B, 1, 7]
        
        
        ####m: alignment to yolo
        feats_V, feats_A, masks = self.backbone(feats_V_aligned[0], feats_A_aligned[0], batched_masks)

        ####
        # feats_V_aligned, feats_A_aligned = self.alignment(
        #     video=feats_V, 
        #     text=feats_A, 
        #     mask_video=masks, 
        #     mask_text=masks
        # )
        ####m: output: V:type:list, len=6, [B, 512, 224](tensor),...,[B,512,7] ---- A:type:list, len=6,...samw as V

        #concat audio and visual output features (B, C, T)->(B, 2C, T)
        # feats_AV = [torch.cat((V, A), 1) for _, (V, A) in enumerate(zip(feats_V_aligned, feats_A_aligned))]
        feats_AV = [torch.cat((V, A), 1) for _, (V, A) in enumerate(zip(feats_V, feats_A))]

        #######m: feats_AV: [8, 1024, 224], ..., [8, 1024, 7]

        # dependency block
        if self.use_dependency:
            feats_AV,  _ = self.dependency_block(feats_AV, masks)

        # out_cls: List[B, #cls, T_i]
        out_cls_logits = self.cls_head(feats_AV, masks) ###m: (6=levels, #cls, 224), ..., (6, #cls, 7)
        # out_offset: List[B, 2, T_i]/[B, 2*cls, T_i]
        out_offsets = self.reg_head(feats_AV, masks) ###m: (8, 200, 224), ..., (8, 200, 7)

        # permute the outputs
        # out_cls: F List[B, #cls, T_i] -> F List[B, T_i, #cls]
        out_cls_logits = [x.permute(0, 2, 1) for x in out_cls_logits]

        if self.class_aware:
            out_offsets = [x.permute(0, 2, 1) for x in out_offsets]
            out_offsets = [x.view(x.shape[0], x.shape[1], self.num_classes, -1).contiguous() for x in out_offsets]
        else:
            # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
            out_offsets = [x.permute(0, 2, 1) for x in out_offsets]
        # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
        fpn_masks = [x.squeeze(1) for x in masks]

        # generate segment/lable List[N x 2] / List[N] with length = B
        # assert video_list[0]['segments'] is not None, "GT action labels does not exist"
        # assert video_list[0]['labels'] is not None, "GT action labels does not exist"
        # gt_offsets = [x['gt_offsets'] for x in video_list]
        # gt_cls_labels = [x['gt_cls_labels'] for x in video_list]
        gt_offsets = video_list['gt_offsets']
        gt_cls_labels = video_list['gt_cls_labels']

        # compute the loss and return
        losses = self.losses(
            fpn_masks,
            out_cls_logits, out_offsets,
            gt_cls_labels, gt_offsets, 
            contrastive_pairs
        )

        # return loss during training
        if self.training:
            return losses

        else:

            # decode the actions (sigmoid / stride, etc)
            results = self.inference(
                video_list, fpn_masks,
                out_cls_logits, out_offsets
            )
            return results, losses

    @torch.no_grad()
    def preprocessing(self, video_list, padding_val=0.0):
        """
            Generate batched features and masks from a list of dict items
        """
        feats_visual = [x['feats']['visual'] for x in video_list]
        feats_audio = [x['feats']['audio'] for x in video_list]
        feats_lens = torch.as_tensor([feat_visual.shape[-1] for feat_visual in feats_visual])
        max_len = feats_lens.max(0).values.item() 

        ###m:
        # adding another key to the video_list dict which call 'scores'. 1 for features inside the segment and 0 for outside the segment
        for idx, video in enumerate(video_list):
            video_list[idx]['m_scores'] = torch.zeros(video['feats']['visual'].shape[-1])
            video_list[idx]['m_cls_labels_feats'] = torch.zeros(video['feats']['visual'].shape[-1], self.num_classes)
            video_list[idx]['m_start_end'] = []
            video_list[idx]['m_label'] = torch.zeros(video['feats']['visual'].shape[-1])
            for seg, label in zip(video['segments'], video['labels']):
                # each 1.28 seconds is one feature 
                # see the start and end time of the segment and convert it to the feature index
                start_idx = torch.div(seg[0],1.28).int()
                end_idx = torch.div(seg[1],1.28).int()
                video_list[idx]['m_start_end'].extend(list(range(start_idx, end_idx+1)))
                video_list[idx]['m_scores'][start_idx:end_idx] = 1
                video_list[idx]['m_cls_labels_feats'][start_idx:end_idx] = torch.nn.functional.one_hot(label, self.num_classes).float()
            video_list[idx]['m_start_end'] = list(set(video_list[idx]['m_start_end']))
            m_start_end = torch.zeros(video['feats']['visual'].shape[-1])
            m_start_end[video_list[idx]['m_start_end']] = 1
            video_list[idx]['m_start_end'] = m_start_end
        scores = [x['m_scores'] for x in video_list]
        start_end_idx = [x['m_start_end'] for x in video_list]
        m_labels = [x['m_cls_labels_feats'] for x in video_list]

        ###

        if self.training:
            assert max_len <= self.max_seq_len, "Input length must be smaller than max_seq_len during training"
            # set max_len to self.max_seq_len
            max_len = self.max_seq_len
        else:
            if max_len <= self.max_seq_len:
                max_len = self.max_seq_len
            else:
                # pad the input to the next divisible size
                stride = self.max_div_factor
                max_len = (max_len + (stride - 1)) // stride * stride

        # batch input shape B, C, T->visual
        batch_shape_visual = [len(feats_visual), feats_visual[0].shape[0], max_len]
        batched_inputs_visual = feats_visual[0].new_full(batch_shape_visual, padding_val)
        batched_scores = scores[0].new_full([len(scores), max_len], padding_val)
        batched_start_end_index = start_end_idx[0].new_full([len(start_end_idx), max_len], padding_val)
        batched_m_labels = m_labels[0].new_full([len(m_labels), max_len, self.num_classes], padding_val)
        for feat_visual, pad_feat_visual in zip(feats_visual, batched_inputs_visual):
            pad_feat_visual[..., :feat_visual.shape[-1]].copy_(feat_visual)
        for m_label, pad_m_label in zip(m_labels, batched_m_labels):
            pad_m_label[:m_label.shape[0]].copy_(m_label)

        ###m:
        for score, start_end, pad_score, pad_start_end in zip(scores, start_end_idx, batched_scores, batched_start_end_index):
            pad_score[:score.shape[-1]].copy_(score)
            pad_start_end[:start_end.shape[-1]].copy_(start_end)
        ###
        # audio 
        batch_shape_audio = [len(feats_audio), feats_audio[0].shape[0], max_len]
        batched_inputs_audio = feats_audio[0].new_full(batch_shape_audio, padding_val)
        for feat_audio, pad_feat_audio in zip(feats_audio, batched_inputs_audio):
            pad_feat_audio[..., :feat_audio.shape[-1]].copy_(feat_audio) 
   
        # generate the mask 
        batched_masks = torch.arange(max_len)[None, :] < feats_lens[:, None]
        # push to device
        device = feats_visual[0].device
        batched_inputs_visual = batched_inputs_visual.to(device)
        batched_inputs_audio = batched_inputs_audio.to(device)
        batched_scores = batched_scores.to(device)
        batched_start_end_index = batched_start_end_index.to(device)
        batched_m_labels = batched_m_labels.to(device)
        
        batched_masks = batched_masks.unsqueeze(1).to(device)

        return batched_inputs_visual, batched_inputs_audio, batched_masks, batched_scores, batched_start_end_index, batched_m_labels

    def losses(
        self, fpn_masks,
        out_cls_logits, out_offsets,
        gt_cls_labels, gt_offsets, 
        contrastive_pairs
    ):
        # fpn_masks, out_*: F (List) [B, T_i, C]
        # gt_* : B (list) [F T, C]
        # fpn_masks -> (B, FT)
        valid_mask = torch.cat(fpn_masks, dim=1)

        ###m:inter and inta contrastive loss
        inter_loss, intra_loss = self.contrastive_losses(contrastive_pairs)
        
        ###

        # 1. classification loss
        # stack the list -> (B, FT) -> (# Valid, )
        # gt_cls = torch.stack(gt_cls_labels)
        gt_cls = gt_cls_labels
        pos_mask = torch.logical_and((gt_cls.sum(-1) > 0), valid_mask)
        
        # cat the predicted offsets -> (B, FT, 2 (xC)) -> # (#Pos, 2 (xC))
        pred_offsets = torch.cat(out_offsets, dim=1)[pos_mask] 
        # gt_offsets = torch.stack(gt_offsets)[pos_mask]
        gt_offsets = gt_offsets[pos_mask] 

        # update the loss normalizer
        num_pos = pos_mask.sum().item()
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
            1 - self.loss_normalizer_momentum
        ) * max(num_pos, 1)

        # gt_cls is already one hot encoded now, simply masking out
        gt_target = gt_cls[valid_mask]

        # optinal label smoothing
        gt_target *= 1 - self.train_label_smoothing
        gt_target += self.train_label_smoothing / (self.num_classes + 1)

        # focal loss
        cls_loss = sigmoid_focal_loss(
            torch.cat(out_cls_logits, dim=1)[valid_mask],
            gt_target,
            reduction='sum'
        )
        cls_loss /= self.loss_normalizer

        # 2. regression using IoU/GIoU loss (defined on positive samples)
        if num_pos == 0:
            reg_loss = 0 * pred_offsets.sum()
        else:
            # giou loss defined on positive samples
            reg_loss = ctr_diou_loss_1d(
                pred_offsets,
                gt_offsets,
                reduction='sum',
                class_aware=self.class_aware
            )
            reg_loss /= self.loss_normalizer

        if self.train_loss_weight > 0:
            loss_weight = self.train_loss_weight
        else:
            loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)

        # return a dict of losses
        final_loss = cls_loss + reg_loss * loss_weight\
                    + inter_loss* self.inter_contr_weight + intra_loss* self.intra_contr_weight\
                        + contrastive_pairs['score_loss_video']* self.score_V_weight + contrastive_pairs['score_loss_text']* self.score_T_weight
        return {'cls_loss'   : cls_loss,
                'reg_loss'   : reg_loss * loss_weight,
                'final_loss' : final_loss, 
                'inter_contr_loss' : inter_loss * self.inter_contr_weight,
                'intra_contr_loss' : intra_loss * self.intra_contr_weight, 
                'score_loss_video' : contrastive_pairs['score_loss_video'] * self.score_V_weight,
                'score_loss_audio' : contrastive_pairs['score_loss_text'] * self.score_T_weight
                }

    @torch.no_grad()
    def inference(
        self,
        video_list,
        fpn_masks,
        out_cls_logits, out_offsets
    ):
        # video_list B (list) [dict]
        # points F (list) [T_i, 4]
        # fpn_masks, out_*: F (List) [B, T_i, C]
        results = []

        # 1: gather video meta information
        # vid_idxs = [x['video_id'] for x in video_list]
        # vid_fps = [x['fps'] for x in video_list]
        # vid_lens = [x['duration'] for x in video_list]
        # vid_ft_stride = [x['feat_stride'] for x in video_list]
        # vid_ft_nframes = [x['feat_num_frames'] for x in video_list]
        # vid_points = [x['points'] for x in video_list]
        vid_idxs = video_list['video_id']
        vid_fps = video_list['fps']
        vid_lens = video_list['duration']
        vid_ft_stride = video_list['feat_stride']
        vid_ft_nframes = video_list['feat_num_frames']
        vid_points = [
            [video_list['points'][j][i] for j in range(len(video_list['points']))]
            for i in range(video_list['points'][0].shape[0])
        ]

        # 2: inference on each single video and gather the results
        # upto this point, all results use timestamps defined on feature grids
        for idx, (vidx, fps, vlen, stride, nframes, points) in enumerate(
            zip(vid_idxs, vid_fps, vid_lens, vid_ft_stride, vid_ft_nframes, vid_points)
        ):
            # gather per-video outputs
            cls_logits_per_vid = [x[idx] for x in out_cls_logits]
            offsets_per_vid = [x[idx] for x in out_offsets]
            fpn_masks_per_vid = [x[idx] for x in fpn_masks]
            # inference on a single video (should always be the case)
            results_per_vid = self.inference_single_video(
                points, fpn_masks_per_vid,
                cls_logits_per_vid, offsets_per_vid
            )
            # pass through video meta info
            results_per_vid['video_id'] = vidx
            results_per_vid['fps'] = fps
            results_per_vid['duration'] = vlen
            results_per_vid['feat_stride'] = stride
            results_per_vid['feat_num_frames'] = nframes
            results.append(results_per_vid)

        # step 3: postprocssing
        results = self.postprocessing(results)

        return results

    @torch.no_grad()
    def inference_single_video(
        self,
        points,
        fpn_masks,
        out_cls_logits,
        out_offsets,
    ):
        # points F (list) [T_i, 4]
        # fpn_masks, out_*: F (List) [T_i, C]
        segs_all = []
        scores_all = []
        cls_idxs_all = []

        # loop over fpn levels
        for cls_i, offsets_i, pts_i, mask_i in zip(
                out_cls_logits, out_offsets, points, fpn_masks
            ):
            # sigmoid normalization for output logits
            pred_prob = (cls_i.sigmoid() * mask_i.unsqueeze(-1)).flatten()

            # Apply filtering to make NMS faster following detectron2
            # 1. Keep seg with confidence score > a threshold
            keep_idxs1 = (pred_prob > self.test_pre_nms_thresh)
            pred_prob = pred_prob[keep_idxs1] 
            topk_idxs = keep_idxs1.nonzero(as_tuple=True)[0] 

            # 2. Keep top k top scoring boxes only
            num_topk = min(self.test_pre_nms_topk, topk_idxs.size(0))
            pred_prob, idxs = pred_prob.sort(descending=True)
            pred_prob = pred_prob[:num_topk].clone()
            topk_idxs = topk_idxs[idxs[:num_topk]].clone()

            # fix a warning in pytorch 1.9
            pt_idxs =  torch.div(
                topk_idxs, self.num_classes, rounding_mode='floor'
            ) 
            cls_idxs = torch.fmod(topk_idxs, self.num_classes) 
            
            ##############################
            if self.class_aware:
                # 3. gather predicted offsets
                offsets_i = offsets_i.view(-1, offsets_i.shape[-1]).contiguous() 
                offsets = offsets_i[topk_idxs] 

            else:
                # 3. gather predicted offsets
                offsets = offsets_i[pt_idxs] 

            pts = pts_i[pt_idxs]

            # 4. compute predicted segments (denorm by stride for output offsets)
            seg_left = pts[:, 0] - offsets[:, 0] * pts[:, 3]
            seg_right = pts[:, 0] + offsets[:, 1] * pts[:, 3]
            pred_segs = torch.stack((seg_left, seg_right), -1) 

            # 5. Keep seg with duration > a threshold (relative to feature grids)
            seg_areas = seg_right - seg_left
            keep_idxs2 = seg_areas > self.test_duration_thresh

            # *_all : N (filtered # of segments) x 2 / 1
            segs_all.append(pred_segs[keep_idxs2])
            scores_all.append(pred_prob[keep_idxs2])
            cls_idxs_all.append(cls_idxs[keep_idxs2])

        # cat along the FPN levels (F N_i, C)
        segs_all, scores_all, cls_idxs_all = [
            torch.cat(x) for x in [segs_all, scores_all, cls_idxs_all]
        ]
        results = {'segments' : segs_all,
                   'scores'   : scores_all,
                   'labels'   : cls_idxs_all}

        return results

    @torch.no_grad()
    def postprocessing(self, results):
        # input : list of dictionary items
        # (1) push to CPU; (2) NMS; (3) convert to actual time stamps
        processed_results = []
        for results_per_vid in results:
            # unpack the meta info
            vidx = results_per_vid['video_id']
            fps = results_per_vid['fps']
            # vlen = results_per_vid['duration']
            vlen = results_per_vid['duration'] 
            stride = results_per_vid['feat_stride']
            nframes = results_per_vid['feat_num_frames']
            # 1: unpack the results and move to CPU
            # segs = results_per_vid['segments'].detach().cpu()
            # scores = results_per_vid['scores'].detach().cpu()
            # labels = results_per_vid['labels'].detach().cpu()
            segs = results_per_vid['segments']
            scores = results_per_vid['scores']
            labels = results_per_vid['labels']
            if self.test_nms_method != 'none':
                # 2: batched nms (only implemented on CPU)
                segs, scores, labels = batched_nms(
                    segs, scores, labels,
                    self.test_iou_threshold,
                    self.test_min_score,
                    self.test_max_seg_num,
                    use_soft_nms = (self.test_nms_method == 'soft'),
                    multiclass = self.test_multiclass_nms,
                    sigma = self.test_nms_sigma,
                    voting_thresh = self.test_voting_thresh
                )
            # 3: convert from feature grids to seconds
            if segs.shape[0] > 0:
                segs = (segs * stride + 0.5 * nframes) / fps
                # truncate all boundaries within [0, duration]
                segs[segs<=0.0] *= 0.0
                segs[segs>=vlen] = segs[segs>=vlen] * 0.0 + vlen
            # 4: repack the results
            device = results_per_vid['segments'].device
            processed_results.append(
                {
                    # 'video_id' : vidx,
                    'segments' : segs.unsqueeze(0).to(device),
                    'scores'   : scores.unsqueeze(0).to(device),
                    'labels'   : labels.unsqueeze(0).to(device),
                    }
            )

        # stack the results
        processed_results = {
            'segments' : torch.cat([x['segments'] for x in processed_results], dim=0),
            'scores'   : torch.cat([x['scores'] for x in processed_results], dim=0),
            'labels'   : torch.cat([x['labels'] for x in processed_results], dim=0),
        }

        return processed_results
