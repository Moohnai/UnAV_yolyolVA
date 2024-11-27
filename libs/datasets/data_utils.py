import os
import copy
import random
import numpy as np
import random
import torch


def trivial_batch_collator(batch):
    """
        A batch collator that does nothing
    """
    return batch

def worker_init_reset_seed(worker_id):
    """
        Reset random seed for each worker
    """
    seed = torch.initial_seed() % 2 ** 31
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def truncate_feats(
    data_dict,
    max_seq_len,
    trunc_thresh,
    crop_ratio=None,
    multi_modal = True,
    max_num_trials=200,
    has_action=True,
    no_trunc=False

):
    """
    Truncate feats and time stamps in a dict item

    data_dict = {'video_id'        : str
                 'feats'           : Tensor C x T
                 'segments'        : Tensor N x 2 (in feature grid)
                 'labels'          : Tensor N
                 'fps'             : float
                 'feat_stride'     : int
                 'feat_num_frames' : in

    """
    # get the meta info
    if multi_modal:
        feat_len = data_dict['feats']['visual'].shape[1]
    else:
        feat_len = data_dict['feats'].shape[1]

    num_segs = data_dict['segments'].shape[0]

    # seq_len < max_seq_len
    if feat_len <= max_seq_len:
        # do nothing
        if crop_ratio == None:
            return data_dict
        # randomly crop the seq by setting max_seq_len to a value in [l, r]
        else:
            max_seq_len = random.randint(
                max(round(crop_ratio[0] * feat_len), 1),
                min(round(crop_ratio[1] * feat_len), feat_len),
            )
            # # corner case
            if feat_len == max_seq_len:
                return data_dict

    # otherwise, deep copy the dict
    data_dict = copy.deepcopy(data_dict)

    # try a few times till a valid truncation with at least one action
    for _ in range(max_num_trials):

        # sample a random truncation of the video feats
        st = random.randint(0, feat_len - max_seq_len)
        ed = st + max_seq_len
        window = torch.as_tensor([st, ed], dtype=torch.float32)

        # compute the intersection between the sampled window and all segments
        window = window[None].repeat(num_segs, 1)
        left = torch.maximum(window[:, 0], data_dict['segments'][:, 0])
        right = torch.minimum(window[:, 1], data_dict['segments'][:, 1])
        inter = (right - left).clamp(min=0)
        area_segs = torch.abs(
            data_dict['segments'][:, 1] - data_dict['segments'][:, 0])
        inter_ratio = inter / area_segs

        # only select those segments over the thresh
        seg_idx = (inter_ratio >= trunc_thresh)

        if no_trunc:
            # with at least one action and not truncating any actions
            seg_trunc_idx = torch.logical_and(
                (inter_ratio > 0.0), (inter_ratio < 1.0)
            )
            if (seg_idx.sum().item() > 0) and (seg_trunc_idx.sum().item() == 0):
                break
        elif has_action:
            # with at least one action
            if seg_idx.sum().item() > 0:
                break
        else:
            # without any constraints
            break
    
    if multi_modal:
        data_dict['feats']['visual'] = data_dict['feats']['visual'][:, st:ed].clone()
        data_dict['feats']['audio'] = data_dict['feats']['audio'][:, st:ed].clone()
    else:
        # feats: C x T
        data_dict['feats'] = data_dict['feats'][:, st:ed].clone()
    # segments: N x 2 in feature grids
    data_dict['segments'] = torch.stack((left[seg_idx], right[seg_idx]), dim=1)
    # shift the time stamps due to truncation
    data_dict['segments'] = data_dict['segments'] - st
    # labels: N
    data_dict['labels'] = data_dict['labels'][seg_idx].clone()

    return data_dict

def collate_fcn(video_list, num_classes, max_seq_len, padding_val=0, training=True, max_div_factor=1):
    """
    Collate function for dataloader
    """
    # deep copy the batch
    feats_visual = [x['feats']['visual'] for x in video_list]
    feats_audio = [x['feats']['audio'] for x in video_list]
    gt_offsets = [x['gt_offsets'] for x in video_list]
    gt_cls_labels = [x['gt_cls_labels'] for x in video_list]
    video_id = [x['video_id'] for x in video_list]
    fps = [x['fps'] for x in video_list]
    duration = [x['duration'] for x in video_list]
    feat_stride = [x['feat_stride'] for x in video_list]
    feat_num_frames = [x['feat_num_frames'] for x in video_list]
    points = [x['points'] for x in video_list]
    feats_lens = torch.as_tensor([feat_visual.shape[-1] for feat_visual in feats_visual])
    max_len = feats_lens.max(0).values.item() 

    ###m:
    # adding another key to the video_list dict which call 'scores'. 1 for features inside the segment and 0 for outside the segment
    for idx, video in enumerate(video_list):
        video_list[idx]['m_scores'] = torch.zeros(video['feats']['visual'].shape[-1])
        video_list[idx]['m_cls_labels_feats'] = torch.zeros(video['feats']['visual'].shape[-1], num_classes)
        video_list[idx]['m_start_end'] = []
        video_list[idx]['m_label'] = torch.zeros(video['feats']['visual'].shape[-1])
        for seg, label in zip(video['segments'], video['labels']):
            # each 1.28 seconds is one feature 
            # see the start and end time of the segment and convert it to the feature index
            start_idx = torch.div(seg[0],1.28).int()
            end_idx = torch.div(seg[1],1.28).int()
            video_list[idx]['m_start_end'].extend(list(range(start_idx, end_idx+1)))
            video_list[idx]['m_scores'][start_idx:end_idx] = 1
            video_list[idx]['m_cls_labels_feats'][start_idx:end_idx] = torch.nn.functional.one_hot(label, num_classes).float()
        video_list[idx]['m_start_end'] = list(set(video_list[idx]['m_start_end']))
        m_start_end = torch.zeros(video['feats']['visual'].shape[-1])
        m_start_end[video_list[idx]['m_start_end']] = 1
        video_list[idx]['m_start_end'] = m_start_end
    scores = [x['m_scores'] for x in video_list]
    start_end_idx = [x['m_start_end'] for x in video_list]
    m_labels = [x['m_cls_labels_feats'] for x in video_list]

    ###

    if training:
        assert max_len <= max_seq_len, "Input length must be smaller than max_seq_len during training"
        # set max_len to self.max_seq_len
        max_len = max_seq_len
    else:
        if max_len <= max_seq_len:
            max_len = max_seq_len
        else:
            # pad the input to the next divisible size
            stride = max_div_factor
            max_len = (max_len + (stride - 1)) // stride * stride

    # batch input shape B, C, T->visual
    batch_shape_visual = [len(feats_visual), feats_visual[0].shape[0], max_len]
    batched_inputs_visual = feats_visual[0].new_full(batch_shape_visual, padding_val)
    batched_scores = scores[0].new_full([len(scores), max_len], padding_val)
    batched_start_end_index = start_end_idx[0].new_full([len(start_end_idx), max_len], padding_val)
    batched_m_labels = m_labels[0].new_full([len(m_labels), max_len, num_classes], padding_val)
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
    
    batched_masks = batched_masks.unsqueeze(1)
    batched_gts = torch.stack(gt_offsets, dim=0)
    batched_cls_labels = torch.stack(gt_cls_labels, dim=0)
    batched_video_id = video_id
    batched_fps = fps
    batched_duration = duration
    batched_feat_stride = feat_stride
    batched_feat_num_frames = feat_num_frames
    batched_points = [torch.stack([points[i][j] for i in range(len(points))], dim=0) for j in range(len(points[0]))]

    # return batched_inputs_visual, batched_inputs_audio, batched_masks, batched_scores, batched_start_end_index, batched_m_labels
    return {
        'visual': batched_inputs_visual,
        'audio': batched_inputs_audio,
        'mask': batched_masks,
        'scores': batched_scores,
        'start_end': batched_start_end_index,
        'm_labels': batched_m_labels,
        'gt_offsets': batched_gts,
        'gt_cls_labels': batched_cls_labels,
        'video_id': batched_video_id,
        'fps': batched_fps,
        'duration': batched_duration,
        'feat_stride': batched_feat_stride,
        'feat_num_frames': batched_feat_num_frames,
        'points': batched_points,
    }
