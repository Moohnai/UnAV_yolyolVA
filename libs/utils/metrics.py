# Modified from official EPIC-Kitchens action detection evaluation code
# see https://github.com/epic-kitchens/C2-Action-Detection/blob/master/EvaluationCode/evaluate_detection_json_ek100.py
import os
import json
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from typing import List
from typing import Tuple
from typing import Dict 
import seaborn as sns


def remove_duplicate_annotations(ants, tol=1e-3):
    # remove duplicate annotations (same category and starting/ending time)
    valid_events = []
    for event in ants:
        s, e, l = event['segment'][0], event['segment'][1], event['label_id']
        valid = True
        for p_event in valid_events:
            if ((abs(s-p_event['segment'][0]) <= tol)
                and (abs(e-p_event['segment'][1]) <= tol)
                and (l == p_event['label_id'])
            ):
                valid = False
                break
        if valid:
            valid_events.append(event)
    return valid_events


def load_gt_seg_from_json(json_file, split=None, label='label_id', label_offset=0):
    # load json file
    with open(json_file, "r", encoding="utf8") as f:
        json_db = json.load(f)
    json_db = json_db['database']

    vids, starts, stops, labels = [], [], [], []
    for k, v in json_db.items():

        # filter based on split
        if (split is not None) and v['subset'].lower() != split:
            continue
        # remove duplicated instances
        ants = remove_duplicate_annotations(v['annotations'])
        # video id
        vids += [k] * len(ants)
        # for each event, grab the start/end time and label
        for event in ants:
            starts += [float(event['segment'][0])]
            stops += [float(event['segment'][1])]
            if isinstance(event[label], (Tuple, List)):
                # offset the labels by label_offset
                label_id = 0
                for i, x in enumerate(event[label][::-1]):
                    label_id += label_offset**i + int(x)
            else:
                # load label_id directly
                label_id = int(event[label])
            labels += [label_id]

    # move to pd dataframe
    gt_base = pd.DataFrame({
        'video-id' : vids,
        't-start' : starts,
        't-end': stops,
        'label': labels
    })

    return gt_base


def load_pred_seg_from_json(json_file, label='label_id', label_offset=0):
    # load json file
    with open(json_file, "r", encoding="utf8") as f:
        json_db = json.load(f)
    json_db = json_db['database']

    vids, starts, stops, labels, scores = [], [], [], [], []
    for k, v, in json_db.items():
        # video id
        vids += [k] * len(v)
        # for each event
        for event in v:
            starts += [float(event['segment'][0])]
            stops += [float(event['segment'][1])]
            if isinstance(event[label], (Tuple, List)):
                # offset the labels by label_offset
                label_id = 0
                for i, x in enumerate(event[label][::-1]):
                    label_id += label_offset**i + int(x)
            else:
                # load label_id directly
                label_id = int(event[label])
            labels += [label_id]
            scores += [float(event['scores'])]

    # move to pd dataframe
    pred_base = pd.DataFrame({
        'video-id' : vids,
        't-start' : starts,
        't-end': stops,
        'label': labels,
        'score': scores
    })

    return pred_base


class ANETdetection(object):
    """Adapted from https://github.com/activitynet/ActivityNet/blob/master/Evaluation/eval_detection.py"""

    def __init__(
        self,
        ant_file,
        split=None,
        model_name=None,
        tiou_thresholds=np.linspace(0.1, 0.5, 5),
        label='label_id',
        label_offset=0,
        num_workers=8,
        dataset_name=None,
    ):

        self.tiou_thresholds = tiou_thresholds
        self.ap = None
        self.num_workers = num_workers
        if dataset_name is not None:
            self.dataset_name = dataset_name
        else:
            self.dataset_name = os.path.basename(ant_file).replace('.json', '')

        # Import ground truth and predictions
        self.split = split
        self.ground_truth = load_gt_seg_from_json(
            ant_file, split=self.split, label=label, label_offset=label_offset)

        # remove labels that does not exists in gt
        self.activity_index = {j: i for i, j in enumerate(sorted(self.ground_truth['label'].unique()))}
        self.ground_truth['label']=self.ground_truth['label'].replace(self.activity_index)

    def _get_predictions_with_label(self, prediction_by_label, label_name, cidx):
        """Get all predicitons of the given label. Return empty DataFrame if there
        is no predcitions with the given label.
        """
        try:
            res = prediction_by_label.get_group(cidx).reset_index(drop=True)
            return res
        except:
            print('Warning: No predictions of label \'%s\' were provdied.' % label_name)
            return pd.DataFrame()

    def wrapper_compute_average_precision(self, preds):
        """Computes average precision for each class in the subset.
        """
        ap = np.zeros((len(self.tiou_thresholds), len(self.activity_index)))

        # Adaptation to query faster
        ground_truth_by_label = self.ground_truth.groupby('label')
        prediction_by_label = preds.groupby('label')

        results = Parallel(n_jobs=self.num_workers)(
            delayed(compute_average_precision_detection)(
                ground_truth=ground_truth_by_label.get_group(cidx).reset_index(drop=True),
                prediction=self._get_predictions_with_label(prediction_by_label, label_name, cidx),
                tiou_thresholds=self.tiou_thresholds,
                # ## M:
                # gt_cls=label_name,
                # pred_cls=cidx,
                # ## M
            ) for label_name, cidx in self.activity_index.items())

        # ## M:
        # self.activity_index = []
        # for i in sorted(self.ground_truth['label'].unique()):
        #     for j in sorted(self.ground_truth['label'].unique()):
        #         self.activity_index.append((i, j))
        # ## M

        # ## M:
        # results = Parallel(n_jobs=self.num_workers)(
        #     delayed(compute_average_precision_detection)(
        #         ground_truth=ground_truth_by_label.get_group(label_name).reset_index(drop=True),
        #         prediction=self._get_predictions_with_label(prediction_by_label, label_name, cidx),
        #         tiou_thresholds=self.tiou_thresholds,
        #         ## M:
        #         gt_cls=label_name,
        #         pred_cls=cidx,
        #         ## M
        #     ) for (label_name, cidx) in self.activity_index)
        ## M

        for i, cidx in enumerate(self.activity_index.values()):
                ap[:,cidx] = results[i]

        # ## M:
        # for (i, res) in enumerate(results):
        #     ## M:
        #     gt, prd = res['activity_index']
        #     ## M
        #     if gt == prd:
        #         ap[:,gt] = res['ap']

        # # for each threshold, print 5 highest and lowest mAP in each class
        # for i, tiou in enumerate(self.tiou_thresholds):
        #     print(f"tIOU: {tiou}")
        #     # sort the mAP
        #     sorted_ap = np.argsort(ap[i,:])
        #     # print the results
        #     print(f"Threshold: {tiou}, \
        #           top 5 classes with highest mAP: {sorted_ap[-5:]}, \
        #           top 5 highest mAP: {ap[i,sorted_ap[-5:]]}, \
        #           top 5 classes with lowest mAP: {sorted_ap[:5]}, \
        #           top 5 lowest mAP: {ap[i,sorted_ap[:5]]}")
        # ## M

        # ## M:
        # # create a confusion matrix
        # confusion_matrix = np.zeros((len(self.tiou_thresholds), len(self.ground_truth['label'].unique()), (len(self.ground_truth['label'].unique()))))  
        # # populate the confusion matrix
        # for (i, res) in enumerate(results):
        #     gt, prd = res['activity_index']
        #     confusion_matrix[:,gt,prd] = res['tp']

        # # plot confusion matrix
        # for i, tiou in enumerate(self.tiou_thresholds):
        #     #normalize the confusion matrix by row, confusion_matrix[i] is a 2D array
        #     for j in range(0,100):
        #         print(f"number of true pisitive and false positives with thereshold {tiou}, class{j}: {confusion_matrix[i][j]}")
        #     print("---------------------------------")
        #     confusion_matrix[i] = confusion_matrix[i] / confusion_matrix[i].sum(axis=1)[:, np.newaxis]
        #     plt.figure(figsize=(10, 10))
        #     ax = sns.heatmap(confusion_matrix[i], cmap="Blues", vmin=0, vmax=confusion_matrix[i].max(),
        #                 xticklabels=[str(i) for i in range(len(self.ground_truth['label'].unique()))],
        #                 yticklabels=[str(i) for i in range(len(self.ground_truth['label'].unique()))])
            
        
                        
        #     # Set the font size of x-tick and y-tick labels
        #     plt.xticks(fontsize=3)  # X-tick labels font size
        #     plt.yticks(fontsize=3)  # Y-tick labels font size
            

        #     # Optionally set labels
        #     # ax.set_xticklabels(['Label 1', 'Label 2', 'Label 3'])
        #     # ax.set_yticklabels(['Row 1', 'Row 2', 'Row 3'])
        #     plt.ylabel('Actual')
        #     plt.xlabel('Predicted')
        #     plt.title(f'Confusion Matrix tIOU: {tiou}_yolVA')
        #     plt.savefig(f'confusion_matrix_tIOU_{tiou}_yolyolVA.png', dpi=900)
        #     plt.close()


        return ap

    def evaluate(self, preds, verbose=True):
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        preds can be (1) a pd.DataFrame; or (2) a json file where the data will be loaded;
        or (3) a python dict item with numpy arrays as the values
        """

        if isinstance(preds, pd.DataFrame):
            assert 'label' in preds
        elif isinstance(preds, str) and os.path.isfile(preds):
            preds = load_pred_seg_from_json(preds)
        elif isinstance(preds, Dict):
            # move to pd dataframe
            # did not check dtype here, can accept both numpy / pytorch tensors
            preds = pd.DataFrame({
                'video-id' : preds['video-id'],
                't-start' : preds['t-start'].tolist(),
                't-end': preds['t-end'].tolist(),
                'label': preds['label'].tolist(),
                'score': preds['score'].tolist()
            })
        # always reset ap
        self.ap = None

        # make the label ids consistent
        preds['label'] = preds['label'].replace(self.activity_index)

        # compute mAP
        self.ap = self.wrapper_compute_average_precision(preds)
        mAP = self.ap.mean(axis=1)
        average_mAP = mAP.mean()

        # print results
        if verbose:
            # print the results
            print('[RESULTS] Action detection results on {:s}.'.format(
                self.dataset_name)
            )
            block = ''
            for tiou, tiou_mAP in zip(self.tiou_thresholds, mAP):
                block += '\n|tIoU = {:.2f}: mAP = {:.2f} (%)'.format(tiou, tiou_mAP*100)
            print(block)
            print('Avearge mAP: {:.2f} (%)'.format(average_mAP*100))

        # return the results
        return mAP, average_mAP


def compute_average_precision_detection(
    ground_truth,
    prediction,
    tiou_thresholds=np.linspace(0.1, 0.5, 5),
    # ## M:
    # gt_cls=0,
    # pred_cls=0,
    # ## M
):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.
    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.
    Outputs
    -------
    ap : float
        Average precision score.
    """
    ap = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        return ap

    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds),len(ground_truth))) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')

    # Assigning true positive to truly ground truth instances.
    for idx, this_pred in prediction.iterrows():

        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred['video-id'])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(this_pred[['t-start', 't-end']].values,
                               this_gt[['t-start', 't-end']].values)
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]['index']] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]['index']] = idx
                # ## M:
                # if tiou_thr > 0.5:
                #     corr_or_not = "correct" if gt_cls == pred_cls else "incorrect"
                #     print(f"Correct_or_not: {corr_or_not}, \
                #             video-id: {this_pred['video-id']}, \
                #             threshold: {tiou_thr}, \
                #             tIOU: {tiou_arr} \
                #             pred_cls: {pred_cls}, \
                #             gt_cls: {gt_cls}, \
                #             score: {this_pred['score']}, \
                #             pred t-start: {this_pred['t-start']}, \
                #             pred t-end: {this_pred['t-end']}, \
                #             gt t-start: {this_gt.loc[jdx]['t-start']}, \
                #             gt t-end: {this_gt.loc[jdx]['t-end']}, \
                #             gt_cls: {this_gt.loc[jdx]['label']}, \
                #         ")
                # ## M
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(np.single) #np.float
    fp_cumsum = np.cumsum(fp, axis=1).astype(np.single) #np.float
    recall_cumsum = tp_cumsum / npos

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = interpolated_prec_rec(precision_cumsum[tidx,:], recall_cumsum[tidx,:])

    return ap
    # ## M:
    # return {
    #     'ap': ap,
    #     'activity_index': (gt_cls, pred_cls),
    #     'tp': tp.sum(axis=1),
    # }
    # ## M


def segment_iou(target_segment, candidate_segments):
    """Compute the temporal intersection over union between a
    target segment and all the test segments.
    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.
    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
                     + (target_segment[1] - target_segment[0]) - segments_intersection
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / segments_union
    return tIoU


def interpolated_prec_rec(prec, rec):
    """Interpolated AP - VOCdevkit from VOC 2011.
    """
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap
