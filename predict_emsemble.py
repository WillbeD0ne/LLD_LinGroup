'''
generate prediction on unlabeled data
'''
import argparse
import os
import json
import csv
import glob
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from contextlib import suppress
from torch.utils.data.dataloader import DataLoader
from timm.models import create_model, apply_test_time_pool, load_checkpoint, is_model, list_models
from timm.utils import setup_default_logging, set_jit_legacy

import models
from metrics import *
from validate_score import compute_metrics, compute_multimodel_metrics
from datasets.mp_liver_dataset import MultiPhaseLiverDataset, MultiPhaseLiverDataset_withglobal, collate_fn_eval_train

from models import l2_penalty, Merge_uniformer, Merge_uniformer_Base, Merge_uniformer_twoloss

has_apex = False
try:
    from apex import amp
    has_apex = True
except ImportError:
    pass

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('validate')


parser = argparse.ArgumentParser(description='Prediction on unlabeled data')

parser.add_argument('--img_size', default=(16, 128, 128),
                    type=int, nargs='+', help='input image size.')
parser.add_argument('--crop_size', default=(14, 112, 112),
                    type=int, nargs='+', help='cropped image size.')
parser.add_argument('--val_data_dir', default='./images/', type=str)
parser.add_argument('--val_anno_file', default='./labels/test.txt', type=str)
parser.add_argument('--val_transform_list',
                    default=['center_crop'], nargs='+', type=str)
parser.add_argument('--model', '-m', metavar='NAME', default='resnet50',
                    help='model architecture (default: dpn92)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--num-classes', type=int, default=7,
                    help='Number classes in dataset')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--test-pool', dest='test_pool', action='store_true',
                    help='enable test time pool')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--amp', action='store_true', default=False,
                    help='Use AMP mixed precision. Defaults to Apex, fallback to native Torch AMP.')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--tf-preprocessing', action='store_true', default=False,
                    help='Use Tensorflow preprocessing pipeline (require CPU TF installed')
parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                    help='use ema version of weights if present')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--legacy-jit', dest='legacy_jit', action='store_true',
                    help='use legacy jit mode for pytorch 1.5/1.5.1/1.6 to get back fusion performance')
parser.add_argument('--results-dir', default='', type=str, metavar='FILENAME',
                    help='Output csv file for validation results (summary)')
parser.add_argument('--team_name', default='', type=str,
                    required=True, help='Please enter your team name')


def validate(args):
    # might as well try to validate something
    args.pretrained = args.pretrained or not args.checkpoint
    # amp_autocast = suppress  # do nothing
    if args.amp:
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
        else:
            _logger.warning("Neither APEX or Native Torch AMP is available.")
    assert not args.apex_amp or not args.native_amp, "Only one AMP mode should be set."
    # if args.native_amp:
    #     amp_autocast = torch.cuda.amp.autocast
    #     _logger.info('Validating in mixed precision with native PyTorch AMP.')
    # elif args.apex_amp:
    #     _logger.info('Validating in mixed precision with NVIDIA APEX AMP.')
    # else:
    #     _logger.info('Validating in float32. AMP not enabled.')

    if args.legacy_jit:
        set_jit_legacy()

    model_list = []
    if args.model == 'ensemble':  
        is_cls2 = False   
        checkpoints = ["/home/mdisk2/linzihao/medical_Imaging/LLD-MMRI2023/main/output/unpretrained_sample_aug/7clsrnd_2_base_ouths/fold1/checkpoint-104.pth.tar",
                        "/home/mdisk2/linzihao/medical_Imaging/LLD-MMRI2023/main/output/unpretrained_sample_aug/7clsrnd_2_base_ouths/fold2/checkpoint-119.pth.tar",
                        "/home/mdisk2/linzihao/medical_Imaging/LLD-MMRI2023/main/output/unpretrained_sample_aug/7clsrnd_2_base_ouths/fold3/checkpoint-108.pth.tar",
                        "/home/mdisk2/linzihao/medical_Imaging/LLD-MMRI2023/main/output/unpretrained_sample_aug/7clsrnd_2_base_ouths/fold4/checkpoint-138.pth.tar",
                        "/home/mdisk2/linzihao/medical_Imaging/LLD-MMRI2023/main/output/unpretrained_sample_aug/7clsrnd_2_base_ouths/fold5/checkpoint-100.pth.tar"]
        # checkpoints_cali = ["/home/mdisk2/linzihao/medical_Imaging/LLD-MMRI2023/main/output/unpretrained_sample_aug/7clsrnd_3_small/fold5/checkpoint-80.pth.tar"]
        for checkpoint_path in checkpoints:
            model = Merge_uniformer_Base(pretrained=args.pretrained,
                                num_head=args.num_classes,
                                num_phase=8,
                                return_visualization=True)
            load_checkpoint(model, checkpoint_path)
            param_count = sum([m.numel() for m in model.parameters()])
            _logger.info('Model %s created, param count: %d' %
                        (args.model, param_count))
            model = model.cuda()
            model.eval()
            model_list.append(model)
        # for checkpoint_path in checkpoints_cali:
        #     model = Merge_uniformer(pretrained=args.pretrained,
        #                         num_head=args.num_classes,
        #                         num_phase=8,
        #                         return_visualization=True)
        #     load_checkpoint(model, checkpoint_path)
        #     param_count = sum([m.numel() for m in model.parameters()])
        #     _logger.info('Model %s created, param count: %d' %
        #                 (args.model, param_count))
        #     model = model.cuda()
        #     model.eval()
        #     model_list.append(model)
    elif args.model == "ensemble_rnd3":
        is_cls2 = False   
        checkpoints = ["/home/mdisk2/linzihao/medical_Imaging/LLD-MMRI2023/main/output/unpretrained_sample_aug/7clsrnd_3_resample/fold1/model_best.pth.tar",
                        "/home/mdisk2/linzihao/medical_Imaging/LLD-MMRI2023/main/output/unpretrained_sample_aug/7clsrnd_3_resample/fold2/model_best.pth.tar",
                        "/home/mdisk2/linzihao/medical_Imaging/LLD-MMRI2023/main/output/unpretrained_sample_aug/7clsrnd_3_resample/fold3/model_best.pth.tar",
                        "/home/mdisk2/linzihao/medical_Imaging/LLD-MMRI2023/main/output/unpretrained_sample_aug/7clsrnd_3_resample/fold4/model_best.pth.tar",
                        "/home/mdisk2/linzihao/medical_Imaging/LLD-MMRI2023/main/output/unpretrained_sample_aug/7clsrnd_3_resample/fold5/model_best.pth.tar",
                        ]
        # grid_mask
        # checkpoints = ["/home/mdisk2/linzihao/medical_Imaging/LLD-MMRI2023/main/output/unpretrained_sample_aug/7cls_test_gridmask/20230805-160109-uniformer_small_IL/checkpoint-82.pth.tar"]
        # extraaug
        # checkpoints = ["/home/mdisk2/linzihao/medical_Imaging/LLD-MMRI2023/main/output/unpretrained_sample_aug/7cls_test_extraaug/20230805-192548-uniformer_small_IL/checkpoint-124.pth.tar"]
        for checkpoint_path in checkpoints:
            model = Merge_uniformer(pretrained=args.pretrained,
                                num_head=args.num_classes,
                                num_phase=8,
                                return_visualization=True)
            load_checkpoint(model, checkpoint_path)
            param_count = sum([m.numel() for m in model.parameters()])
            _logger.info('Model %s created, param count: %d' %
                        (args.model, param_count))
            model = model.cuda()
            model.eval()
            model_list.append(model)
    elif args.model == 'ensemble_both':
        is_cls2 = False   
        checkpoints1 = ["/home/mdisk2/linzihao/medical_Imaging/LLD-MMRI2023/main/output/unpretrained_sample_aug/7clsrnd_2_base_ouths/fold1/checkpoint-104.pth.tar",
                        "/home/mdisk2/linzihao/medical_Imaging/LLD-MMRI2023/main/output/unpretrained_sample_aug/7clsrnd_2_base_ouths/fold2/checkpoint-119.pth.tar",
                        "/home/mdisk2/linzihao/medical_Imaging/LLD-MMRI2023/main/output/unpretrained_sample_aug/7clsrnd_2_base_ouths/fold3/checkpoint-108.pth.tar",
                        "/home/mdisk2/linzihao/medical_Imaging/LLD-MMRI2023/main/output/unpretrained_sample_aug/7clsrnd_2_base_ouths/fold4/checkpoint-138.pth.tar",
                        "/home/mdisk2/linzihao/medical_Imaging/LLD-MMRI2023/main/output/unpretrained_sample_aug/7clsrnd_2_base_ouths/fold5/checkpoint-100.pth.tar"]
        for checkpoint_path in checkpoints1:
            model = Merge_uniformer_Base(pretrained=args.pretrained,
                                num_head=args.num_classes,
                                num_phase=8,
                                return_visualization=True)
            load_checkpoint(model, checkpoint_path)
            param_count = sum([m.numel() for m in model.parameters()])
            _logger.info('Model %s created, param count: %d' %
                        (args.model, param_count))
            model = model.cuda()
            model.eval()
            model_list.append(model)
        checkpoints2 = ["/home/mdisk2/linzihao/medical_Imaging/LLD-MMRI2023/main/output/unpretrained_sample_aug/7clsrnd_3_bigsize/fold1/checkpoint-63.pth.tar",
                        "/home/mdisk2/linzihao/medical_Imaging/LLD-MMRI2023/main/output/unpretrained_sample_aug/7clsrnd_3_bigsize/fold2/checkpoint-131.pth.tar",
                        "/home/mdisk2/linzihao/medical_Imaging/LLD-MMRI2023/main/output/unpretrained_sample_aug/7clsrnd_3_bigsize/fold3/checkpoint-106.pth.tar",
                        "/home/mdisk2/linzihao/medical_Imaging/LLD-MMRI2023/main/output/unpretrained_sample_aug/7clsrnd_3_bigsize/fold4/checkpoint-104.pth.tar",
                        "/home/mdisk2/linzihao/medical_Imaging/LLD-MMRI2023/main/output/unpretrained_sample_aug/7clsrnd_3_bigsize/fold5/checkpoint-61.pth.tar"]
        for checkpoint_path in checkpoints2:
            model = Merge_uniformer(pretrained=args.pretrained,
                                num_head=args.num_classes,
                                num_phase=8,
                                return_visualization=True)
            load_checkpoint(model, checkpoint_path)
            param_count = sum([m.numel() for m in model.parameters()])
            _logger.info('Model %s created, param count: %d' %
                        (args.model, param_count))
            model = model.cuda()
            model.eval()
            model_list.append(model)
    elif args.model == 'ensemble_alltwoloss':
        is_cls2 = True
        checkpoints = ["/home/mdisk2/linzihao/medical_Imaging/LLD-MMRI2023/main/output/unpretrained_sample_aug/twolossrnd_2637_2cls/small_fold1/checkpoint-69.pth.tar",
                        "/home/mdisk2/linzihao/medical_Imaging/LLD-MMRI2023/main/output/unpretrained_sample_aug/twolossrnd_2637_2cls/small_fold2/checkpoint-93.pth.tar",
                        "/home/mdisk2/linzihao/medical_Imaging/LLD-MMRI2023/main/output/unpretrained_sample_aug/twolossrnd_2637_2cls/small_fold3/checkpoint-43.pth.tar",
                        "/home/mdisk2/linzihao/medical_Imaging/LLD-MMRI2023/main/output/unpretrained_sample_aug/twolossrnd_2637_2cls/small_fold4/checkpoint-143.pth.tar",
                        "/home/mdisk2/linzihao/medical_Imaging/LLD-MMRI2023/main/output/unpretrained_sample_aug/twolossrnd_2637_2cls/small_fold5/checkpoint-70.pth.tar",
                        ]
        for checkpoint_path in checkpoints:
            model = Merge_uniformer_twoloss(pretrained=args.pretrained,
                                num_head=args.num_classes,
                                num_phase=8,
                                return_visualization=True)
            load_checkpoint(model, checkpoint_path)
            param_count = sum([m.numel() for m in model.parameters()])
            _logger.info('Model %s created, param count: %d' %
                        (args.model, param_count))
            model = model.cuda()
            model.eval()
            model_list.append(model)
    

    dataset = MultiPhaseLiverDataset(args, is_training=False, is_cls2=is_cls2)

    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        num_workers=args.workers,
                        pin_memory=args.pin_mem,
                        shuffle=False,
                        collate_fn=collate_fn_eval_train)
    if args.model != 'ensemble_both':
        predictions_list = []
        labels_list = []
        eval_img_path_list = []
        err_label_list = []
        for idx, model in enumerate(model_list):
            predictions, label, eval_img_path, err_label = val_onemodel(model, dataset, loader, args, is_cls2=is_cls2)
            predictions_list.append(predictions)
            labels_list.append(label)
            eval_img_path_list.append(eval_img_path)
            err_label_list.append(err_label)
        err_id_dict = count_repeti_id(eval_img_path_list, err_label_list)
        predictions_total_ = [torch.cat(i) for i in predictions_list]
        predictions_total_ = torch.stack(predictions_total_).mean(dim=0)
    else:
        predictions_list = []
        labels_list = []
        eval_img_path_list = []
        err_label_list = []
        for idx, model in enumerate(model_list[0:5]):
            predictions, label, eval_img_path, err_label = val_onemodel(model, dataset, loader, args, is_cls2=is_cls2)
            predictions_list.append(predictions)
            labels_list.append(label)
            eval_img_path_list.append(eval_img_path)
            err_label_list.append(err_label)
        predictions_total_1 = [torch.cat(i) for i in predictions_list]
        predictions_total_1 = torch.stack(predictions_total_1).mean(dim=0)
        for idx, model in enumerate(model_list[5:]):
            predictions, label, eval_img_path, err_label = val_onemodel(model, dataset, loader, args, is_cls2=is_cls2)
            predictions_list.append(predictions)
            labels_list.append(label)
            eval_img_path_list.append(eval_img_path)
            err_label_list.append(err_label)
        predictions_total_2 = [torch.cat(i) for i in predictions_list[5:]]
        predictions_total_2 = torch.stack(predictions_total_2).mean(dim=0)
        predictions_total_ = torch.stack([predictions_total_1, predictions_total_2]).mean(0)
        err_id_dict = count_repeti_id(eval_img_path_list, err_label_list)

    for cur_pred, cur_label in zip(predictions_list, labels_list) :
        evaluation_metrics, confusion_matrix = compute_metrics(cur_pred,cur_label,args)
        print(confusion_matrix)
        # evaluation_metrics_total.append(evaluation_metrics)
        # confusion_matrix_total.append(confusion_matrix) 

    multimodel_metrics, multimodel_confusion_matrix = compute_multimodel_metrics(predictions_total_, labels_list[0], args)
    print(multimodel_confusion_matrix)

    output_str = 'Test:\n'
    for key, value in multimodel_metrics.items():
        output_str += f'{key}: {value}\n'
    _logger.info(output_str)
    
    return process_prediction(predictions_total_)

def val_onemodel(model, dataset, loader, args, is_cls2=False):
    amp_autocast = suppress
    
    if is_cls2:
        predictions = []
        labels = []
        predictions_2cls = []
        labels_twocls = []
        pbar = tqdm(total=len(dataset))
        with torch.no_grad():
            for batch_idx, eval_data in enumerate(loader):
                input = eval_data['img'].cuda()
                target = eval_data['label'].cuda()
                target_cls2 = eval_data['label_twocls'].cuda()
                with amp_autocast():
                    output, output_2cls, _ = model(input)
                predictions.append(output)
                predictions_2cls.append(output_2cls)
                labels.append(target)
                labels_twocls.append(target_cls2)
                pbar.update(args.batch_size)
            pbar.close()
            # score = process_prediction(predictions)     
            # label = torch.cat(labels, dim=0).detach()
        return predictions, labels
    else:
        predictions = []
        labels = []
        eval_img_path = []
        eval_err_label = []
        pbar = tqdm(total=len(dataset))
        with torch.no_grad():
            for batch_idx, eval_data in enumerate(loader):
                img_id = eval_data['img_path']
                input = eval_data['img'].cuda()
                target = eval_data['label'].cuda()
                with amp_autocast():
                    output, _ = model(input)
                predictions.append(output)
                labels.append(target)

                error_index_list = np.where((output.argmax(dim=1) == target).cpu().numpy() == False)[0].tolist()
                error_id = [img_id[i] for i in error_index_list]
                err_label = [target[i].item() for i in error_index_list]
                eval_img_path.extend(error_id)
                eval_err_label.extend(err_label)
                pbar.update(args.batch_size)
            pbar.close()
            # score = process_prediction(predictions)     
            # label = torch.cat(labels, dim=0).detach()
        return predictions, labels, eval_img_path, eval_err_label

def count_repeti_id(error_id_list, err_label_list):
    error_id_dict = {}
    for i_f, err_id_onefold in enumerate(error_id_list):
        for i, err_id in enumerate(err_id_onefold):
            if err_id not in error_id_dict.keys():
                error_id_dict[err_id] = {}
                error_id_dict[err_id]['count'] = 1
                error_id_dict[err_id]['label'] = err_label_list[i_f][i]
            else:
                error_id_dict[err_id]['count'] += 1
    return error_id_dict

def process_prediction(outputs):
    if isinstance(outputs,list):
        outputs = torch.cat(outputs, dim=0).detach()
    pred_score = torch.softmax(outputs, dim=1)
    return pred_score.cpu().numpy()

def hard_voting(score):
    """
    params: score n * [len(), 7] 
    """
    score = torch.stack(score, dim=0)
    score = score.max(dim=0)
    return score

def soft_voting(score):
    """
    params: score n * [len(), 7] 
    """
    score = torch.stack(score, dim=0)
    score = score.mean(dim=0)
    return score


def write_score2json(score_info, args):
    score_info = score_info.astype(float)
    score_list = []
    anno_info = np.loadtxt(args.val_anno_file, dtype=np.str_)
    for idx, item in enumerate(anno_info):
        id = item[0].rsplit('/', 1)[-1]
        score = list(score_info[idx])
        pred = score.index(max(score))
        pred_info = {
            'image_id': id,
            'prediction': pred,
            'score': score,
        }
        score_list.append(pred_info)
    json_data = json.dumps(score_list, indent=4)
    save_name = os.path.join(args.results_dir, args.team_name+'.json')
    file = open(save_name, 'w')
    file.write(json_data)
    file.close()
    _logger.info(f"Prediction has been saved to '{save_name}'.")


def main():
    setup_default_logging()
    args = parser.parse_args()
    score = validate(args)
    os.makedirs(args.results_dir, exist_ok=True)
    write_score2json(score, args)


if __name__ == '__main__':
    main()