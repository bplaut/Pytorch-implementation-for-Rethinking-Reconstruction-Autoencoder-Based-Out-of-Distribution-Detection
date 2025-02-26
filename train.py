import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torchvision
import argparse
import pickle
import torch
from collections import defaultdict
from datetime import datetime
import torch.nn as nn
import numpy as np
#from mmcv.cnn import get_model_complexity_info
import shutil
from configs import cfg, update_config
from dataset.augmentation import get_transform
from metrics.ood_metrics import get_ood_metrics
from metrics.cls_metrics import get_cls_metrics
from losses.ILCEloss import ILCE_loss
from tools.distributed import distribute_bn
from batch_engine import valid_trainer, batch_trainer
from dataset.ood.ood_dataset import ood_dataloader
from models.base_block import Classifier, Network, ILCE
from tools.function import get_model_log_path, get_reload_weight, seperate_weight_decay
from tools.utils import build_optimizer_schedular, time_str, save_ckpt, ReDirectSTD, set_seed, str2bool
from models.backbone.wideresnet import WideResNet
from models.backbone.dense_net import DenseNet
from draw import create_annotated_ood_images
from make_tex import make_tex

torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)

def main(cfg, args):
    
    set_seed(605)
    exp_dir = os.path.join('exp_result', cfg.DATASET.NAME)
    model_dir, log_dir = get_model_log_path(exp_dir, cfg.NAME)
    stdout_file = os.path.join(log_dir, f'stdout_{time_str()}.txt')
    save_model_path = os.path.join(model_dir, f'ckpt_max_{time_str()}.pth')

    if cfg.REDIRECTOR:
        print('redirector stdout')
        ReDirectSTD(stdout_file, 'stdout', False)

    args.distributed = None
    args.world_size = 1
    args.rank = 0
    
    train_tsfm, valid_tsfm = get_transform(cfg)
    train_fraction = 0.8
    train_set = ood_dataloader(cfg.DATASET.NAME, datapath=args.train_datapath, transform=train_tsfm, lo=0, hi=train_fraction)
    valid_set = ood_dataloader(cfg.DATASET.NAME, datapath=args.train_datapath, transform=valid_tsfm, lo=train_fraction, hi=1)
    if args.test_datapath is not None:
        test_dir = os.path.dirname(args.test_datapath)
        test_set = ood_dataloader(test_dir, datapath=args.test_datapath, transform=valid_tsfm)
    args.cls_num = train_set.attr_num
    print(cfg)
        
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=cfg.TRAIN.BATCH_SIZE, sampler=train_sampler, 
        shuffle=train_sampler is None,
        num_workers=4,
        pin_memory=True,
        drop_last=True)
    
    if cfg.NAME == 'ID':
    
        valid_loader = torch.utils.data.DataLoader(dataset=valid_set, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
        valid_loader_list = [valid_loader]
        
        if args.local_rank == 0:
            print('-' * 60)
            print(f'{cfg.DATASET.NAME},'
                f'{cfg.DATASET.TRAIN_SPLIT} set: {len(train_set)}, '
                f'{cfg.DATASET.TEST_SPLIT} set: {len(valid_set)}, ')

    elif cfg.NAME == 'OOD':
        
        valid_loader = torch.utils.data.DataLoader(dataset=valid_set, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

        test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

        valid_loader_list = [valid_loader, test_loader]
        if args.local_rank == 0:
            print('-' * 60)
            print(f'{cfg.DATASET.NAME},'
                f'{cfg.DATASET.TRAIN_SPLIT} set: {len(train_set)}, '
                f'{cfg.DATASET.TEST_SPLIT} set: {len(valid_loader.dataset)}, '
                f'{os.path.dirname(args.test_datapath)} set: {len(test_loader.dataset)}, ')
   
    #do not use self.dropout but F.dropout with self.training in forward feeding arg.
    
    if cfg.BACKBONE.TYPE == 'densenet':
        
        depth = 100
        growth_rate = 12
        efficient = True
        if (depth - 4) % 3:
            raise Exception('Invalid depth')
        block_config = [(depth - 4) // 6 for _ in range(3)]
        backbone = DenseNet(growth_rate=growth_rate,block_config=block_config,num_init_features=growth_rate*2,
        num_classes=None,small_inputs=True,efficient=efficient)
        classifier = Classifier(342,args.cls_num)
        
        if cfg.NAME == 'ID':
            decoder_1 = None
            decoder_2 = None
        else:
            decoder_1 = ILCE(args.cls_num,342,342,342)
            decoder_2 = ILCE(args.cls_num,512,256,args.cls_num)
    
    elif cfg.BACKBONE.TYPE == 'wideresnet':
        
        backbone = WideResNet(28,10,cfg.BACKBONE.DROP_OUT)
        classifier = Classifier(640,args.cls_num)
    
        if cfg.NAME == 'ID':
            decoder_1 = None
            decoder_2 = None
        else:
            decoder_1 = ILCE(args.cls_num,640,640,640)
            decoder_2 = ILCE(args.cls_num,640,256,args.cls_num)
            
    else:
        raise Exception('Invalid backbone.')
        
        
    model = Network(backbone, classifier, decoder_1, decoder_2)
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    
    model = torch.nn.DataParallel(model.cuda())

    if cfg.NAME == 'OOD' or cfg.RELOAD.TYPE:
        model = get_reload_weight(model, cfg.RELOAD.PTH)

    criterion = ILCE_loss(cfg.TRAIN.CLS_LOSS_WEIGHT)
    criterion = criterion.cuda()
        
    if cfg.NAME == 'ID':
        if cfg.TRAIN.BN_WD:
            param_groups = [{'params': model.module.backbone.parameters(),
                            'lr': cfg.TRAIN.LR_SCHEDULER.LR_FT,
                            'weight_decay': cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY},
                        {'params': model.module.classifier.parameters(),
                            'lr': cfg.TRAIN.LR_SCHEDULER.LR_NEW,
                            'weight_decay': cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY}]
        else:
            ft_params = seperate_weight_decay(
                model.module.backbone.parameters(),
                lr=cfg.TRAIN.LR_SCHEDULER.LR_FT,
                weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY)

            fresh_params = seperate_weight_decay(
                model.module.classifier.parameters(),
                lr=cfg.TRAIN.LR_SCHEDULER.LR_NEW,
                weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY)
            
            param_groups = ft_params + fresh_params
    else:
        if cfg.TRAIN.BN_WD:
            param_groups = [{'params': filter(lambda p: p.requires_grad, model.module.parameters()),'lr': cfg.TRAIN.LR_SCHEDULER.LR_NEW, 'weight_decay': cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY}]
        else:
            param_groups = seperate_weight_decay(filter(lambda p: p.requires_grad, model.module.parameters()), lr=cfg.TRAIN.LR_SCHEDULER.LR_NEW, weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY)

    optimizer, lr_scheduler = build_optimizer_schedular(cfg, param_groups, train_loader, train_set)

    best_metric, epoch = trainer(cfg, args, epoch=cfg.TRAIN.MAX_EPOCH,
                                 model=model, 
                                 train_loader=train_loader,
                                 valid_loader_list=valid_loader_list,
                                 criterion=criterion,
                                 optimizer=optimizer,
                                 lr_scheduler=lr_scheduler,
                                 path=save_model_path)
    if args.local_rank == 0:
        print(f'{cfg.NAME},  best_metrc : {best_metric} in epoch{epoch}')

def trainer(cfg, args, epoch, model, train_loader, valid_loader_list, criterion, optimizer, lr_scheduler,
            path):
    
    maximum = float(-np.inf)
    best_epoch = 0

    result_list = defaultdict()
    result_path = os.path.join(os.path.dirname(path), 'metric.pkl')
    

    for e in range(epoch):
        
        lr = optimizer.param_groups[0]['lr']
                
        train_loss, train_gt, train_logits = batch_trainer(
            cfg,
            args=args,
            epoch=e,
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=lr_scheduler if cfg.TRAIN.LR_SCHEDULER.TYPE == 'annealing_cosine' or cfg.TRAIN.LR_SCHEDULER.TYPE == 'cosine_annealing' else None,
        )
        
        ###
        #if cfg.TRAIN.LR_SCHEDULER.TYPE == 'plateau':
        #    lr_scheduler.step(metrics=valid_loss)
        if cfg.TRAIN.LR_SCHEDULER.TYPE == 'warmup_cosine':
            lr_scheduler.step(epoch=e + 1)
        elif cfg.TRAIN.LR_SCHEDULER.TYPE == 'multistep':
            lr_scheduler.step()
            
        if (e >= cfg.TRAIN.MAX_EPOCH * 0.7 and e % 4 == 0) or args.always_eval:
            if args.save_image_path is not None:
                output_dir = os.path.join(args.save_image_path, f'ood_visualized_epoch_{e}')
                create_annotated_ood_images(cfg, args, model, valid_loader_list, output_dir)
                make_tex(output_dir, f'ood_visualized_epoch_{e}.tex')
            valid_loss, valid_gt, valid_logits = valid_trainer(
                cfg,
                args=args,
                epoch=e,
                model=model,
                valid_loader_list=valid_loader_list,
                criterion=criterion,
            )
                
            if cfg.NAME == 'OOD':
                preds_all = valid_logits[1]
                valid_logits = valid_logits[0]
                ood_result = get_ood_metrics(preds_all, cfg=cfg)

                train_result = get_cls_metrics(train_logits,train_gt)
                valid_result = get_cls_metrics(valid_logits,valid_gt)
                
                result_list[e] = {
                    'train_result': train_result,
                    'valid_result': valid_result,
                    'train_gt': train_gt,
                    'valid_gt': valid_gt,
                    'ood_result': ood_result,
                    'pred_all': preds_all,
                    }

                if args.local_rank == 0:
                    print(f'Evaluation on train set, train losses {train_loss[0]}, {train_loss[1]}, {train_loss[2]}\n',
                            'accuracy: {:.4f} \n'.format(train_result.acc))
                    print(f'current best {maximum} at {best_epoch}\n')
                    print(f'Evaluation on valid set, valid losses {valid_loss[0]}, {valid_loss[1]}, {valid_loss[2]}\n',
                            'accuracy: {:.4f} \n'.format(valid_result.acc),
                            f'\Performance on test set:\nFPR@TPR=95 = {ood_result[0].fpr}\nDetection error = {ood_result[0].de}\nAUROC = {ood_result[0].roc}\nAUPR = {ood_result[0].pr} \n')
                    
                cur_metric = ood_result[0].roc
                
            elif cfg.NAME == 'ID':
                
                train_result = get_cls_metrics(train_logits,train_gt)
                valid_result = get_cls_metrics(valid_logits,valid_gt)
            
                if args.local_rank == 0:
                    print(f'Evaluation on train set, train losses {train_loss}\n',
                            'accuracy: {:.4f} \n'.format(train_result.acc))
                    print(f'current best {maximum} at {best_epoch}\n')
                    print(f'Evaluation on valid set, valid losses {valid_loss}\n',
                            'accuracy: {:.4f} \n'.format(valid_result.acc))
                    print(f'{time_str()}')
                    print('-' * 60)
                    
                result_list[e] = {
                    'train_result': train_result,
                    'valid_result': valid_result,
                    'train_gt': train_gt,
                    'valid_gt': valid_gt,
                    }
                cur_metric = valid_result.acc
                
            if cur_metric > maximum:
                maximum = cur_metric
                best_epoch = e
                save_ckpt(model, path, e, maximum)

            with open(result_path, 'wb') as f:
                pickle.dump(result_list, f)

    return maximum, best_epoch


def argument_parser():
    parser = argparse.ArgumentParser(description="out-of-distribution detection",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--cfg", help="decide which cfg to use", type=str,
        default="./configs/ood/cifar100.yaml",
    )

    parser.add_argument("--debug", type=str2bool, default="true")
    parser.add_argument('--local_rank', help='node rank for distributed training', default=0,
                        type=int)
    parser.add_argument('--dist_bn', type=str, default='',
                        help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
    parser.add_argument('-r', '--train_datapath', type=str, default=None,
                        help='Path to the training dataset')
    parser.add_argument('-e', '--test_datapath', type=str, default=None,
                        help='Path to the test dataset')
    parser.add_argument('-s', '--save_image_path', type=str, default=None,
                        help='Path to save images (if None, images are not saved)')
    parser.add_argument('--always_eval', action='store_true',help='Evaluate performance every epoch (as opposed to just towards the end of the process)')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = argument_parser()
    update_config(cfg, args)
    main(cfg, args)
