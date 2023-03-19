import torch
import torchvision
from PIL import Image
from torchvision import transforms
import numpy as np
import glob
from torch.utils.data import DataLoader
from calibration_models import *
from torch import nn, optim
import os
#from tensorboardX import SummaryWriter
import time
import datetime
import os
import sys
import argparse
import random
from calibration_dataloader import *
from parallel import DataParallelModel, DataParallelCriterion

from ignite.engine import *
from ignite.handlers import *
from ignite.utils import *

sys.path.append(os.path.realpath(".."))


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=None, type=int, help='index of used GPU')
parser.add_argument('--model-name', default='LTS', type=str, help='model name: LTS, TS, Logistic, Dirichlet, Meta, Binary')
parser.add_argument('--epochs', default=40, type=int, help='max epochs')
parser.add_argument('--batch-size', default=20, type=int, help='batch size')
parser.add_argument('--lr', default=1e-3, type=float, help='inital learning rate')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training.')
parser.add_argument('--save-per-epoch', default=1, type=int, help='number of epochs to save model.')
parser.add_argument('--resume', action='store_true')
         
if __name__ == "__main__":

    args = parser.parse_args()
    model_name = str(args.model_name)
    
    
#    data_name = 'cityscapes'
    data_name = 'ade20k'
#    data_name = 'coco164k'
#    data_name = 'bdd100k'
#    data_name = 'davis480'
#    data_name = 'spacenet7ts'
#############################################    
    num_class = 150
#    eval_model_name = 'segformer_mit-b5_640x640_160k_ade20k_inp56' ##C8
#    meta_threshold = 0.7222
    
    eval_model_name = 'segmenter_vit-l_mask_8x1_640x640_160k_ade20k_inp56' ##C7  
    meta_threshold = 1.012
    
#    eval_model_name = 'knet_s3_upernet_swin-l_8x2_640x640_adamw_80k_ade20k_inp56'
#    meta_threshold = 0.8416
    
#    eval_model_name = 'knet_s3_deeplabv3_r50-d8_8x2_512x512_adamw_80k_ade20k_inp56'
#    meta_threshold = 0.9450

#    eval_model_name = 'upernet_convnext_large_fp16_640x640_160k_ade20k_inp56'
#    meta_threshold = 0.7677

#############################################   
#    num_class = 171
#    eval_model_name = 'segformer_mit-b5_512x512_80k_coco164k_inp56' ##C8 
#    meta_threshold = 1.6693
    
#    eval_model_name = 'segmenter_vit-b_mask_8x1_512x512_160k_coco164k_inp56' ##C8 
#    meta_threshold = 1.7535
    
#    eval_model_name = 'knet_s3_upernet_swin-l_8x2_640x640_adamw_80k_coco164k_inp56'
#    meta_threshold = 1.6897
   
#    eval_model_name = 'knet_s3_deeplabv3_r50-d8_8x2_512x512_adamw_80k_coco164k_inp56'
#    meta_threshold = 1.9549

#############################################   
#    num_class = 19
#    eval_model_name = 'segmenter_vit-b_mask_8x1_768x768_160k_bdd100k_inp56' ##C8 
#    meta_threshold = 0.6850

#    eval_model_name = 'segformer_mit-b5_512x1024_160k_bdd100k_inp56' ##C8 
#    meta_threshold = 0.5233

#    eval_model_name = 'knet_s3_upernet_swin-l_8x2_512x1024_adamw_80k_bdd100k_inp56'
#    meta_threshold = 0.4959
        
#    eval_model_name = 'knet_s3_deeplabv3_r50-d8_8x2_512x1024_adamw_80k_bdd100k_inp56'
#    meta_threshold = 0.5952

#############################################   
#    num_class = 51
##    eval_model_name = 'segmenter_vit-b_mask_8x1_512x512_40k_davis480_inp56' ##C8 
##    meta_threshold = 0.1868
#
##    eval_model_name = 'segformer_mit-b5_512x512_40k_davis480_inp56' ##C8 
##    meta_threshold = 0.1853
#
##    eval_model_name = 'knet_s3_upernet_swin-l_8x2_512x512_adamw_20k_davis480_inp56'
##    meta_threshold = 0.1860
#        
#    eval_model_name = 'knet_s3_deeplabv3_r50-d8_8x2_512x512_adamw_40k_davis480_inp56'
#    meta_threshold = 0.1858
#############################################

    train_logits_list = glob.glob('./'+data_name+'_eval/'+eval_model_name+'/'+data_name+'_train_*.logits.npz')
#    train_logits_list.sort()
    train_logits_list = train_logits_list[:int(len(train_logits_list))]
    
    test_logits_list = glob.glob('./'+data_name+'_eval/'+eval_model_name+'/'+data_name+'_val_*.logits.npz')
#    test_logits_list.sort()
    test_logits_list = validate_logits_list[:int(len(validate_logits_list))]
    

    
    nll_criterion = nn.CrossEntropyLoss(ignore_index=255)
    max_epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr

    trainset = calibration_dataloader(train_logits_list, 'val')
    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    testset = test_dataloader(test_logits_list, 'test')
    test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)    

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')


    if model_name == 'LTS':
        experiment_name = model_name + '_'+ data_name + '_epoch_' + str(max_epochs) + '_batchsize_' + str(batch_size) + '_lr_' + str(lr)
        calibration_model = LTS_CamVid_With_Image()
    elif model_name == 'TS':
        experiment_name = model_name + '_'+ data_name + '_epoch_' + str(max_epochs) + '_batchsize_' + str(batch_size) + '_lr_' + str(lr)
        calibration_model = Temperature_Scaling()
    elif model_name == 'Logistic':
        experiment_name = model_name + '_'+ data_name + '_epoch_' + str(max_epochs) + '_batchsize_' + str(batch_size) + '_lr_' + str(lr)
        calibration_model = Vector_Scaling()
    elif model_name == 'Dirichlet':
        experiment_name = model_name + '_'+ data_name + '_epoch_' + str(max_epochs) + '_batchsize_' + str(batch_size) + '_lr_' + str(lr)
        calibration_model = Dirichlet_Scaling()
    elif model_name == 'Meta':
        experiment_name = model_name + '_'+ data_name + '_epoch_' + str(max_epochs) + '_batchsize_' + str(batch_size) + '_lr_' + str(lr)
        calibration_model = Meta_Scaling()
    elif model_name == 'Binary':
        experiment_name = model_name + '_'+ data_name + '_epoch_' + str(max_epochs) + '_batchsize_' + str(batch_size) + '_lr_' + str(lr)
        calibration_model = Binary_Classifier()    
    else:
        raise ValueError('Wrong Model Name!')

    calibration_model.weights_init()
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        calibration_model.cuda(args.gpu)
    else:
        calibration_model.cuda()
        calibration_model = nn.DataParallel(calibration_model)

    optimizer = optim.AdamW(calibration_model.parameters(), lr=lr, weight_decay=1e-6)
    
    start_epoch = 0
    if args.resume:
        print('===> Resume from checkpoint...')
        checkpoint = torch.load('./P25_Combine_E20_'+data_name+'_'+ model_name +'_calibrated_'+ eval_model_name +'_checkpoint.pth.tar_binary_last')
            
        # load params
        new_state_dict = checkpoint['state_dict']
        calibration_model.load_state_dict(new_state_dict, strict=True)
        start_epoch = checkpoint['epoch']+1
        optimizer.load_state_dict(checkpoint['optimizer'])
        max_epochs = start_epoch+1
        
    calibration_model.train()


################################# Training ############################################################    
    min_loss = 999
    acc = 0
    for epoch in range(start_epoch, max_epochs):
        for i, (image, logits, labels) in enumerate(train_dataloader):
            image, logits, labels = image.cuda(args.gpu), logits.float().cuda(args.gpu), labels.long().cuda(args.gpu)
            optimizer.zero_grad()

            if model_name == 'Meta': #SegFormer:0.6513 #Segmenter:0.9186 #Convnext: 0.6716 #Knet-swin:0.7339 #Knet-deeplab:1.1398 #Adapter:5.0311
                logits_cali, labels = calibration_model(logits, labels, meta_threshold)
                loss = torch.sum(nll_criterion(logits_cali, labels))
            elif model_name == 'Dirichlet':
                logits_cali = calibration_model(logits)
                loss = torch.sum(nll_criterion(logits_cali, labels))
            elif model_name == 'Logistic':
                logits_cali = calibration_model(logits)
                loss = torch.sum(nll_criterion(logits_cali, labels))       
            elif model_name == 'LTS':
                logits_cali = calibration_model(logits, image, args)
                loss = torch.sum(nll_criterion(logits_cali, labels))
            elif model_name == 'Binary':
                logits_cali, binary_labels = calibration_model(logits, labels)
                 
#                ############## Duplicate Negative Samples (Data Imbalance) ###########################
                logits_cali = logits_cali.permute(0,2,3,1).view(-1, 2).squeeze()
                binary_labels = binary_labels.view(-1).long()
                
                b = binary_labels == 0
                indices = b.nonzero()
                copy_logits_cali = logits_cali[indices]
                copy_binary_labels = binary_labels[indices]
               
                ############## duplicate
                copy_logits_cali = torch.cat([copy_logits_cali]*1).squeeze()  ###################### data duplicates: 1/2/4 times
                copy_binary_labels = torch.cat([copy_binary_labels]*1).squeeze()
               
                logits_cali = torch.cat([logits_cali, copy_logits_cali], dim=0)
                binary_labels = torch.cat([binary_labels,copy_binary_labels], dim=0)
                
                loss = torch.sum(nll_criterion(logits_cali, binary_labels))

            else:
                logits_cali = calibration_model(logits)
                loss = torch.sum(nll_criterion(logits_cali, labels))
                      
            loss.backward()
            optimizer.step()
            
################################################################################################################

        calibration_model.eval()
        ece = 0
        correct = 0
        total = 0
        loss = 0

        mislabel_correct = 0
        mislabel_total = 0
        
        corrlabel_correct = 0
        corrlabel_total = 0
        
        mis_lr_auc = 0
        corr_lr_auc = 0 
        
        mislabel_pred_total = 1
        corrlabel_pred_total = 1
        
        for i, (image, logits, labels) in enumerate(test_dataloader):
            image, logits, labels = image.cuda(args.gpu), logits.float().cuda(args.gpu), labels.long().cuda(args.gpu)
            
            if model_name == 'Meta': #SegFormer:0.6513 #Segmenter:0.9186 #Convnext: 0.6716 #Knet-swin:0.7339 #Knet-deeplab:1.1398 #Adapter:5.0311
                logits_cali, labels = calibration_model(logits, labels, meta_threshold)
                loss += torch.sum(nll_criterion(logits_cali, labels)).detach()
            elif model_name == 'Binary':        
            
                logits_cali, binary_labels = calibration_model(logits, labels)
                loss += torch.sum(nll_criterion(logits_cali, binary_labels)).detach()
                
                softmax = torch.nn.Softmax(dim=1)
                probs = softmax(logits_cali)
                probs = probs.permute(0,2,3,1).reshape(-1, 2) # Binary Classifier Class=2
                _, preds = torch.max(probs, dim=1)
                binary_labels = binary_labels.reshape(-1)
                
                b = labels.reshape(-1) != 255
                indices = b.nonzero()
        
                correct += torch.eq(binary_labels[indices], preds[indices]).sum().item()
                total += binary_labels[indices].size(0)
                
                ###################################### Incorrect Label Evaluation
                b = ((labels.reshape(-1) != 255) & (binary_labels == 0))
                indices = b.nonzero()
                mislabel_correct += torch.eq(binary_labels[indices], preds[indices]).sum().item()
                mislabel_total += binary_labels[indices].size(0)
                
                b = labels.reshape(-1) != 255
                indices = b.nonzero()
                mislabel_pred_total += np.count_nonzero((preds[indices]==0).cpu().numpy())

                
                ###################################### Correct Label Evaluation
                b = ((labels.reshape(-1) != 255) & (binary_labels == 1))
                indices = b.nonzero()
                corrlabel_correct += torch.eq(binary_labels[indices], preds[indices]).sum().item()
                corrlabel_total += binary_labels[indices].size(0)
                
                b = labels.reshape(-1) != 255
                indices = b.nonzero()
                corrlabel_pred_total += np.count_nonzero((preds[indices]==1).cpu().numpy())

                cal_acc = mislabel_correct/mislabel_total
                
            elif model_name == 'LTS':
                logits_cali = calibration_model(logits, image, args) 
                loss += torch.sum(nll_criterion(logits_cali, labels)).detach()           
            else:
                logits_cali = calibration_model(logits)
                loss += torch.sum(nll_criterion(logits_cali, labels)).detach()


        if model_name == 'Binary':
            mis_lr_auc /= i
            corr_lr_auc /= i
            
            current_state = {'epoch': epoch,
                          'state_dict': calibration_model.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'corr_acc': corrlabel_correct/corrlabel_pred_total,
                          'mis_acc': mislabel_correct/mislabel_pred_total
                                    }
                                    
            torch.save(current_state, './Scenario_E'+str(max_epochs)+'_' + data_name + '_' + model_name + '_calibrated_' + eval_model_name+'_checkpoint.pth.tar_binary_last') 
            print("{} epoch, mean_accuracy: {:.5f}, corr_accuracy: {:.5f}, mis_accuracy: {:.5f}".format(epoch, cal_acc, corrlabel_correct/corrlabel_pred_total, mislabel_correct/mislabel_pred_total) )    

        else:
            current_state = {'epoch': epoch,
                          'state_dict': calibration_model.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'min_loss': min_loss,
                                    }
            if loss.item()/i < min_loss:
                min_loss = loss.item()/i
                torch.save(current_state, './Scenario_E'+str(max_epochs) +'_' + data_name + '_' + model_name + '_calibrated_' + eval_model_name+'_checkpoint.pth.tar_binary_best')
                print("===> Saving...")
            torch.save(current_state, './Scenario_E'+str(max_epochs) +'_' + data_name + '_' + model_name + '_calibrated_' + eval_model_name+'_checkpoint.pth.tar_binary_last')
            print("{} epoch, test loss: {:.5f}".format(epoch, loss.item()/i))

        