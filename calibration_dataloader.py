from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import torch.nn.functional as F

#data_name = 'cityscapes'
data_name = 'ade20k'
#data_name = 'coco164k'
#data_name = 'bdd100k'
#data_name = 'davis480'
#data_name = 'spacenet7ts'

#eval_model_name = 'segformer_mit-b5_640x640_160k_ade20k_inp56' ##C8
eval_model_name = 'segmenter_vit-l_mask_8x1_640x640_160k_ade20k_inp56' ##C7  
#eval_model_name = 'knet_s3_upernet_swin-l_8x2_640x640_adamw_80k_ade20k_inp56'
#eval_model_name = 'knet_s3_deeplabv3_r50-d8_8x2_512x512_adamw_80k_ade20k_inp56'
#eval_model_name = 'upernet_convnext_large_fp16_640x640_160k_ade20k_inp56'
#eval_model_name = 'mask2former_beit_adapter_large_896_80k_ade20_inp56k'

#eval_model_name = 'segformer_mit-b5_512x512_80k_coco164k_inp56'
#eval_model_name = 'segmenter_vit-b_mask_8x1_512x512_160k_coco164k_inp56'
#eval_model_name = 'knet_s3_upernet_swin-l_8x2_640x640_adamw_80k_coco164k_inp56'
#eval_model_name = 'knet_s3_deeplabv3_r50-d8_8x2_512x512_adamw_80k_coco164k_inp56'
#eval_model_name = 'mask2former_beit_adapter_large_896_80k_cocostuff164k_inp56'

#eval_model_name = 'segmenter_vit-b_mask_8x1_768x768_160k_bdd100k_inp56'
#eval_model_name = 'segformer_mit-b5_512x1024_160k_bdd100k_inp56'
#eval_model_name = 'knet_s3_upernet_swin-l_8x2_512x1024_adamw_80k_bdd100k_inp56'
#eval_model_name = 'knet_s3_deeplabv3_r50-d8_8x2_512x1024_adamw_80k_bdd100k_inp56'
 
#eval_model_name = 'segmenter_vit-b_mask_8x1_512x512_40k_davis480_inp56'
#eval_model_name = 'segformer_mit-b5_512x512_40k_davis480_inp56'
#eval_model_name = 'knet_s3_upernet_swin-l_8x2_512x512_adamw_20k_davis480_inp56'
#eval_model_name = 'knet_s3_deeplabv3_r50-d8_8x2_512x512_adamw_40k_davis480_inp56'

#eval_model_name = 'knet_s3_deeplabv3_r50-d8_8x2_512x512_adamw_40k_spacenet7_sp_inp56'
#eval_model_name = 'knet_s3_upernet_swin-l_8x2_512x512_adamw_20k_spacenet7_sp_inp56'
    
class calibration_dataloader(Dataset):
    def __init__(self, logits_list, exp_name):
        self.logits_list = logits_list
        self.exp_name = exp_name

    def __len__(self):
        return len(self.logits_list)

    def __getitem__(self, id):

#####################################################################################################################################################################        
        logits_item_file = './'+data_name+'_eval/'+eval_model_name+'/'+data_name+'_train_'+ self.logits_list[id][-15:-11] + '.logits.npz'
        label_item_file = './'+data_name+'_eval/'+'Ground_Truth'+'/'+data_name+'_train_'+ self.logits_list[id][-15:-11]  + '.gt.npz'
        image_item_file = './'+data_name+'_eval/'+eval_model_name+'/'+data_name+'_train_'+ self.logits_list[id][-15:-11]  + '.img.npz'  ######## coco train
        
        logits_array = np.load(logits_item_file, allow_pickle=True)['data'].squeeze()
        label_array = np.load(label_item_file,  allow_pickle=True)['data'].squeeze()
        image_array = np.load(image_item_file,  allow_pickle=True)['data'].squeeze()
        
        label_array[label_array == 0] = 255
        label_array = label_array - 1
        label_array[label_array == 254] = 255
        label_array = F.interpolate( torch.from_numpy(label_array[np.newaxis,np.newaxis,:,:]).float(), size=(56, 56), mode = 'nearest').long().numpy().squeeze()
        
        logits_item_file = './'+data_name+'_eval/'+eval_model_name+'/'+data_name+'_neg_'+ self.logits_list[id][-15:-11] + '.logits.npz'
        image_item_file = './'+data_name+'_eval/'+eval_model_name+'/'+data_name+'_neg_'+ self.logits_list[id][-15:-11]  + '.img.npz'  ### coco train
        
        logits_array = np.stack((logits_array, np.load(logits_item_file, allow_pickle=True)['data'].squeeze()), axis=0)
        label_array = np.stack((label_array, np.ones_like(label_array)+255), axis=0)
        image_array = np.stack((image_array, np.load(image_item_file,  allow_pickle=True)['data'].squeeze()), axis=0) 
        
        return image_array, logits_array, label_array



class test_dataloader(Dataset):
    def __init__(self, logits_list, exp_name):
        self.logits_list = logits_list
        self.exp_name = exp_name

    def __len__(self):
        return len(self.logits_list)

    def __getitem__(self, id):

#####################################################################################################################################################################        
        logits_item_file = './'+data_name+'_eval/'+eval_model_name+'/'+data_name+'_val_'+ self.logits_list[id][-15:-11] + '.logits.npz'
        label_item_file = './'+data_name+'_eval/'+'Ground_Truth'+'/'+data_name+'_val_'+ self.logits_list[id][-15:-11]  + '.gt.npz'
        image_item_file = './'+data_name+'_eval/'+eval_model_name+'/'+data_name+'_val_'+ self.logits_list[id][-15:-11]  + '.img.npz'  ### coco train
        
        logits_array = np.load(logits_item_file, allow_pickle=True)['data'].squeeze()
        label_array = np.load(label_item_file,  allow_pickle=True)['data'].squeeze()
        image_array = np.load(image_item_file,  allow_pickle=True)['data'].squeeze()
        
        label_array[label_array == 0] = 255 
        label_array = label_array - 1
        label_array[label_array == 254] = 255

        label_array = F.interpolate( torch.from_numpy(label_array[np.newaxis,np.newaxis,:,:]).float(), size=(56, 56), mode = 'nearest').long().numpy().squeeze()

        return image_array, logits_array, label_array




