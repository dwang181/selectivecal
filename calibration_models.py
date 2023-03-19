import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

import numpy as np
import math 
from typing import Tuple
from matplotlib import pyplot as plt

#num_class = 19
num_class = 150
#num_class = 171
#num_class = 2

## Borrowed from https://gist.github.com/gpleiss/0b17bc4bd118b49050056cfcd5446c71
def make_model_diagrams(outputs, labels,  n_bins=10):
    """
    outputs - a torch tensor (size n x num_classes) with the outputs from the final linear layer
    - NOT the softmaxes
    labels - a torch tensor (size n) with the labels
    """
    softmaxes = torch.nn.functional.softmax(outputs, 1)
    confidences, predictions = softmaxes.max(1)
    accuracies = torch.eq(predictions, labels)
    overall_accuracy = (predictions==labels).sum().item()/len(labels)
    
    # Reliability diagram
    bins = torch.linspace(0, 1, n_bins + 1)
    width = 1.0 / n_bins
    bin_centers = np.linspace(0, 1.0 - width, n_bins) + width / 2
    bin_indices = [confidences.ge(bin_lower) * confidences.lt(bin_upper) for bin_lower, bin_upper in zip(bins[:-1], bins[1:])]
    
    bin_corrects = np.array([ torch.mean(accuracies[bin_index].float()) for bin_index in bin_indices])
    bin_scores = np.array([ torch.mean(confidences[bin_index].float()) for bin_index in bin_indices])
    bin_corrects = np.nan_to_num(bin_corrects)
    bin_scores = np.nan_to_num(bin_scores)
    
    plt.figure(0, figsize=(8, 8))
    gap = np.array(bin_scores - bin_corrects)
    
    confs = plt.bar(bin_centers, bin_corrects, color=[0, 0, 1], width=width, ec='black')
    bin_corrects = np.nan_to_num(np.array([bin_correct  for bin_correct in bin_corrects]))
    gaps = plt.bar(bin_centers, gap, bottom=bin_corrects, color=[1, 0.7, 0.7], alpha=0.5, width=width, hatch='//', edgecolor='r')
    
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.legend([confs, gaps], ['Accuracy', 'Gap'], loc='upper left', fontsize='x-large')

    ece = _calculate_ece(outputs, labels)

    # Clean up
    bbox_props = dict(boxstyle="square", fc="lightgrey", ec="gray", lw=1.5)
    plt.text(0.17, 0.82, "ECE: {:.4f}".format(ece), ha="center", va="center", size=20, weight = 'normal', bbox=bbox_props)

    plt.title("Reliability Diagram", size=22)
    plt.ylabel("Accuracy",  size=18)
    plt.xlabel("Confidence",  size=18)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.savefig('reliability_diagram.png')
    plt.show()
    return ece

def _calculate_ece(logits, labels, n_bins=10):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    softmaxes = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1, device=logits.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece.item()
    
def initialization(m):
    # Initialize kernel weights with Gaussian distributions
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
        
    
        

############################################################################################

class Temperature_Scaling(nn.Module):
    def __init__(self):
        super(Temperature_Scaling, self).__init__()
        self.temperature_single = nn.Parameter(torch.ones(1))

    def weights_init(self):
        self.temperature_single.data.fill_(1)

    def forward(self, logits):
        temperature = self.temperature_single.expand(logits.size()).cuda()
        return logits / temperature


class Vector_Scaling(nn.Module):
    def __init__(self):
        super(Vector_Scaling, self).__init__()
        self.vector_parameters = nn.Parameter(torch.ones(1, num_class, 1, 1))
        self.vector_offset = nn.Parameter(torch.zeros(1, num_class, 1, 1))

    def weights_init(self):
#        pass
        self.vector_offset.data.fill_(0)
        self.vector_parameters.data.fill_(1)

    def forward(self, logits):
        return logits * self.vector_parameters.cuda() + self.vector_offset.cuda()
        
class Stochastic_Spatial_Scaling(nn.Module):
    def __init__(self):
        super(Stochastic_Spatial_Scaling, self).__init__()

        conv_fn = nn.Conv2d
        self.rank = 10
        self.num_classes = num_class
        self.epsilon = 1e-5
        self.diagonal = False  # whether to use only the diagonal (independent normals)
        self.conv_logits = conv_fn(num_class, num_class, kernel_size=(1, ) * 2)

    def weights_init(self):
        initialization(self.conv_logits)
        
    def fixed_re_parametrization_trick(dist, num_samples):
        assert num_samples % 2 == 0
        samples = dist.rsample((num_samples // 2,))
        mean = dist.mean.unsqueeze(0)
        samples = samples - mean
        return torch.cat([samples, -samples]) + mean
                
    def forward(self, logits):

        batch_size = logits.shape[0]
        event_shape = (self.num_classes,) + logits.shape[2:]


        mean = self.conv_logits(logits)
        cov_diag = (mean*1e-5).exp() + self.epsilon
        mean = mean.view((batch_size, -1))
        cov_diag = cov_diag.view((batch_size, -1))                     

        base_distribution = td.Independent(td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1)
        distribution = ReshapedDistribution(base_distribution, event_shape)

        num_samples=2
        samples = distribution.rsample((num_samples // 2,)).cpu()
        mean = distribution.mean.unsqueeze(0).cpu()
        samples = samples - mean
        logit_samples = torch.cat([samples, -samples]) + mean
        logit_mean = logit_samples.mean(dim=0).cuda()

        return logit_mean
        

class Dirichlet_Scaling(nn.Module):
    def __init__(self):
        super(Dirichlet_Scaling, self).__init__()
        self.dirichlet_linear = nn.Linear(num_class, num_class)

    def weights_init(self):
        self.dirichlet_linear.weight.data.copy_(torch.eye(self.dirichlet_linear.weight.shape[0]))
        self.dirichlet_linear.bias.data.copy_(torch.zeros(*self.dirichlet_linear.bias.shape))

    def forward(self, logits):
        logits = logits.permute(0,2,3,1)
        softmax = torch.nn.Softmax(dim=-1)
        probs = softmax(logits)
        ln_probs = torch.log(probs+1e-10)

        return self.dirichlet_linear(ln_probs).permute(0,3,1,2)   

        
        
class Meta_Scaling(nn.Module):
    def __init__(self):
        super(Meta_Scaling, self).__init__()
        self.temperature_single = nn.Parameter(torch.ones(1))
        self.alpha = 0.05

    def weights_init(self):
        self.temperature_single.data.fill_(1)
        
    def forward(self, logits, gt, threshold):

        logits = logits.permute(0,2,3,1).view(-1, num_class)
        gt = gt.view(-1)
    
        if self.training:
            neg_ind = torch.argmax(logits, axis=1) == gt
            
            xs_pos, ys_pos = logits[~neg_ind], gt[~neg_ind]
            xs_neg, ys_neg = logits[neg_ind], gt[neg_ind]
            
            start = np.random.randint(int(xs_neg.shape[0]*1/3))+1
            x2 = torch.cat((xs_pos, xs_neg[start:int(xs_neg.shape[0]/2)+start]), 0)
            y2 = torch.cat((ys_pos, ys_neg[start:int(xs_neg.shape[0]/2)+start]), 0)
            
            softmax = torch.nn.Softmax(dim=-1)
            p = softmax(x2)
            scores_x2 = torch.sum(-p*torch.log(p), dim=-1)
        
            cond_ind = scores_x2 < threshold
            cal_logits, cal_gt = x2[cond_ind], y2[cond_ind]
        
            temperature = self.temperature_single.expand(cal_logits.size())
            cal_logits = cal_logits / temperature
            
        else:
            x2 = logits
            y2 = gt
        
            softmax = torch.nn.Softmax(dim=-1)
            p = softmax(x2)
            scores_x2 = torch.sum(-p*torch.log(p), dim=-1)
        
            cond_ind = scores_x2 < threshold
            scaled_logits, scaled_gt = x2[cond_ind], y2[cond_ind]
            inference_logits, inference_gt = x2[~cond_ind], y2[~cond_ind]
        
            temperature = self.temperature_single.expand(scaled_logits.size())
            scaled_logits = scaled_logits / temperature

            inference_logits = torch.ones_like(inference_logits)
            
            cal_logits = torch.cat((scaled_logits, inference_logits), 0)
            cal_gt = torch.cat((scaled_gt, inference_gt), 0)

        return cal_logits, cal_gt

class LTS_CamVid_With_Image(nn.Module):
    def __init__(self):
        super(LTS_CamVid_With_Image, self).__init__()
        self.temperature_level_2_conv1 = nn.Conv2d(num_class, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_conv2 = nn.Conv2d(num_class, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_conv3 = nn.Conv2d(num_class, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_conv4 = nn.Conv2d(num_class, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_param1 = nn.Conv2d(num_class, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_param2 = nn.Conv2d(num_class, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_param3 = nn.Conv2d(num_class, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_conv_img = nn.Conv2d(3, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_param_img = nn.Conv2d(num_class, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)

    def weights_init(self):
        torch.nn.init.zeros_(self.temperature_level_2_conv1.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv1.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv2.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv2.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv3.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv3.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv4.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv4.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_param1.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_param1.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_param2.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_param2.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_param3.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_param3.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv_img.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv_img.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_param_img.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_param_img.bias.data)


    def forward(self, logits, image):
        temperature_1 = self.temperature_level_2_conv1(logits)
        temperature_1 += (torch.ones(1)).cuda()
        temperature_2 = self.temperature_level_2_conv2(logits)
        temperature_2 += (torch.ones(1)).cuda()
        temperature_3 = self.temperature_level_2_conv3(logits)
        temperature_3 += (torch.ones(1)).cuda()
        temperature_4 = self.temperature_level_2_conv4(logits)
        temperature_4 += (torch.ones(1)).cuda()
        temperature_param_1 = self.temperature_level_2_param1(logits)
        temperature_param_2 = self.temperature_level_2_param2(logits)
        temperature_param_3 = self.temperature_level_2_param3(logits)
        temp_level_11 = temperature_1 * torch.sigmoid(temperature_param_1) + temperature_2 * (1.0 - torch.sigmoid(temperature_param_1))
        temp_level_num_class = temperature_3 * torch.sigmoid(temperature_param_2) + temperature_4 * (1.0 - torch.sigmoid(temperature_param_2))
        temp_1 = temp_level_11 * torch.sigmoid(temperature_param_3) + temp_level_num_class * (1.0 - torch.sigmoid(temperature_param_3))
        temp_2 = self.temperature_level_2_conv_img(image) + torch.ones(1).cuda()
        temp_param = self.temperature_level_2_param_img(logits)
        temperature = temp_1 * torch.sigmoid(temp_param) + temp_2 * (1.0 - torch.sigmoid(temp_param))
        sigma = 1e-8
        temperature = F.relu(temperature + torch.ones(1).cuda()) + sigma
        temperature = temperature.repeat(1, num_class, 1, 1)
        return logits / temperature



class Binary_Classifier(nn.Module):
    def __init__(self):
        super(Binary_Classifier, self).__init__()
        self.dirichlet_linear = nn.Linear(num_class, num_class)
        self.binary_linear = nn.Linear(num_class, 2)
        
        self.bn0 = nn.BatchNorm2d(num_class)
        self.linear_1 = nn.Linear(num_class, num_class*2)
        self.bn1 = nn.BatchNorm2d(num_class*2)
        self.linear_2 = nn.Linear(num_class*2, num_class)
        self.bn2 = nn.BatchNorm2d(num_class)

        self.relu = nn.ReLU()        


    def weights_init(self):
        self.dirichlet_linear.weight.data.copy_(torch.eye(self.dirichlet_linear.weight.shape[0]))
        self.dirichlet_linear.bias.data.copy_(torch.zeros(*self.dirichlet_linear.bias.shape))
        pass
    def forward(self, logits, gt):
        logits = logits.permute(0,2,3,1)   
        softmax = torch.nn.Softmax(dim=-1)
        probs = softmax(logits)

        ln_probs = torch.log(probs+1e-16)


        out = self.dirichlet_linear(ln_probs)
        out = self.bn0(out.permute(0,3,1,2))
        out = self.relu(out)
        
        out = self.linear_1(out.permute(0,2,3,1))
        out = self.bn1(out.permute(0,3,1,2))
        out = self.relu(out)
        
        out = self.linear_2(out.permute(0,2,3,1))
        out = self.bn2(out.permute(0,3,1,2))
        out = self.relu(out)       
     
        
        tf_positive = self.binary_linear(out.permute(0,2,3,1))
        _, pred = torch.max(probs, dim=-1)
        
        mask = pred == gt
        
        return  tf_positive.permute(0,3,1,2), mask.long()
        
        
