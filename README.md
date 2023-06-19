
# On Calibrating Semantic Segmentation Models: Analyses and An Algorithm (CVPR 2023)
[\[Paper\]](https://arxiv.org/pdf/2212.12053.pdf)

We provide a systematic study on the calibration of semantic segmentation models and propose a simple yet effective approach of selective scaling. Source code is released for selective scaling. Common questions could be discussed in the issues.

# Miscalibration Obeservation for Semantic Segmentation Models 
![alt text][miscalibration]

[miscalibration]: https://github.com/dwang181/selectivecal/blob/main/Figures/miscalibration.PNG

# Requirements

1. Install [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/blob/main/README.md) from OpenMMLab.
 

# Quick reference

The general implantation framework is as follows:
1.	Group validation images into calibrator training/validation/testing sets.
2.	Label all pixels given the predictive correctness. 
3.	Train a binary classifier with training set and hyperparameters are tuned with validation set. Here, the training data pair is prepared with the predictive probability and correctness label. 
4.	Evaluate testing performance. Given the classifier’s prediction, a separate scaling on logits (before Softmax) is conducted. Correctly predicted pixels are scaled with 1 (i.e. non-scaling), while mispredictions are scaled with a larger temperature T2. When the classifier is more accurate, the temperature T2 can be more aggressively large, like 1e10. When the classifier’s correctness is moderately low, it is suggested that the temperature is around 2.
5.	Note that background pixels with 255 or -1 or other index number are excluded from any set.
The general setting: batch-size is 20 (i.e. 20 pixel-based probability vectors) and the optimizer is the default AdamW with weight decay of 1e-6. The training epoch is 40 and the best validated one is selected for evaluation. The training/validation/testing separation is randomly separated based upon random shuffling. So the performance may vary, but it should be not much.


## Citation
```yaml
@article{wang2022calibrating,
  title={On Calibrating Semantic Segmentation Models: Analyses and An Algorithm},
  author={Wang, Dongdong and Gong, Boqing and Wang, Liqiang},
  journal={arXiv preprint arXiv:2212.12053},
  year={2022}
}
```
