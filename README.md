# Universal Novelty Detection Through Adaptive Contrastive Learning

Official PyTorch implementation of
["**Universal Novelty Detection Through Adaptive Contrastive Learning**"]() (CVPR 2024) by
[Hossein Mirzaei](),
[Mojtaba Nafez](),
[Mohammad Jafari](),
[Mohammad Bagher Soltani](),
[Jafar Habibi](),
[Mohammad Sabokrou](),
and [MohammadHossein Rohban]().

<p align="center">
    <img src=figures/method.png width="500"> 
</p>

## 1. Requirements
### Environments
- [torchlars](https://github.com/kakaobrain/torchlars) == 0.1.2 

### Datasets 

For ImageNet-30, please download the following datasets to `~/data`.
* [ImageNet-30-train](https://drive.google.com/file/d/1B5c39Fc3haOPzlehzmpTLz6xLtGyKEy4/view),
[ImageNet-30-test](https://drive.google.com/file/d/13xzVuQMEhSnBRZr-YaaO08coLU2dxAUq/view)

## 2. Training

### Unlabeled one-class & multi-class 
To train unlabeled one-class & multi-class models in the paper, run this command:

```train
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train.py --dataset <DATASET> --model <NETWORK> --mode simclr_CSI --shift_trans_type rotation --batch_size 32 --one_class_idx <One-Class-Index>
```

> Option --one_class_idx denotes the in-distribution of one-class training.
> For multi-class training, set --one_class_idx as None.
> To run SimCLR simply change --mode to simclr.
> Total batch size should be 512 = 4 (GPU) * 32 (--batch_size option) * 4 (cardinality of shifted transformation set). 

### Unlabeled multi-class
To train labeled multi-class model (confidence calibrated classifier) in the paper, run this 

## 3. Evaluation

We provide the checkpoint of the CSI pre-trained model. Download the checkpoint from the following link:
- One-class CIFAR-10: [ResNet-18]()
- One-class MVtecAD: [Wide-Res]()
- multi-class CIFAR-10: [Wide-Res]()
- multi-class MVtecAD: [Wide-Res]()

### Unlabeled one-class
To evaluate my model on unlabeled one-class & multi-class out-of-distribution (OOD) detection setting, run this command:

```eval
python eval.py --mode ood_pre --dataset <DATASET> --model <NETWORK> --ood_score CSI --shift_trans_type rotation --print_score --ood_samples 10 --resize_factor 0.54 --resize_fix --one_class_idx <One-Class-Index> --load_path <MODEL_PATH>
```

> Option --one_class_idx denotes the in-distribution of one-class evaluation.
> For multi-class evaluation, set --one_class_idx as None.
> The resize_factor & resize fix option fix the cropping size of RandomResizedCrop().
> For SimCLR evaluation, change --ood_score to simclr.

### UnLabeled multi-class 
To evaluate my model on labeled multi-class 

## Citation
```
@inproceedings{ND2024unode,
  title={Universal Novelty Detection Through Adaptive Contrastive Learning},
  author={Hossein Mirzaei and Mojtaba Nafez and Mohammad Jafari and Mohammad Bagher Soltani and Jafar Habibi and Mohammad Sabokrou and MohammadHossein Rohban},
  booktitle={Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```
