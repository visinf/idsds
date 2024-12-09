# Benchmarking the Attribution Quality of Vision Models

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)

[R. Hesse](https://robinhesse.github.io/), [S. Schaub-Meyer](https://schaubsi.github.io/), and [S. Roth](https://www.visinf.tu-darmstadt.de/visual_inference/people_vi/stefan_roth.en.jsp). **Benchmarking the Attribution Quality of Vision Models**. _NeurIPS Datasets and Benchmarks Track_, 2024.

[Paper](https://openreview.net/pdf?id=XmyxQaTyck) | [ArXiv](https://arxiv.org/abs/2407.11910) | [Video (TBD)](TBD) | [Poster](https://github.com/visinf/idsds/blob/main/poster.jpeg)

## Evaluation

### In-Domain Single Deletion Score

To evaluate an attribution method on a model, you can run the following command:

```
python evaluate.py --evaluation_protocol single_deletion --grid_rows_and_cols 4 --data_dir /datasets/imagenet --model your_model --pretrained True --pretrained_ckpt /data/rhesse/phd_imagenet_patches_evaluation/checkpoints/resnet50_imagenet1000_lr0.001_epochs30_step10_osdi_checkpoint_best.pth.tar --explainer your_explainer --batch_size 128
```

You just need to specify the ```--data_dir``` (the path to ImageNet), the ```--model``` (e.g., ```resnet50```,```vgg16```, or ```vit_base_patch16_224```), the ```--pretrained_ckpt``` (see table with model weights), and the ```--explainer``` (see table with attribution methods). Please see the code in ```evaluate.py``` for additional details for all parameters.

### Single Deletion Score

The single deletion protocol works similar to our proposed in-domain single deletion score but without fine-tuning the model. Thus, you can use the same commands as before, simply specifying ```--pretrained True --pretrained_ckpt none```. For models that do not automatically download the pre-trained ImageNet weights you need to download them and specify them with ```--pretrained_ckpt path_to_weights```. 

### Incremental Deletion Score

To run the indremental deletion protocol, run the following command:

```
python evaluate.py --evaluation_protocol incremental_deletion --id_baseline zeros --id_steps 32 --id_order ascending --data_dir /fastdata/rhesse/datasets/imagenet --model your_model --workers 0 --pretrained True --pretrained_ckpt none --seed 0 --nr_images 4096 --explainer your_explainer --batch_size 128
```

Set ```--id_update_attribution True``` to update the attribution in each deletion step.

## Attribution Methods

The following table provides an overview of the evaluated attribution methods and how to set the parameters to use them. If you want to add your own attribution method, you need to add the method in ```/explainers```, implement your explainer wrapper in ```/explainers/explainer_wrapper.py```, add the name of you method to the parameter list in ```evaluate.py```, and instanciate your attribution method in ```evaluate.py``` when its name is used as the explainer parameter.

| Parameter | Name |
| --- | --- |
| `--explainer IxG ` | InputXGradient |
| `--explainer IxG-SG` | InputXGradient + SmoothGrad |
| `--explainer IG` | Integrated Gradients (zero baseline) |
| `--explainer IG-U` | Integrated Gradients (uniform baseline) |
| `--explainer IG-SG` | Integrated Gradients (zero baseline) + SmoothGrad |
| `--explainer IxG --attribution_transform abs` | InputXGradient (absolute) |
| `--explainer IxG-SG --attribution_transform abs` | InputXGradient + SmoothGrad (absolute) |
| `--explainer IG --attribution_transform abs` | Integrated Gradients (zero baseline) (absolute) |
| `--explainer IG-U --attribution_transform abs` | Integrated Gradients (uniform baseline) (absolute) |
| `--explainer IG-SG --attribution_transform abs` | Integrated Gradients (zero baseline) + SmoothGrad (absolute)|
| `--explainer IG-SG-SQ --attribution_transform abs` | Integrated Gradients (zero baseline) + SmoothGrad (squared) |
| `--explainer RISE --attribution_transform abs` | RISE (zero baseline) |
| `--explainer RISE-U --attribution_transform abs` | RISE (uniform baseline) |
| `--explainer Grad-CAM` | Grad-CAM (CNN only) |
| `--explainer Grad-CAMpp` | Grad-CAM++ (CNN only) |
| `--explainer SG-CAMpp` | Grad-CAM++ + SmoothGrad (CNN only) |
| `--explainer XG-CAM` | XGrad-CAM (CNN only) |
| `--explainer Layer-CAM` | Layer-CAM (CNN only) |
| `--explainer Rollout` | Rollout (ViT only) |
| `--explainer CheferLRP` | CheferLRP (ViT only) |
| `--explainer Bcos` | Bcos (Bcos only) |
| `--explainer BagNet` | BagNet (BagNet only) |



## Models

If you want to add your own models, you need to add the model file in ```/models```, implement your model wrapper in ```/models/model_wrapper.py```, add the name of you model to the parameter list in ```evaluate.py```, and load your model in ```evaluate.py``` when that name is used as model parameter.

### Downloads

The following table provides the model weights for the fine-tuned models needed for our in-domain single deletion score.

| Model | Parameter | Download |
| --- | --- | --- |
| ResNet-18 | `--model resnet18` | [resnet18](https://download.visinf.tu-darmstadt.de/data/2024-neurips-hesse-idsds/models_finetuned/resnet18_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar) |
| ResNet-50 | `--model resnet50` | [resnet50](https://download.visinf.tu-darmstadt.de/data/2024-neurips-hesse-idsds/models_finetuned/resnet50_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar) |
| ResNet-101 | `--model resnet101` | [resnet101](https://download.visinf.tu-darmstadt.de/data/2024-neurips-hesse-idsds/models_finetuned/resnet101_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar) |
| ResNet-152 | `--model resnet152` | [resnet152](https://download.visinf.tu-darmstadt.de/data/2024-neurips-hesse-idsds/models_finetuned/resnet152_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar) |
| Wide ResNet-50 | `--model wide_resnet50_2` | [wide_resnet50_2](https://download.visinf.tu-darmstadt.de/data/2024-neurips-hesse-idsds/models_finetuned/wide_resnet50_2_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar) |
| ResNet-50 w/o BatchNorm | `--model fixup_resnet50` | [fixup_resnet50](https://download.visinf.tu-darmstadt.de/data/2024-neurips-hesse-idsds/models_finetuned/fixup_resnet50_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar) |
| ResNet-50 w/o BatchNorm w/o bias | `--model x_resnet50` | [x_resnet50](https://download.visinf.tu-darmstadt.de/data/2024-neurips-hesse-idsds/models_finetuned/xresnet50_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar) |
| VGG-11 | `--model vgg11` | [vgg11](https://download.visinf.tu-darmstadt.de/data/2024-neurips-hesse-idsds/models_finetuned/vgg11_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar) |
| VGG-13 | `--model vgg13` | [vgg13](https://download.visinf.tu-darmstadt.de/data/2024-neurips-hesse-idsds/models_finetuned/vgg13_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar) |
| VGG-16 | `--model vgg16` | [vgg16](https://download.visinf.tu-darmstadt.de/data/2024-neurips-hesse-idsds/models_finetuned/vgg16_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar) |
| VGG-19 | `--model vgg19` | [vgg19](https://download.visinf.tu-darmstadt.de/data/2024-neurips-hesse-idsds/models_finetuned/vgg19_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar) |
| VGG-16 w/ BatchNorm | `--model vgg16_bn` | [vgg16_bn](https://download.visinf.tu-darmstadt.de/data/2024-neurips-hesse-idsds/models_finetuned/vgg16_bn_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar) |
| VGG-16 w/o BatchNorm w/o bias | `--model x_vgg16` | [x_vgg16](https://download.visinf.tu-darmstadt.de/data/2024-neurips-hesse-idsds/models_finetuned/xvgg16_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar) |
| ViT-B-16 | `--model vit_base_patch16_224` | [vit_base_patch16_224](https://download.visinf.tu-darmstadt.de/data/2024-neurips-hesse-idsds/models_finetuned/vit_base_patch16_224_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar) |
| Bcos-ResNet-50 | `--model bcos_resnet50` | [bcos_resnet50](https://download.visinf.tu-darmstadt.de/data/2024-neurips-hesse-idsds/models_finetuned/bcos_resnet50_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar) |
| BagNet-33 | `--model bagnet33` | [bagnet33](https://download.visinf.tu-darmstadt.de/data/2024-neurips-hesse-idsds/models_finetuned/bagnet33_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar) |

### Training

If you want to train your own models, you can follow the steps described in the beginning of this section and run the following command:

```
python train.py --data_dir /datasets/imagenet --model your_model --lr 0.001 --pretrained --store_path /checkpoints/ --checkpoint_prefix your_model_imagenet1000_lr0.001_epochs30_step10
```

## Citation

If you find our work helpful, please consider citing
```
@inproceedings{Hesse:2024:IDSDS,
  title     = {Benchmarking the Attribution Quality of Vision Models},
  author    = {Hesse, Robin and Schaub-Meyer, Simone and Roth, Stefan},
  booktitle = {NeurIPS},
  year      = {2024},
}
```
