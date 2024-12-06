import os
import argparse
import random
import torch
from tqdm import tqdm

from captum.attr import IntegratedGradients, GuidedGradCam, LayerGradCam, GuidedBackprop, InputXGradient, Saliency, DeepLift, NoiseTunnel
from torchcam.methods import GradCAM, GradCAMpp, SmoothGradCAMpp, XGradCAM, LayerCAM, ScoreCAM, SSCAM, ISCAM

from utils.log import AverageMeter, ProgressMeter, Summary, accuracy, save_checkpoint
from models.model_wrapper import StandardModel, ViTModel, BcosModel
from models.resnet import resnet18, resnet50, resnet101, resnet152, wide_resnet50_2
from models.vgg import vgg16, vgg16_bn, vgg13, vgg19, vgg11
from models.ViT.ViT_new import vit_base_patch16_224
from models.ViT.ViT_LRP import vit_base_patch16_224 as vit_LRP
from models.bagnets.pytorchnet import bagnet33
from models.xdnns.xfixup_resnet import xfixup_resnet50, fixup_resnet50
from models.xdnns.xvgg import xvgg16
from models.bcos_v2.bcos_resnet import resnet50 as bcos_resnet50
from models.bcos_v2.bcos_resnet import resnet18 as bcos_resnet18

from utils.utils import get_imagenet_loaders, str2bool
from explainers.explainer_wrapper import CaptumAttributionExplainer, CaptumNoiseTunnelAttributionExplainer, TorchcamExplainer, ViTGradCamExplainer, ViTRolloutExplainer, ViTCheferLRPExplainer, BcosExplainer, BagNetExplainer, BcosIGUExplainer, BcosGCExplainer, RiseExplainer
from single_deletion import single_deletion_protocol
from incremental_deletion import incremental_deletion_protocol



parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data_dir', metavar='DIR', default='imagenet',
                    help='path to dataset (default: imagenet)')
parser.add_argument('--model', required=True,
                    choices=['resnet18', 'resnet50', 'resnet101', 'resnet152', 'wide_resnet50_2', 'fixup_resnet50', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg16_bn', 'bagnet9', 'bagnet33', 'x_resnet50', 'vit_base_patch16_224', 'bcos_resnet50', 'bcos_resnet18', 'x_vgg16'],
                    help='model architecture')
parser.add_argument('--explainer', required=True,
                    choices=['Gradient', 'IxG', 'IG', 'IG-U', 'IG-SG', 'IxG-SG', 'IG-SG-SQ', 'IG-SG-VG', 'EG', 'AGI', 'Grad-CAM', 'Grad-CAMpp', 'SG-CAMpp', 'XG-CAM', 'Layer-CAM', 'Score-CAM', 'SS-CAM', 'IS-CAM', 'Rollout', 'CheferLRP', 'Bcos', 'BagNet', 'RISE', 'RISE-U'],
                    help='explainer')
parser.add_argument('--evaluation_protocol', required=True,
                    choices=['accuracy', 'single_deletion', 'incremental_deletion', 'accuracy_train_test_w_wo_patches'],
                    help='evaluation protocol to run')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('-b', '--batch_size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--pretrained', default='False', type=str2bool,
                    help='use pre-trained model')
parser.add_argument('--pretrained_ckpt', type=str, default='none')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use. If None, all GPUs are used')
parser.add_argument('--overwrite', required=False, help='Overwrite result if already exists. If False, evaluation is skipped. Default=False', default='False', type=str2bool)


parser.add_argument('--attribution_transform',
                    choices=['raw', 'abs', 'relu'], default = 'raw',
                    help='transformation applied to attribution')
parser.add_argument('--nr_images', default=-1, type=int,
                    help='number of images to use in the protocol. -1 for all images')

parser.add_argument('--use_softmax', default='False', type=str2bool,
                    help='compute attribution for each permutation new')

# for single_deletion
parser.add_argument('--grid_rows_and_cols', default=4, type=int,
                    help='number of rows and cols in the intervention grid')
parser.add_argument('--sd_baseline', required=False, default='zeros',
                    choices=['zeros', 'blur', 'average', 'random'],
                    help='baseline for perturbation')

# for incremental_deletion (id)
parser.add_argument('--id_baseline', required=False, default='zeros',
                    choices=['zeros', 'blur', 'average', 'random'],
                    help='baseline for perturbation')
parser.add_argument('--id_baseline_gaussian_kernel', default=51, type=int,
                    help='kernel size for Gaussian baseline')
parser.add_argument('--id_baseline_gaussian_sigma', default=41, type=int,
                    help='sigma for Gaussian baseline')
parser.add_argument('--id_steps', default=32, type=int,
                    help='number of steps for the protocol')
parser.add_argument('--id_order', required=False,
                    choices=['ascending', 'descending'],
                    default='ascending',
                    help='selection mode for perturbation')
parser.add_argument('--id_update_attribution', default='False', type=str2bool,
                    help='compute attribution for each permutation new')

def main():
    args = parser.parse_args()
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_loader, val_loader = get_imagenet_loaders(args, shuffle_val=True, train_with_eval_transform=True)

    if args.gpu:
        device = 'cuda:' + str(args.gpu)
    else:
        device = 'cuda'

    #device = 'cpu'

    # create model
    if args.model == 'resnet50':
        model = resnet50(pretrained=args.pretrained)
        model = StandardModel(model, gradcam_target_layer = 'model.layer4', use_softmax=args.use_softmax)
    elif args.model == 'resnet18':
        model = resnet18(pretrained=args.pretrained)
        model = StandardModel(model, gradcam_target_layer = 'model.layer4')
    elif args.model == 'resnet101':
        model = resnet101(pretrained=args.pretrained)
        model = StandardModel(model, gradcam_target_layer = 'model.layer4')
    elif args.model == 'resnet152':
        model = resnet152(pretrained=args.pretrained)
        model = StandardModel(model, gradcam_target_layer = 'model.layer4')
    elif args.model == 'wide_resnet50_2':
        model = wide_resnet50_2(pretrained=args.pretrained)
        model = StandardModel(model, gradcam_target_layer = 'model.layer4')
    elif args.model == 'fixup_resnet50':
        model = fixup_resnet50()
        model = StandardModel(model, gradcam_target_layer = 'model.layer4')
    elif args.model == 'vgg11':
        model = vgg11(pretrained=args.pretrained)
        model = StandardModel(model, gradcam_target_layer = 'model.features')
    elif args.model == 'vgg13':
        model = vgg13(pretrained=args.pretrained)
        model = StandardModel(model, gradcam_target_layer = 'model.features')
    elif args.model == 'vgg16':
        model = vgg16(pretrained=args.pretrained)
        model = StandardModel(model, gradcam_target_layer = 'model.features', use_softmax=args.use_softmax)
    elif args.model == 'vgg19':
        model = vgg19(pretrained=args.pretrained)
        model = StandardModel(model, gradcam_target_layer = 'model.features')
    elif args.model == 'vgg16_bn':
        model = vgg16_bn(pretrained=args.pretrained)
        model = StandardModel(model, gradcam_target_layer = 'model.features')
    elif args.model == 'x_vgg16':
        model = xvgg16()
        model = StandardModel(model, gradcam_target_layer = 'model.features')
    elif args.model == 'bagnet33':
        model = bagnet33(pretrained=args.pretrained)
        model = StandardModel(model)
    elif args.model == 'x_resnet50':
        model = xfixup_resnet50()
        model = StandardModel(model, gradcam_target_layer = 'model.layer4')
    elif args.model == 'bcos_resnet50':
        model = bcos_resnet50(pretrained=args.pretrained, long_version=False)
        model = BcosModel(model)
    elif args.model == 'bcos_resnet18':
        model = bcos_resnet18(pretrained=args.pretrained)
        model = BcosModel(model)
   
    elif args.model == 'vit_base_patch16_224':
        if args.explainer == 'CheferLRP':
            model = vit_LRP(pretrained=args.pretrained)
        else:
            model = vit_base_patch16_224(pretrained=args.pretrained)
        model = ViTModel(model)
    else:
        print('Model not implemented')
    
    if args.pretrained_ckpt != 'none':
        state_dict = torch.load(args.pretrained_ckpt, map_location=torch.device('cpu'))
        if 'state_dict' in state_dict.keys():
            state_dict = state_dict['state_dict']
        
        new_state_dict = state_dict
        if 'module.model.' in list(state_dict.keys())[0]:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith("module.model."):
                    name = k[13:] # remove `model.`
                else:
                    name = k
                new_state_dict[name] = v
        elif 'module.' in list(state_dict.keys())[0]:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
        elif 'model.' in list(state_dict.keys())[0]:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith("model."):
                    name = k[6:] # remove `model.`
                else:
                    name = k
                new_state_dict[name] = v

        model.load_state_dict(new_state_dict)

    model = model.to(device)
    model.eval()

    # create explainer
    if args.explainer == 'IxG':
        explainer = InputXGradient(model)
        explainer = CaptumAttributionExplainer(explainer, attribution_transform=args.attribution_transform)
    elif args.explainer == 'Gradient':
        explainer = Saliency(model)
        explainer = CaptumAttributionExplainer(explainer, attribution_transform=args.attribution_transform)
    elif args.explainer == 'IG':
        explainer = IntegratedGradients(model)
        baseline = torch.zeros((1,3,224,224)).to(device)
        explainer = CaptumAttributionExplainer(explainer, baseline=baseline, attribution_transform=args.attribution_transform)
    elif args.explainer == 'IG-U':
        baseline = torch.rand((1,3,224,224)).to(device) * 2. - 1. # range is -1 to 1 which is approximately image range
        explainer = IntegratedGradients(model)
        explainer = CaptumAttributionExplainer(explainer, baseline=baseline, attribution_transform=args.attribution_transform)
    elif args.explainer == 'IG-SG':
        baseline = torch.zeros((1,3,224,224)).to(device)
        explainer = IntegratedGradients(model)
        explainer = NoiseTunnel(explainer)
        explainer = CaptumNoiseTunnelAttributionExplainer(explainer, baseline=baseline, nt_type='smoothgrad', attribution_transform=args.attribution_transform)
    elif args.explainer == 'IxG-SG':
        explainer = InputXGradient(model)
        explainer = NoiseTunnel(explainer)
        explainer = CaptumNoiseTunnelAttributionExplainer(explainer, nt_type='smoothgrad', attribution_transform=args.attribution_transform)
    elif args.explainer == 'IG-SG-SQ':
        baseline = torch.zeros((1,3,224,224)).to(device)
        explainer = IntegratedGradients(model)
        explainer = NoiseTunnel(explainer)
        explainer = CaptumNoiseTunnelAttributionExplainer(explainer, baseline=baseline, nt_type='smoothgrad_sq', attribution_transform=args.attribution_transform)
    elif args.explainer == 'IG-SG-VG':
        baseline = torch.zeros((1,3,224,224)).to(device)
        explainer = IntegratedGradients(model)
        explainer = NoiseTunnel(explainer)
        explainer = CaptumNoiseTunnelAttributionExplainer(explainer, baseline=baseline, nt_type='vargrad', attribution_transform=args.attribution_transform)
    elif args.explainer == 'Grad-CAM':
        if args.model != 'vit_base_patch16_224':
            explainer = GradCAM(model, target_layer=model.gradcam_target_layer)
            explainer = TorchcamExplainer(explainer, model)
        elif args.model == 'vit_base_patch16_224':
            explainer = ViTGradCamExplainer(model)
    elif args.explainer == 'Grad-CAMpp':
        explainer = GradCAMpp(model, target_layer=model.gradcam_target_layer)
        explainer = TorchcamExplainer(explainer, model)
    elif args.explainer == 'SG-CAMpp':
        explainer = SmoothGradCAMpp(model, target_layer=model.gradcam_target_layer)
        explainer = TorchcamExplainer(explainer, model)
    elif args.explainer == 'XG-CAM':
        explainer = XGradCAM(model, target_layer=model.gradcam_target_layer)
        explainer = TorchcamExplainer(explainer, model)
    elif args.explainer == 'Layer-CAM':
        explainer = LayerCAM(model, target_layer=model.gradcam_target_layer)
        explainer = TorchcamExplainer(explainer, model)
    elif args.explainer == 'Score-CAM':
        explainer = ScoreCAM(model, target_layer=model.gradcam_target_layer)
        explainer = TorchcamExplainer(explainer, model)
    elif args.explainer == 'SS-CAM':
        explainer = SSCAM(model, target_layer=model.gradcam_target_layer)
        explainer = TorchcamExplainer(explainer, model)
    elif args.explainer == 'IS-CAM':
        explainer = ISCAM(model, target_layer=model.gradcam_target_layer)
        explainer = TorchcamExplainer(explainer, model)
    elif args.explainer == 'Rollout':
        explainer = ViTRolloutExplainer(model)
    elif args.explainer == 'CheferLRP':
        explainer = ViTCheferLRPExplainer(model)
    elif args.explainer == 'Bcos':
        explainer = BcosExplainer(model)
    elif args.explainer == 'BagNet':
        explainer = BagNetExplainer(model)
    elif args.explainer == 'RISE':
        assert args.use_softmax == False # make sure the model does not use softmax output because it is used in RISE
        baseline = torch.zeros((1,3,224,224)).to(device)
        explainer = RiseExplainer(model, args.seed, baseline)
    elif args.explainer == 'RISE-U':
        assert args.use_softmax == False # make sure the model does not use softmax output because it is used in RISE
        baseline = torch.rand((1,3,224,224)).to(device) * 2. - 1. # range is -1 to 1 which is approximately image range
        explainer = RiseExplainer(model, args.seed, baseline)
    else:
        print('Explainer not implemented')

    if args.evaluation_protocol == 'single_deletion':
        result = single_deletion_protocol(model, explainer, val_loader, args, device)
        print('Mean rank correlation: ', result)

    elif args.evaluation_protocol == 'incremental_deletion':
        result = incremental_deletion_protocol(model, explainer, val_loader, device, args)
        print('Incremental deletion AUC: ', result)
    
    elif args.evaluation_protocol == 'accuracy':
        
        top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
        top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    
        for images, target, _ in tqdm(val_loader):
            images = images.cuda(device, non_blocking=True)
            target = target.cuda(device, non_blocking=True)

            # compute output
            output = model(images)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            #losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

        result = 'Acc@1: ' + str(round(top1.avg.item(),2)) + ' Acc@5 ' + str(round(top5.avg.item(),4)) 
        print(result)


    

    

if __name__ == '__main__':
    main()