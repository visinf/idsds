import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod
from captum.attr import LayerAttribution
from torchvision import transforms
from PIL import Image

class AbstractExplainer():
    def __init__(self, explainer, attribution_transform='raw', baseline = None):
        """
        An abstract wrapper for explanations.
        Args:
            model: PyTorch neural network model
        """
        self.explainer = explainer
        self.explainer_name = type(self.explainer).__name__
        self.baseline = baseline
        self.attribution_transform = attribution_transform

    @abstractmethod
    def explain(self, input):
        return self.explainer.explain(self.model, input)


class AbstractAttributionExplainer(AbstractExplainer):
    
    @abstractmethod
    def explain(self, input):
        return self.explainer.explain(self.model, input)
    
class CaptumAttributionExplainer(AbstractAttributionExplainer):
    
    """
    A wrapper for Captum attribution methods.
    Args:
        explainer: Captum explanation method
    """
    def explain(self, input, target=None, baseline=None):
        if self.explainer_name == 'Saliency':
            attr = self.explainer.attribute(input, target=target, abs=False)
        elif self.explainer_name == 'InputXGradient': 
            attr = self.explainer.attribute(input, target=target) 
        elif self.explainer_name == 'LayerGradCam': 
            B,C,H,W = input.shape
            attr = self.explainer.attribute(input, target=target, relu_attributions=True)
            m = transforms.Resize((H,W), interpolation=Image.NEAREST)
            attr = m(attr)

        elif self.explainer_name == 'IntegratedGradients':
            attr = self.explainer.attribute(input, target=target, baselines=self.baseline, n_steps=50)

        attr = attr.sum(dim=1, keepdim=True)

        if self.attribution_transform == 'raw':
            attr = attr
        elif self.attribution_transform == 'abs':
            attr = attr.abs()
        elif self.attribution_transform == 'relu':
            m = nn.ReLU()
            attr = m(attr)

        return attr

        
class CaptumNoiseTunnelAttributionExplainer(AbstractAttributionExplainer):
    
    """
    A wrapper for Captum attribution methods.
    Args:
        explainer: Captum explanation method
    """
    def __init__(self, explainer, baseline = None, nt_type='smoothgrad', attribution_transform='raw'):
        """
        An abstract wrapper for explanations.
        Args:
            model: PyTorch neural network model
        """
        self.explainer = explainer
        self.explainer_name = type(self.explainer.attribution_method).__name__
        self.baseline = baseline
        self.nt_type = nt_type
        self.nt_samples = 4
        self.attribution_transform = attribution_transform


    def explain(self, input, target=None, baseline=None):
        
        if self.explainer_name == 'Saliency' or self.explainer_name == 'InputXGradient': 
            attr= self.explainer.attribute(input, target=target, nt_type=self.nt_type, nt_samples=self.nt_samples)
            
        elif self.explainer_name == 'IntegratedGradients':
            attr = self.explainer.attribute(input, target=target, baselines=self.baseline, n_steps=50, nt_type=self.nt_type, nt_samples=self.nt_samples)
            
        attr = attr.sum(dim=1, keepdim=True)

        if self.attribution_transform == 'raw':
            attr = attr
        elif self.attribution_transform == 'abs':
            attr = attr.abs()
        elif self.attribution_transform == 'relu':
            m = nn.ReLU()
            attr = m(attr)

        return attr
    
class TorchcamExplainer(AbstractAttributionExplainer):
    
    """
    A wrapper for Captum attribution methods.
    Args:
        explainer: Captum explanation method
    """
    def __init__(self, explainer, model):
        """
        An abstract wrapper for explanations.
        Args:
            model: PyTorch neural network model
        """
        self.explainer = explainer
        self.model = model
        #self.explainer_name = type(self.explainer).__name__
        self.resizer = transforms.Resize((224,224), interpolation=Image.BILINEAR)
        #self.baseline = baseline
        #self.nt_type = nt_type
        #self.nt_samples = 10
        #print(self.explainer_name)


    def explain(self, input, target=None, baseline=None):
        B,C,H,W = input.shape
        cams_for_batch = []
        for b_idx in range(B):
            out = self.model(input[b_idx].unsqueeze(0))
            cams = self.explainer(target[b_idx].item(), out)
            assert len(cams) == 1
            cam = cams[0]#.unsqueeze(0).unsqueeze(0)
            #B, C, H, W = cam.shape
            #assert C == 1
            #cam = cam[0,0,:,:] # remove channel for resizing
            cam = self.resizer(cam)
            cam = cam.unsqueeze(0)
            #print(cam.shape)
            cams_for_batch.append(cam)
        return torch.cat(cams_for_batch, dim = 0)

from models.ViT.ViT_explanation_generator import Baselines, LRP
class ViTGradCamExplainer(AbstractAttributionExplainer):
    def __init__(self, model):
        self.model = model
        self.explainer = Baselines(self.model.model)

    def explain(self, input, target):
        B,C,H,W = input.shape
        assert B == 1
        input_ = torch.nn.functional.interpolate(input, (224,224))
        attribution = self.explainer.generate_cam_attn(input_, index=target).reshape(1, 1, 14, 14)
        m = transforms.Resize((H,W), interpolation=Image.NEAREST)
        attribution = m(attribution)
        return attribution

class ViTRolloutExplainer(AbstractAttributionExplainer):
    def __init__(self, model):
        self.model = model
        self.explainer = Baselines(self.model.model)

    def explain(self, input, target):
        B,C,H,W = input.shape
        assert B == 1
        input_ = torch.nn.functional.interpolate(input, (224,224))
        attribution = self.explainer.generate_rollout(input_, start_layer=1).reshape(1, 1, 14, 14)
        m = transforms.Resize((H,W), interpolation=Image.NEAREST)
        attribution = m(attribution)
        return attribution

class ViTCheferLRPExplainer(AbstractAttributionExplainer):
    def __init__(self, model):
        self.model = model
        self.explainer = LRP(self.model.model)

    def explain(self, input, target):
        B,C,H,W = input.shape
        assert B == 1
        input_ = torch.nn.functional.interpolate(input, (224,224))
        attribution = self.explainer.generate_LRP(input_, index=target, start_layer=1, method="transformer_attribution").reshape(1, 1, 14, 14)
        m = transforms.Resize((H,W), interpolation=Image.NEAREST)
        attribution = m(attribution)
        return attribution

def explanation_mode(model, active=True):
    for mod in model.modules():
        if hasattr(mod, "explanation_mode"):
            mod.explanation_mode(active)

class AddInverse(nn.Module):

    def __init__(self, dim=1):
        """
            Adds (1-in_tensor) as additional channels to its input via torch.cat().
            Can be used for images to give all spatial locations the same sum over the channels to reduce color bias.
        """
        super().__init__()
        self.dim = dim

    def forward(self, in_tensor):
        out = torch.cat([in_tensor, 1-in_tensor], self.dim)
        return out

from torch.autograd import Variable
class BcosExplainer(AbstractAttributionExplainer):
    
    def __init__(self, model):
        """
        An explainer for bcos explanations.
        Args:
            model: PyTorch neural network model
        """
        self.model = model

    def explain(self, input, target):
        
        #explanation_mode(self.model, True)
        #print(self.model.model.)
        #output = self.model(input) # if directly using self.model, the gradient computation does not work
        #target = output[0][target]
        #input.grad = None
        #target[0].backward(retain_graph=True)
        #w1 = input.grad
        #attribution = w1 * input
        
        assert input.shape[0] == 1 # batch size = 1
        model = self.model.model
        with model.explanation_mode():
        
            _input = Variable(AddInverse()(input), requires_grad=True)  # not sure if this should be here or rather in the model wrapper.      
            output = model(_input) # if directly using self.model it returns None
            #model.explanation_mode()
            target = output[0][target]
            _input.grad = None
            target[0].backward(retain_graph=True)
            w1 = _input.grad
            attribution = w1 * _input
            attribution = attribution.sum(dim=1, keepdim=True)

        #self.model.model.explanation_mode()

        return attribution

from captum.attr import IntegratedGradients, GuidedGradCam, LayerGradCam, GuidedBackprop, InputXGradient, Saliency, DeepLift, NoiseTunnel
class BcosIGUExplainer(AbstractAttributionExplainer):
    
    def __init__(self, model):
        """
        An abstract wrapper for explanations.
        Args:
            model: PyTorch neural network model
        """
        self.model = model
    
    def explain(self, input, target=None):
        
        assert input.shape[0] == 1 # batch size = 1
        model = self.model.model

        baseline = torch.rand((1,6,224,224)).to(input.device) * 2. - 1. # range is -1 to 1 which is approximately image range
        #baseline = torch.zeros((1,6,224,224)).to(input.device)
        
        explainer = IntegratedGradients(model)
        #with model.explanation_mode(): # this seems to make it worse
        
        _input = Variable(AddInverse()(input), requires_grad=True)  # not sure if this should be here or rather in the model wrapper.      
        attr = explainer.attribute(_input, target=target, baselines=baseline, n_steps=50)
        attr = attr.sum(dim=1, keepdim=True)
        return attr

from torchcam.methods import GradCAM, GradCAMpp, SmoothGradCAMpp, XGradCAM, LayerCAM, ScoreCAM, SSCAM, ISCAM
class BcosGCExplainer(AbstractAttributionExplainer):
    
    def __init__(self, model):
        """
        An abstract wrapper for explanations.
        Args:
            model: PyTorch neural network model
        """
        self.model = model
        self.resizer = transforms.Resize((224,224), interpolation=Image.BILINEAR)
    
    def explain(self, input, target=None):
        
        assert input.shape[0] == 1 # batch size = 1
        model = self.model.model
        _input = Variable(AddInverse()(input), requires_grad=True)  # not sure if this should be here or rather in the model wrapper.      
        
        explainer = GradCAM(model, target_layer='layer4')
        B,C,H,W = input.shape
        cams_for_batch = []
        #with model.explanation_mode(): # this seems to make it worse
        for b_idx in range(B):
            out = model(_input[b_idx].unsqueeze(0))
            cams = explainer(target[b_idx].item(), out)
            assert len(cams) == 1
            cam = cams[0].unsqueeze(0).unsqueeze(0)
            cam = self.resizer(cam)
            cams_for_batch.append(cam)
        return torch.cat(cams_for_batch, dim = 0)
    
        #
        
        _input = Variable(AddInverse()(input), requires_grad=True)  # not sure if this should be here or rather in the model wrapper.      
        attr = explainer.attribute(_input, target=target, baselines=baseline, n_steps=50)
        attr = attr.sum(dim=1, keepdim=True)
        return attr
        
from models.bagnets.utils import generate_heatmap_pytorch
from models.bagnets.utils import generate_heatmap_pytorch2

class BagNetExplainer(AbstractAttributionExplainer):
    """
    A wrapper for LIME.
    https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20images%20-%20Pytorch.ipynb
    Args:
        model: PyTorch model.
    """
    def __init__(self, model):
        """
        An abstract wrapper for explanations.
        Args:
            model: PyTorch neural network model
        """
        self.model = model

    def explain(self, input, target):
        assert input.shape[0] == 1
        attribution_numpy = generate_heatmap_pytorch(self.model, input.cpu(), target, 33)
        attribution = torch.from_numpy(attribution_numpy).unsqueeze(0).unsqueeze(0).to(input.device)
        return attribution
    
from .rise import RISE
import os 
class RiseExplainer(AbstractAttributionExplainer):
    """
    A wrapper for RISE.
    Args:
        model: PyTorch model.
    """
    def __init__(self, model, seed, baseline):
        """
        An abstract wrapper for explanations.
        Args:
            model: PyTorch neural network model
        """
        self.explainer = RISE(model, (224, 224), baseline, gpu_batch=128)
        # Generate masks for RISE or use the saved ones.
        self.seed = seed
        maskspath = '/fastdata/rhesse/rise_masks_phd_imagenet_patches_evaluation_seed' + str(self.seed) + '.npy'
        generate_new = False

        if generate_new or not os.path.isfile(maskspath):
            self.explainer.generate_masks(N=1000, s=8, p1=0.1, savepath=maskspath)
            print('Masks are generated.')
        else:
            self.explainer.load_masks(maskspath, p1=0.1)
            print('Masks are loaded.')

    def explain(self, input, target=None):
        assert input.shape[0] == 1
        attribution = self.explainer(input)
        attribution = attribution[target[0].int()].unsqueeze(0).unsqueeze(0)
        
        return attribution