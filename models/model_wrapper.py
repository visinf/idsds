import torch.nn as nn
from abc import abstractmethod

class ModelExplainerWrapper:

    def __init__(self, model, explainer):
        """
        A generic wrapper that takes any model and any explainer to putput model predictions 
        and explanations that highlight important input image part.
        Args:
            model: PyTorch neural network model
            explainer: PyTorch model explainer    
        """
        self.model = model
        self.explainer = explainer

    def predict(self, input):
        return self.model.forward(input)

    def explain(self, input):
        return self.explainer.explain(self.model, input)


class AbstractModel(nn.Module):
    def __init__(self, model, gradcam_target_layer='None', use_softmax=False):
        """
        An abstract wrapper for PyTorch models implementing functions required for evaluation.
        Args:
            model: PyTorch neural network model
        """
        super().__init__()
        self.model = model
        self.gradcam_target_layer = gradcam_target_layer
        self.use_softmax = use_softmax

    @abstractmethod
    def forward(self, input):
        return self.model

class StandardModel(AbstractModel):
    """
    A wrapper for standard PyTorch models (e.g. ResNet, VGG, AlexNet, ...).
    Args:
        model: PyTorch neural network model
    """

    def forward(self, input):
        
        if not self.use_softmax:
            return self.model(input) # w/o softmax
        elif self.use_softmax:
            m = nn.Softmax(dim=1)
            return m(self.model(input)) # w/ softmax

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

import torch
import torchvision.transforms as T
class ViTModel(AbstractModel):
    """
    A wrapper for standard PyTorch models (e.g. ResNet, VGG, AlexNet, ...).
    Args:
        model: PyTorch neural network model
    """

    def forward(self, input):
        #device = input.device
        #B,C,H,W = input.shape
        #assert B == 1 # else to PIL is not working
        #to_pil = T.ToPILImage()
        #res = T.Resize((224,224))
        #to_tensor = T.ToTensor()
        #input = to_tensor(res(to_pil(input[0])))
        #input = input.unsqueeze(0).to(device)
        #input = torch.nn.functional.interpolate(input, (224,224)) # this slightly changes the accuracy but doesn't matter too much

        return self.model(input)

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)


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

class BcosModel(AbstractModel):
    """
    A wrapper for Bcos models.
    Args:
        model: PyTorch bcos model
    """

    def forward(self, input):
        _input = Variable(AddInverse()(input), requires_grad=True)
        return self.model(_input)

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
