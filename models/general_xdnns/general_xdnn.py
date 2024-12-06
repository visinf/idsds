import torch
import torch.nn as nn

class GeneralEfficientlyExplainableNN(nn.Module):
    def __init__(
        self,
        model: nn.Module
    ) -> None:
        super(GeneralEfficientlyExplainableNN, self).__init__()
        self.model = model
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_magnitude = torch.norm(x.flatten(1), p='fro', dim=(-1)).unsqueeze(1)
        B,C,H,W = x.shape
        x_flat = x.flatten(1)
        x_unit = x_flat / (x_magnitude + 1e-8)
        x_unit = x_unit.view(B,C,H,W)
        y = x_magnitude * self.model(x_unit)
        return y