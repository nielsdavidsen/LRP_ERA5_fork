from captum.attr._utils.lrp_rules import EpsilonRule
from captum.attr import LRP
import torch
import torch.nn as nn
import numpy as np

#### Borrowed from Melcher ####
class StableBatchNormRule(EpsilonRule):
    def __init__(self, epsilon=1e-6, stabilizer=1e-3):
        super().__init__(epsilon)
        self.stabilizer = stabilizer
    
    def forward_hook(self, module, inputs, outputs):
        """Store normalized inputs for better relevance propagation"""
        self.normalized_input = (inputs[0] - module.running_mean[None, :]) / torch.sqrt(module.running_var[None, :] + module.eps)
        return super().forward_hook(module, inputs, outputs)
    
    def backward_hook(self, module, grad_input, grad_output):
        """Stabilize the gradient propagation"""
        # Apply stabilization to prevent explosion
        grad_modified = grad_input[0] / (torch.norm(grad_input[0], dim=1, keepdim=True) + self.stabilizer)
        return (grad_modified,) + grad_input[1:]
    
rules_dict = {
    nn.ReLU: EpsilonRule(epsilon=1e-6), 
    nn.Sigmoid: EpsilonRule(epsilon=1e-6), 
    nn.ELU: EpsilonRule(epsilon=1e-6),
    nn.BatchNorm1d: StableBatchNormRule(epsilon=1e-6, stabilizer=1e-3)
}
##############################


def lrp_calc(model, input_tensor, test_shape=None):
    from captum.attr import LRP

    input_tensor = input_tensor.clone().detach().requires_grad_(True)  # Ensure input is detached from the graph

    # Applying rules to the model
    for layer in model.modules():
        for key, value in rules_dict.items():
            if isinstance(layer, key):
                layer.rule = value

    lrp = LRP(model)
    attributions = lrp.attribute(input_tensor)
    attributions = attributions.detach().cpu().numpy()  # Convert to numpy array

    if test_shape is not None:
        return attributions, np.sum(attributions.reshape(test_shape), axis=0)
    else:
        return attributions