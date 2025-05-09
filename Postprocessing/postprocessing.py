import sys 
from captum.attr import LRP
from captum.attr._utils.lrp_rules import EpsilonRule, IdentityRule

def confusion_matrix(y_true, y_pred, threshold=0.5):
    """
    Compute confusion matrix for binary classification.
    Args:
        y_true: np.array, true labels
        y_pred: np.array, predicted labels
        threshold: float, threshold for binary classification
    Returns:
        plot of confusion matrix
    """
    from sklearn.metrics import confusion_matrix
    from seaborn import heatmap
    from numpy import sum as npsum
    from numpy import array
    import matplotlib.pyplot as plt

    y_true = (y_true > threshold).astype(int)
    y_pred = (y_pred > threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)/npsum(confusion_matrix(y_true, y_pred))
    fig = heatmap(cm, annot=True, fmt=".2%", cmap='Blues')
    # add true negatives, false positives, false negatives, true positives labels
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    return fig, cm

def heidke_skill_score(y_true, y_pred, threshold=0.5):
    from sklearn.metrics import confusion_matrix
    from numpy import array

    y_true = (y_true > threshold).astype(int)
    y_pred = (y_pred > threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    TN = cm[0,0]
    TP = cm[1,1]
    FN = cm[1,0]
    FP = cm[0,1]

    HSS = (2*(TP*TN - FN*FP))/((TP+FN)*(FN+TN) + (TP+FP)*(FP+TN))

    return HSS

def roc_curve(y_true, y_pred, threshold=0.5):
    """
    Compute ROC curve for binary classification.
    Args:
        y_true: np.array, true labels
        y_pred: np.array, predicted labels
    Returns:
        plot of ROC curve
    """
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    from numpy import round as npround
    
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_ = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='model')
    plt.plot([0, 1], [0, 1], 'k--', label='random')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.annotate(f'AUC: {npround(auc_,2)}', xy=(0.6, 0.4), xycoords='axes fraction', fontsize=12)
    plt.legend()
    return plt, auc_


def lrp_FFNN(model, X, rules_dict = None, device = 'cpu', idx = None, only_true = False, truth = None):
    
    from captum.attr import LRP
    from captum.attr._utils.lrp_rules import EpsilonRule
    import torch
    import torch.nn as nn

    model = model.to(device)
    X = torch.tensor(X).to(device)
    lrp_for_model = LRP(model)
    lrp_of_data = []

    if rules_dict == None:
        rules_dict = {nn.ReLU: EpsilonRule(), nn.Sigmoid: EpsilonRule()}

    if only_true:
        
        for i, month in enumerate(X):
            for layer in model.modules():
                for key, value in rules_dict.items():
                    if isinstance(layer, key):
                        layer.rule = value
            if int(model(X[i]).detach().cpu().numpy() > 0.5) == truth[i]:
                lrp_of_data.append(lrp_for_model.attribute(month))
            if len(lrp_of_data)-1 == idx:
                return lrp_of_data[idx]
    
    else:
        if idx is None:
            for i, month in enumerate(X):
                for layer in model.modules():
                    for key, value in rules_dict.items():
                        if isinstance(layer, key):
                            layer.rule = value
                lrp_of_data.append(lrp_for_model.attribute(month))
        else:
            for layer in model.modules():
                for key, value in rules_dict.items():
                    if isinstance(layer, key):
                        layer.rule = value
            lrp_of_data = lrp_for_model.attribute(X[idx])
    if len(lrp_of_data) == 0:
        raise ValueError('lrp_of_data is empty')
    return lrp_of_data

def lrp_CNN(model, X, rules_dict = None, device = 'cpu', idx = None, only_true = False, truth = None):
    
    from captum.attr import LRP
    from captum.attr._utils.lrp_rules import EpsilonRule
    import torch
    import torch.nn as nn

    model = model.to(device)
    X = torch.tensor(X).to(device)
    lrp_for_model = LRP(model)
    lrp_of_data = []

    if rules_dict == None:
        rules_dict = {nn.ReLU: EpsilonRule(), nn.Sigmoid: EpsilonRule()}

    if only_true:
    
        for layer in model.modules():
            for key, value in rules_dict.items():
                if isinstance(layer, key):
                    layer.rule = value
        
        model.eval()
        model_output = model(X).detach().cpu().numpy()
        truth_mask = torch.where((model_output > 0.5 == truth), True, False)
        if idx is not None:
            truth_mask = truth_mask[idx]
            lrp_of_data = lrp_for_model.attribute(X[truth_mask:truth_mask+1])
            return lrp_of_data
        
        return lrp_for_model.attribute(X[truth_mask])
        
    
    else:
        for layer in model.modules():
            for key, value in rules_dict.items():
                if isinstance(layer, key):
                    layer.rule = value

        if idx is not None:
            lrp_of_data = lrp_for_model.attribute(X[idx:idx+1])
        else:
            lrp_of_data = lrp_for_model.attribute(X)

    return lrp_of_data

def plot_lrp(idx, map_shape, model, X, lon_lat_span, rules_dict = None, device = 'cpu', only_true = False, truth = None, tick_spacing = (5,5)):
    import matplotlib.pyplot as plt
    import torch
    import numpy as np
    import sys 
    sys.path.append('../')
    import models as mod
    if idx == 0 and only_true:
        raise ValueError('Cannot specify idx=0 and only_true at the same time')
    X = torch.tensor(X).to(device)
    model = model.to(device)
    if type(model) == mod.CNN:
        output = lrp_CNN(model, X, rules_dict, device, idx, only_true, truth).detach().cpu().numpy()
        pred = model(X[idx:idx+1]).detach().cpu().numpy() > 0.5
        print(output.shape) 
        print('i flatten the output')
        output = output.flatten()
    else:
        output = lrp_FFNN(model, X, rules_dict, device, idx, only_true, truth).detach().cpu().numpy()

        pred = model(X[idx]).detach().cpu().numpy() > 0.5
    print(output.shape)
    fig, ax = plt.subplots(2,1, figsize=(10, 5), sharex=True)

    ax[0].annotate(f'Prediction: {pred}', xy=(0.3, 1.2), xycoords='axes fraction', fontsize=18)
    ax[0].set_title('Temperature 2 meter')
    ax[0].imshow(output[:np.prod(map_shape)].reshape(map_shape), cmap='coolwarm')
    ax[0].set_ylabel('Latitude')
    ax[0].set_yticklabels(np.arange(lon_lat_span[1][0], lon_lat_span[1][1], tick_spacing[1]))
    ax[0].set_xticklabels(np.arange(lon_lat_span[0][0], lon_lat_span[0][1], tick_spacing[0]))
    # add a colorbar coolwarm centered at 0
    from matplotlib.colors import TwoSlopeNorm 
    print(np.min(output[:np.prod(map_shape)]), np.max(output[:np.prod(map_shape)]))
    print(np.min(output[np.prod(map_shape):]), np.max(output[np.prod(map_shape):]))
    norm = TwoSlopeNorm(vmin=np.min(output[:np.prod(map_shape)]), vcenter=0, vmax=np.max(output[:np.prod(map_shape)]))
    cbar = plt.colorbar(ax[0].imshow(output[:np.prod(map_shape)].reshape(map_shape), cmap='coolwarm', norm=norm), ax=ax[0], orientation='vertical')
    ax[1].set_title('Surface Pressure')
    ax[1].imshow(output[np.prod(map_shape):].reshape(map_shape), cmap='coolwarm')
    norm = TwoSlopeNorm(vmin=np.min(output[np.prod(map_shape):]), vcenter=0, vmax=np.max(output[np.prod(map_shape):]))
    cbar = plt.colorbar(ax[1].imshow(output[np.prod(map_shape):].reshape(map_shape), cmap='coolwarm', norm=norm), ax=ax[1], orientation='vertical')

    ax[0].set_ylabel('Latitude')
    ax[1].set_ylabel('Latitude')
    ax[1].set_xlabel('Longitude')
    
    return fig



from captum.attr._utils.lrp_rules import PropagationRule
import torch

class TopNPercentRule(PropagationRule):
    def __init__(self, N = 95):
        """
        Initializes the rule with a given percentage of relevance to retain.
        
        Parameters:
        - N: Percentage of relevance to retain. Default is 95.
        """
        self.N = N


    def apply(self, relevance_input, relevance_output):
        """
        Filters out elements in relevance_output that are not in the top N% of relevance scores.
        
        Parameters:
        - relevance_input: Input relevance scores.
        - relevance_output: Output relevance scores to be filtered.
        - N: Percentage of relevance to retain. Default is 95.

        Returns:
        - Filtered relevance_output with only top N% relevance scores.
        """
        
        # Flatten the output relevance tensor to find the 50th percentile
        flat_relevance = relevance_output.flatten()
        
        # Determine the relevance threshold for the top 50%
        threshold = torch.quantile(flat_relevance, 1 - self.N/100)
        
        # Create a mask for elements that are below the threshold
        mask = relevance_output >= threshold
        
        # Apply the mask: retain only top 50% of relevance, set the rest to zero
        filtered_relevance_output = relevance_output * mask.float()
        
        return relevance_input, filtered_relevance_output

    def _manipulate_weights(self, module, inputs, outputs) -> None:
        pass


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


def stabilize_relevance(relevance_scores, clip_value=1e-3):
    """Stabilize relevance scores while preserving relative importance"""
    # Normalize by the maximum absolute value
    max_abs = torch.abs(relevance_scores).max()
    if max_abs > 0:
        relevance_scores = relevance_scores / max_abs
    
    # Apply soft thresholding
    mask = torch.abs(relevance_scores) < clip_value
    relevance_scores[mask] = 0
    
    return relevance_scores