import torch
import torch.nn as nn
from typing import Tuple, Optional


def compute_awq_scaling_factors(
    weight: torch.Tensor, 
    activations: torch.Tensor,
    salient_ratio: float = 0.01,
    alpha: float = 0.5
) -> torch.Tensor:
    """
    Compute AWQ scaling factors based on activation magnitudes.
    
    Args:
        weight: Weight tensor of shape [out_features, in_features]
        activations: Calibration activations of shape [n_samples, seq_len, in_features]
        salient_ratio: Percentage of channels to protect (default: 1%)
        alpha: Smoothing parameter for scaling computation (default: 0.5)
    
    Returns:
        scaling_factors: Per-channel scaling factors of shape [in_features]
    """
    # Compute activation scales (maximum absolute values per channel)
    act_scales = torch.max(torch.abs(activations), dim=(0, 1))[0]  # [in_features]
    
    # Compute weight scales
    weight_scales = torch.max(torch.abs(weight), dim=0)[0]  # [in_features]
    
    # Identify salient channels based on activation magnitude
    n_salient = max(1, int(salient_ratio * weight.shape[1]))
    _, salient_indices = torch.topk(act_scales, n_salient)
    
    # Initialize scaling factors
    scaling_factors = torch.ones_like(weight_scales)
    
    # Compute scaling for salient channels using AWQ formula
    # s = (act_scale^alpha) / (weight_scale^(1-alpha))
    salient_act_scales = act_scales[salient_indices]
    salient_weight_scales = weight_scales[salient_indices] 
    
    # Avoid division by zero
    salient_weight_scales = torch.clamp(salient_weight_scales, min=1e-8)
    
    # AWQ scaling formula
    salient_scaling = torch.pow(salient_act_scales, alpha) / torch.pow(salient_weight_scales, 1 - alpha)
    
    # Normalize scaling factors to avoid extreme values
    salient_scaling = torch.clamp(salient_scaling, min=0.1, max=10.0)
    
    # Apply scaling to salient channels
    scaling_factors[salient_indices] = salient_scaling
    
    return scaling_factors


def apply_awq_scaling(
    weight: torch.Tensor,
    scaling_factors: torch.Tensor,
    activations: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply AWQ scaling to weights and optionally to activations.
    
    Args:
        weight: Original weight tensor [out_features, in_features]
        scaling_factors: Per-channel scaling factors [in_features]
        activations: Optional activations to scale inversely
    
    Returns:
        scaled_weight: Scaled weight tensor
        scaled_activations: Scaled activations (if provided)
    """
    # Scale weights: W' = W * s
    scaled_weight = weight * scaling_factors.unsqueeze(0)
    
    # Scale activations inversely if provided: X' = X / s
    scaled_activations = None
    if activations is not None:
        scaled_activations = activations / scaling_factors.unsqueeze(0).unsqueeze(0)
    
    return scaled_weight, scaled_activations


def compute_qparams_awq(
    layer: nn.Module,
    calibration_data: torch.Tensor,
    quantizer,
    opts: dict,
    search: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute quantization parameters using AWQ-informed approach.
    
    Args:
        layer: The layer to quantize
        calibration_data: Calibration activations
        quantizer: MinMaxQuantizer instance
        opts: Quantization options
        search: Whether to perform grid search for optimal parameters
    
    Returns:
        scale: Quantization scale
        zero: Quantization zero point
        scaling_factors: AWQ scaling factors applied
    """
    weight = layer.weight.data.float()
    
    # Step 1-2: Compute AWQ scaling factors
    salient_ratio = opts.get('awq_salient_ratio', 0.01)
    alpha = opts.get('awq_alpha', 0.5)
    
    scaling_factors = compute_awq_scaling_factors(
        weight, 
        calibration_data,
        salient_ratio=salient_ratio,
        alpha=alpha
    )
    
    # Apply scaling to weights
    scaled_weight, _ = apply_awq_scaling(weight, scaling_factors)
    
    # Step 3: Compute initial quantization parameters for scaled weights
    if search:
        # Use MMSE for initial parameters
        scale, zero = quantizer.find_params_H(scaled_weight, None, search=True)
    else:
        # Use MinMax for initial parameters
        scale, zero = quantizer.find_params_H(scaled_weight, None, search=False)
    
    return scale, zero, scaling_factors


class AWQPreprocessor:
    """
    AWQ preprocessor for integration with BOA quantization flow.
    """
    
    def __init__(self, opts: dict):
        self.salient_ratio = opts.get('awq_salient_ratio', 0.01)
        self.alpha = opts.get('awq_alpha', 0.5)
        self.scaling_factors_cache = {}
    
    def compute_scaling_factors(
        self,
        layer_name: str,
        weight: torch.Tensor,
        calibration_data: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute and cache AWQ scaling factors for a layer.
        """
        if layer_name in self.scaling_factors_cache:
            return self.scaling_factors_cache[layer_name]
        
        scaling_factors = compute_awq_scaling_factors(
            weight,
            calibration_data,
            self.salient_ratio,
            self.alpha
        )
        
        self.scaling_factors_cache[layer_name] = scaling_factors
        return scaling_factors
    
    def apply_scaling(
        self,
        weight: torch.Tensor,
        scaling_factors: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply AWQ scaling to weights.
        """
        scaled_weight, _ = apply_awq_scaling(weight, scaling_factors)
        return scaled_weight
    
    def clear_cache(self):
        """
        Clear the scaling factors cache.
        """
        self.scaling_factors_cache.clear()