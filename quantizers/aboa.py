import torch
import torch.nn as nn
from quantizers.boa import BoA
from quantizers.awq import AWQPreprocessor, apply_awq_scaling
from quantizers.utils import get_cholesky_of_inverse, reorder_col, reverse_reorder_col, reorder_row, reverse_reorder_row
from utils.quant_utils import fake_quantize, filter_dead_neuron, damping
from utils.utils import cleanup_memory


class ABoA(BoA):
    """
    AWQ-Informed BOA (A-BOA) quantizer that combines AWQ's activation-aware
    scaling with BOA's Hessian-based optimization.
    """
    
    def __init__(self, layer, opts, hyperparams):
        super().__init__(layer, opts, hyperparams)
        
        # Initialize AWQ preprocessor
        self.awq_preprocessor = AWQPreprocessor(opts) if opts.get('use_awq', False) else None
        self.calibration_data = None
        self.scaling_factors = None
    
    def set_calibration_data(self, calibration_data):
        """
        Set calibration data for AWQ scaling computation.
        
        Args:
            calibration_data: Tensor of shape [n_samples, seq_len, in_features]
        """
        self.calibration_data = calibration_data
    
    def quant(self, print_memory_usage=False):
        """
        Perform AWQ-informed BOA quantization.
        """
        assert self.quantizer is not None, "Quantizer should be defined first."
        assert self.H_col is not None, "Hessian should be computed first."
        
        W, H_col, H_row = self.preprocess()
        
        # Apply AWQ scaling if enabled
        if self.awq_preprocessor is not None and self.calibration_data is not None:
            # Compute AWQ scaling factors
            layer_name = getattr(self.layer, 'name', str(id(self.layer)))
            original_W = W.clone()
            
            # Reshape weight for AWQ computation
            n_heads = H_col.shape[0] if len(H_col.shape) > 2 else 1
            hidden_size = W.shape[-1]
            W_2d = W.view(-1, hidden_size)  # [out_features, in_features]
            
            # Compute scaling factors
            self.scaling_factors = self.awq_preprocessor.compute_scaling_factors(
                layer_name,
                W_2d,
                self.calibration_data
            )
            
            # Apply scaling to weights
            W_scaled = self.awq_preprocessor.apply_scaling(W_2d, self.scaling_factors)
            W = W_scaled.view(W.shape)
        
        # Compute quantization grid with AWQ-scaled weights
        if self.qparam_comput == "MinMax":
            scale, zero = self.quantizer.find_params_H(W, None, search=False)
        elif self.qparam_comput == "MMSE":
            scale, zero = self.quantizer.find_params_H(W, None, search=True)
        elif self.qparam_comput == "Hessian":
            scale, zero = self.quantizer.find_params_H(W, H_col, search=True)
        elif self.qparam_comput == "AWQ":
            # Use AWQ-aware parameter computation
            scale, zero = self.quantizer.find_params_H(W, None, search=True)
        else:
            raise NotImplementedError(f"Unknown qparam_comput method: {self.qparam_comput}")
        
        self.quantizer.scale = scale.reshape(self.quantizer.scale.shape)
        self.quantizer.zero = zero.reshape(self.quantizer.zero.shape)
        
        # Hessian-based re-ordering for columns
        if self.act_order_col:
            W, H_col, invperm_col = reorder_col(W, H_col)
        
        # Perform BOA optimization with AWQ-scaled weights
        if H_row is None:
            Q = self.aboa_gptq(W, H_col, scale, zero)
        else:
            # Hessian-based re-ordering for rows
            if self.act_order_row:
                W, H_row, scale, zero, invperm_row = reorder_row(W, H_row, scale, zero)
            
            Q = self.aboa_search(W, H_col, H_row, scale, zero)
            
            # Reverse re-ordering for rows
            if self.act_order_row:
                Q = reverse_reorder_row(Q, invperm_row)
        
        # Reverse re-ordering for columns
        if self.act_order_col:
            Q = reverse_reorder_col(Q, invperm_col)
        
        if print_memory_usage:
            print(f'\t |GPU memory: {torch.cuda.max_memory_allocated("cuda") / 1024**3:.3f}|')
        
        # Assign quantized (fake-quant) weights
        self.layer.weight.data = Q.reshape(self.org_shape).to(self.org_dtype)
        
        # Store scaling factors for potential runtime use
        if hasattr(self.layer, 'awq_scale'):
            self.layer.awq_scale = self.scaling_factors
    
    def aboa_gptq(self, W, H_col, scale, zero, return_err=False):
        """
        AWQ-informed GPTQ quantization with Hessian-based optimization.
        """
        U_col = get_cholesky_of_inverse(H_col)
        Q = torch.zeros_like(W)
        Err = torch.zeros_like(W)
        
        for idx_col in range(W.shape[-1]):
            # Quantization with AWQ-scaled weights
            w = W[..., idx_col].unsqueeze(-1)
            q = fake_quantize(w, scale, zero, self.quantizer.maxq)
            Q[..., idx_col] = q.squeeze(-1)
            
            # Error compensation with Hessian information
            err = (w - q) / U_col[..., idx_col, idx_col][:, None, None]
            Err[..., idx_col] = err.squeeze(-1)
            W[..., idx_col:] -= err @ U_col[..., idx_col, idx_col:].unsqueeze(-2)
        
        if return_err:
            return Q, Err
        else:
            return Q
    
    def aboa_search(self, W, H_col, H_row, scale, zero):
        """
        AWQ-informed BOA search with dual Hessian optimization.
        """
        U_col = get_cholesky_of_inverse(H_col)
        U_row = get_cholesky_of_inverse(H_row)
        Q = torch.zeros_like(W)
        
        for idx_row in range(W.shape[1]):
            # Quantization with AWQ-scaled weights
            W_sub = W[:, idx_row, :].unsqueeze(-2)
            Q_sub, Err = self.aboa_gptq(
                W_sub, 
                H_col, 
                scale[:, idx_row, :].unsqueeze(-2), 
                zero[:, idx_row, :].unsqueeze(-2), 
                return_err=True
            )
            Q[:, idx_row, :] = Q_sub.squeeze(-2)
            
            # Error compensation with dual Hessian
            W[:, idx_row:, :] -= (
                U_row.transpose(-1, -2)[:, idx_row:, idx_row].unsqueeze(-1) @ Err @ U_col
            ) / U_row[:, idx_row, idx_row][:, None, None]
        
        return Q
    
    def free(self):
        """
        Free memory and clear caches.
        """
        super().free()
        if self.awq_preprocessor is not None:
            self.awq_preprocessor.clear_cache()
        self.calibration_data = None
        self.scaling_factors = None