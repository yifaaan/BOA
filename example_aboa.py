#!/usr/bin/env python3
"""
Example usage script for AWQ-Informed BOA (A-BOA) quantization.

This script demonstrates how to use the A-BOA quantizer to achieve better
low-bit quantization by combining AWQ's activation-aware scaling with
BOA's Hessian-based optimization.
"""

import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.process_args import get_boa_arguments, get_boa_weight_quant_infos
from utils.data_utils import get_calib_data
from quantize import boa_fwrd_awq
from utils.eval_utils import eval_PPL


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="AWQ-Informed BOA (A-BOA) Quantization Example"
    )
    args = get_boa_arguments()
    
    # Example: Enable AWQ mode for 4-bit quantization
    print("=" * 70)
    print("AWQ-Informed BOA (A-BOA) Quantization Example")
    print("=" * 70)
    
    # Load model and tokenizer
    print(f"\nLoading model: {args.llm_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.llm_path,
        device_map="auto",
        torch_dtype=torch.float16,
        cache_dir=args.cache_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        cache_dir=args.cache_dir
    )
    
    # Get calibration data
    print(f"Preparing calibration data: {args.calib_data}")
    calib_data = get_calib_data(
        args.calib_data,
        tokenizer,
        args.nsamples,
        args.seqlen,
        args.seed,
        args.cache_dir
    )
    
    # Configure quantization settings
    qconfigs, boa_opts, hyperparams = get_boa_weight_quant_infos(args)
    
    # Print configuration
    print("\n" + "=" * 50)
    print("Quantization Configuration:")
    print("-" * 50)
    print(f"  Bits: {qconfigs['w_bits']}")
    print(f"  Symmetric: {qconfigs['w_sym']}")
    print(f"  AWQ Enabled: {boa_opts['use_awq']}")
    if boa_opts['use_awq']:
        print(f"  AWQ Salient Ratio: {boa_opts['awq_salient_ratio']}")
        print(f"  AWQ Alpha: {boa_opts['awq_alpha']}")
    print(f"  Qparam Method: {boa_opts['qparam_comput']}")
    print(f"  Column Reordering: {boa_opts['act_order_col']}")
    print(f"  Row Reordering: {boa_opts['act_order_row']}")
    print("=" * 50)
    
    # Evaluate original model if requested
    if args.eval_fp:
        print("\nEvaluating original FP16 model...")
        ppl_fp16 = eval_PPL(model, tokenizer, args.calib_data, args.cache_dir)
        print(f"Original model perplexity: {ppl_fp16:.3f}")
    
    # Apply A-BOA quantization
    print("\n" + "=" * 50)
    print("Applying A-BOA Quantization...")
    print("=" * 50)
    
    # Standard BOA quantization (without AWQ)
    if not args.use_awq:
        print("\nRunning standard BOA quantization...")
        boa_fwrd(model, calib_data, qconfigs, boa_opts, hyperparams, args)
    
    # AWQ-informed BOA quantization
    else:
        print("\nRunning AWQ-informed BOA (A-BOA) quantization...")
        print("This combines AWQ scaling with BOA optimization for improved accuracy.")
        boa_fwrd_awq(model, calib_data, qconfigs, boa_opts, hyperparams, args)
    
    # Evaluate quantized model
    print("\n" + "=" * 50)
    print("Evaluating quantized model...")
    ppl_quant = eval_PPL(model, tokenizer, args.calib_data, args.cache_dir)
    print(f"Quantized model perplexity: {ppl_quant:.3f}")
    
    if args.eval_fp:
        print(f"Perplexity increase: {ppl_quant - ppl_fp16:.3f}")
    
    # Optional: Run LM evaluation harness
    if args.lm_eval:
        print("\n" + "=" * 50)
        print("Running LM Evaluation Harness...")
        from utils.eval_utils import eval_zero_shot
        results = eval_zero_shot(
            model,
            tokenizer,
            args.tasks,
            args.lm_eval_batch_size
        )
        
        print("\nZero-shot evaluation results:")
        for task, score in results.items():
            print(f"  {task}: {score:.3f}")
    
    print("\n" + "=" * 70)
    print("A-BOA Quantization Complete!")
    print("=" * 70)


def boa_fwrd_awq(llm, calib_data, qconfigs, boa_opts, hyperparams, args):
    """
    AWQ-informed BOA forward quantization.
    """
    import functools
    from quantizers.aboa import ABoA
    from quantizers.minmax import MinMaxQuantizer
    from utils.model_utils import get_transformer_blocks, get_head_info, get_rotary_emb, cache_first_transformer_input
    from utils.utils import find_layers, cleanup_memory
    from quantize import get_rotary_matrix, compute_Hessian, QKV_NAMES
    
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    use_cache = llm.config.use_cache
    llm.config.use_cache = False
    
    # Cache inputs for fast quantization
    quant_inps, block_kwargs = cache_first_transformer_input(llm, calib_data)
    
    transformer_blocks = get_transformer_blocks(llm)
    n_heads, n_kv_heads, head_dim = get_head_info(llm)
    rotary_emb = get_rotary_emb(llm)
    rotary_matrix = get_rotary_matrix(rotary_emb, llm.config, block_kwargs['position_ids'].cpu()) if rotary_emb is not None else None
    
    # Quantize each Transformer block
    for i in range(len(transformer_blocks)):
        print(f'\n>>>> Quantizing {i+1}-th Transformer Block.... ({i+1}/{len(transformer_blocks)})')
        transformer_block = transformer_blocks[i].to(dev)
        
        fp_layers = find_layers(transformer_block)
        
        wrappers = {}
        for name, fp_layer in fp_layers.items():
            # Use A-BOA wrapper instead of standard BoA
            wrappers[name] = ABoA(fp_layer, boa_opts, hyperparams)
            wrappers[name].quantizer = MinMaxQuantizer()
            wrappers[name].quantizer.configure(
                qconfigs["w_bits"], 
                per_channel=True, 
                sym=qconfigs["w_sym"], 
                mse=False
            )
            wrappers[name].quantizer.find_params(wrappers[name].layer.weight.data)
            
            # Set calibration data for AWQ scaling
            if boa_opts['use_awq']:
                # Extract relevant activations for this layer
                # This is a simplified version - in practice, you'd need to
                # collect proper activations during forward pass
                wrappers[name].set_calibration_data(quant_inps)
        
        # Compute Hessians
        block_v = boa_opts['block_v']
        compute_Hessian(
            transformer_block, 
            n_heads, 
            n_kv_heads, 
            head_dim, 
            wrappers, 
            quant_inps, 
            block_kwargs, 
            block_v, 
            rotary_matrix
        )
        
        # Quantize with AWQ-informed BOA
        for name in fp_layers:
            print('-' * 50)
            print(f">>> Layer: {name}")
            if boa_opts['use_awq']:
                print(f"    Using AWQ scaling (salient_ratio={boa_opts['awq_salient_ratio']}, alpha={boa_opts['awq_alpha']})")
            wrappers[name].quant(args.print_memory_usage)
            wrappers[name].free()
        
        # Cache inputs for next transformer block
        for j in range(len(quant_inps)):
            quant_inps[j] = transformer_block(quant_inps[j].unsqueeze(0), **block_kwargs)[0]
        
        transformer_blocks[i] = transformer_block.cpu()
        del transformer_block
        del wrappers
        
        cleanup_memory(verbose=False)
    
    llm.config.use_cache = use_cache


if __name__ == "__main__":
    main()