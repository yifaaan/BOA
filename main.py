import time
from contextlib import redirect_stdout
import io
from utils.model_utils import get_model
from utils.data_utils import get_calib_data
from utils.eval_utils import evaluate
from utils.process_args import get_boa_arguments, get_boa_weight_quant_infos
from quantize import boa_fwrd
import torch

if __name__ == '__main__':
    args = get_boa_arguments()
    
    # load model
    with redirect_stdout(io.StringIO()) as f:
        llm = get_model(args.llm_path)
    llm.seqlen = args.seqlen
    llm.eval()

    # evaluate the fp model performance
    if args.eval_fp:
        results = evaluate(llm, args)
        print(results)
        exit(0)

    # load calib. data
    calib_data = get_calib_data(args)

    # quantize
    qconfigs, boa_opts, hyperparams = get_boa_weight_quant_infos(args)
    
    # Print configuration info
    if args.use_awq:
        print("=" * 60)
        print("AWQ-Informed BOA (A-BOA) Quantization")
        print("=" * 60)
        print(f"Model: {args.llm_path}")
        print(f"Bits: {args.w_bits}")
        print(f"AWQ Salient Ratio: {args.awq_salient_ratio}")
        print(f"AWQ Alpha: {args.awq_alpha}")
        print("=" * 60)
    
    print("Start quantization")
    tick = time.time()
    
    # Use A-BOA if AWQ is enabled
    if args.use_awq:
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
        
        # Quantize each Transformer block with A-BOA
        for i in range(len(transformer_blocks)):
            print(f'>>>> Quantizing {i+1}-th Transformer Block.... ({i+1}/{len(transformer_blocks)})')
            transformer_block = transformer_blocks[i].to(dev)
            
            fp_layers = find_layers(transformer_block)
            
            wrappers = {}
            for name, fp_layer in fp_layers.items():
                # Use A-BOA wrapper
                wrappers[name] = ABoA(fp_layer, boa_opts, hyperparams)
                wrappers[name].quantizer = MinMaxQuantizer()
                wrappers[name].quantizer.configure(qconfigs["w_bits"], per_channel=True, sym=qconfigs["w_sym"], mse=False)
                wrappers[name].quantizer.find_params(wrappers[name].layer.weight.data)
                
                # Set calibration data for AWQ
                if boa_opts['use_awq']:
                    wrappers[name].set_calibration_data(quant_inps)
            
            # Compute Hessians
            block_v = boa_opts['block_v']
            compute_Hessian(transformer_block, n_heads, n_kv_heads, head_dim, wrappers, quant_inps, block_kwargs, block_v, rotary_matrix)
            
            # Quantize with A-BOA
            for name in fp_layers:
                print('-' * 50)
                print(f">>> Layer: {name}")
                if boa_opts['use_awq']:
                    print(f"    Using AWQ scaling")
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
    else:
        # Standard BOA quantization
        boa_fwrd(llm, calib_data, qconfigs, boa_opts, hyperparams, args)
    
    process_time = round(time.time() - tick, 3)
    print(f"Quantization processing time: {process_time}")
    
    # evaluate
    print(args)
    results = evaluate(llm, args)
    results['time'] = process_time
    print(results)