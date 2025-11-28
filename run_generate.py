#!/usr/bin/env python3
"""
Run vLLM generation with optional MoE expert logging.

This script runs the Qwen1.5-MoE model on GSM8K prompts and optionally
logs the expert routing decisions for analysis.

Usage:
    # Run without logging (baseline):
    python run_generate.py
    
    # Run with logging:
    VLLM_LOG_MOE=moe_routes.jsonl python run_generate.py
    
    # Run with logging (alternative):
    python run_generate.py --log
"""

import os
import sys
import json
import time
import random
import argparse
import subprocess
from pathlib import Path

# Configuration
SEED = 1234
MODEL_ID = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
MAX_MODEL_LEN = 512
MAX_TOKENS = 128
TEMPERATURE = 0.0
TOP_K_EXPERTS = 4  # Qwen MoE uses top-4 experts


def load_prompts() -> list:
    """Load prompts from file, generating if needed."""
    prompts_file = Path(__file__).parent / "prompts.txt"
    
    if not prompts_file.exists():
        print("Generating prompts from GSM8K dataset...")
        subprocess.run([sys.executable, "make_prompts.py"], check=True, cwd=prompts_file.parent)
    
    with open(prompts_file) as f:
        prompts = f.read().split("\n\n---\n\n")
    
    print(f"Loaded {len(prompts)} prompts from GSM8K")
    return prompts


def run_generation(with_logging: bool = False, log_path: str = "moe_routes.jsonl") -> dict:
    """
    Run generation with or without MoE logging.
    
    Args:
        with_logging: Whether to enable MoE logging
        log_path: Path for MoE log file when logging is enabled
    
    Returns:
        Dictionary with timing information
    """
    # Set random seed
    random.seed(SEED)
    
    # Set up logging if needed - MUST happen BEFORE importing vLLM
    if with_logging:
        os.environ['VLLM_LOG_MOE'] = log_path
        os.environ['VLLM_LOG_MOE_LAYER'] = '0'
        
        # Apply the patch BEFORE importing vLLM
        import patched_fused_moe
        patched_fused_moe.apply_patch()
    
    # Now import vLLM and torch
    import torch
    from vllm import LLM, SamplingParams
    import vllm
    
    # Write log header if logging
    if with_logging:
        import patched_fused_moe as pfm
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
        pfm.write_header(
            model_id=MODEL_ID,
            vllm_version=vllm.__version__,
            torch_version=torch.__version__,
            device=device,
            seed=SEED,
            top_k=TOP_K_EXPERTS
        )
    
    # Load prompts
    prompts = load_prompts()
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        seed=SEED
    )
    
    # Initialize the LLM
    print(f"\nInitializing model: {MODEL_ID}")
    print(f"Max model length: {MAX_MODEL_LEN}")
    
    import torch
    if torch.backends.mps.is_available():
        print("Device: Apple Silicon (MPS)")
    elif torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
    else:
        print("Device: CPU")
    
    llm = LLM(
        model=MODEL_ID,
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
        seed=SEED,
    )
    
    # Run generation
    print(f"\nGenerating responses for {len(prompts)} prompts...")
    print(f"Temperature: {TEMPERATURE}, Max tokens: {MAX_TOKENS}")
    if with_logging:
        print(f"Logging MoE routes to: {log_path}")
    
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    t1 = time.time()
    
    # Calculate statistics
    wall_time = t1 - t0
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    
    print(f"\n{'='*50}")
    print(f"Generation complete!")
    print(f"Wall time: {wall_time:.2f} seconds")
    print(f"Tokens generated: {total_tokens}")
    print(f"Throughput: {total_tokens/wall_time:.2f} tokens/sec")
    print(f"{'='*50}")
    
    # Close log if active
    if with_logging:
        import patched_fused_moe as pfm
        pfm.close_log()
    
    return {
        "wall_time_sec": round(wall_time, 3),
        "tokens_generated": total_tokens,
        "tokens_per_sec": round(total_tokens / wall_time, 2)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run vLLM generation with MoE logging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run baseline (no logging):
    python run_generate.py
    
    # Run with logging:
    python run_generate.py --log
    
    # Or use environment variable:
    VLLM_LOG_MOE=moe_routes.jsonl python run_generate.py
        """
    )
    parser.add_argument('--log', action='store_true',
                       help='Enable MoE expert logging')
    parser.add_argument('--log-path', default='moe_routes.jsonl',
                       help='Path for MoE routing log (default: moe_routes.jsonl)')
    parser.add_argument('--timing-file', default='timing.json',
                       help='Path for timing output (default: timing.json)')
    
    args = parser.parse_args()
    
    timing_data = {}
    timing_file = Path(args.timing_file)
    
    # Load existing timing data
    if timing_file.exists():
        try:
            with open(timing_file) as f:
                timing_data = json.load(f)
        except json.JSONDecodeError:
            timing_data = {}
    
    # Check if logging requested via environment or flag
    with_logging = args.log or os.environ.get('VLLM_LOG_MOE')
    log_path = os.environ.get('VLLM_LOG_MOE', args.log_path)
    
    if with_logging:
        print("\n" + "="*60)
        print("LOGGING MODE - Recording MoE expert selections")
        print("="*60 + "\n")
        
        result = run_generation(with_logging=True, log_path=log_path)
        timing_data["log"] = result
    else:
        print("\n" + "="*60)
        print("BASELINE MODE - No logging")
        print("="*60 + "\n")
        
        result = run_generation(with_logging=False)
        timing_data["no_log"] = result
    
    # Save timing data
    with open(timing_file, 'w') as f:
        json.dump(timing_data, f, indent=2)
    
    print(f"\nTiming data saved to: {timing_file}")
    
    # Calculate overhead if both runs available
    if "no_log" in timing_data and "log" in timing_data:
        no_log_time = timing_data["no_log"]["wall_time_sec"]
        log_time = timing_data["log"]["wall_time_sec"]
        overhead_pct = ((log_time - no_log_time) / no_log_time) * 100
        
        print("\n" + "="*60)
        print("OVERHEAD ANALYSIS")
        print("="*60)
        print(f"Baseline time:     {no_log_time:.2f} sec")
        print(f"With logging:      {log_time:.2f} sec")
        print(f"Logging overhead:  {overhead_pct:+.2f}%")
        print("="*60)


if __name__ == "__main__":
    main()
