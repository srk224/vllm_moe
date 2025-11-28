# vLLM MoE Expert Logging Project

This project modifies vLLM to log which Mixture of Experts (MoE) are selected per token during inference, and generates visualizations of expert usage patterns.

## Overview

**Model**: Qwen/Qwen1.5-MoE-A2.7B-Chat (14.3B total parameters, 2.7B activated)  
**Dataset**: GSM8K test split (first 25 questions)  
**Generation Settings**: max_tokens=128, temperature=0.0, seed=1234


## Setup Instructions

### Prerequisites
- Python 3.8+
- CUDA for acceleration

### Installation

```bash
# Clone/enter project directory
cd llm-moe-project

pip install vllm datasets matplotlib numpy transformers
# Note: Use VLLM_USE_PRECOMPILED=1 pip install vllm for faster installation
```

## Usage

### 1. Generate Prompts
```bash
python make_prompts.py
# Creates prompts.txt with 25 GSM8K questions
```

### 2. Run Baseline (No Logging)
```bash
python run_generate.py
# Creates timing.json with baseline metrics
```

### 3. Run with MoE Logging
```bash
VLLM_LOG_MOE=moe_routes.jsonl python run_generate.py --log-only
```

### 4. Generate Visualization

## Where the Hook Is Applied

The MoE logging is implemented by patching vLLM's `fused_topk` function in `vllm/model_executor/layers/fused_moe/fused_moe.py`. This function is called by the MoE layer to compute the top-k expert selections (topk_ids) and their routing weights (topk_weights) for each token.

**Patch location**: `patched_fused_moe.py` wraps the original `fused_topk` function to:
1. Call the original function to get routing decisions
2. Log the `topk_ids` and `topk_weights` tensors
3. Return results unchanged 

## AI Usage Log

### Tools Used
- **Claude (Anthropic)**: Understanding a lot of stuff, topics , Project design, code generation, documentation
- **Web Search**: vLLM architecture research, MoE implementation details
