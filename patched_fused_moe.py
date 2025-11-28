import json
import os
import threading
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
import torch


@dataclass
class MoELogState:
    """Global state for MoE logging."""
    enabled: bool = False
    log_path: Optional[Path] = None
    file_handle: Optional[object] = None
    write_lock: threading.Lock = field(default_factory=threading.Lock)
    header_written: bool = False
    layers_to_log: List[int] = field(default_factory=lambda: [0])
    token_counter: int = 0
    layer_counter: int = 0  # Tracks which MoE layer in the forward pass


# Global logging state
_log_state: Optional[MoELogState] = None

# Store original functions
_original_fused_topk = None
_patched = False


def get_log_state() -> MoELogState:
    """Get or create the global log state."""
    global _log_state
    if _log_state is None:
        _log_state = MoELogState()
    return _log_state


def init_logging():
    """Initialize MoE logging from environment variables."""
    state = get_log_state()
    
    log_path = os.environ.get('VLLM_LOG_MOE')
    if log_path:
        state.enabled = True
        state.log_path = Path(log_path)
        
        # Parse layers to log
        layers_str = os.environ.get('VLLM_LOG_MOE_LAYER', '0')
        try:
            state.layers_to_log = [int(x.strip()) for x in layers_str.split(',')]
        except ValueError:
            state.layers_to_log = [0]
        
        print(f"[MoE Patch] Logging enabled")
        print(f"[MoE Patch] Log path: {state.log_path}")
        print(f"[MoE Patch] Layers to log: {state.layers_to_log}")
        return True
    return False


def write_header(model_id: str, vllm_version: str, torch_version: str,
                 device: str, seed: int, top_k: int):
    """Write metadata header to log file."""
    state = get_log_state()
    
    if not state.enabled:
        return
    
    with state.write_lock:
        if state.header_written:
            return
        
        state.file_handle = open(state.log_path, 'w')
        
        header = {
            "type": "meta",
            "model_id": model_id,
            "vllm_version": vllm_version,
            "torch_version": torch_version,
            "device": device,
            "seed": seed,
            "layers_logged": state.layers_to_log,
            "top_k": top_k
        }
        
        state.file_handle.write(json.dumps(header) + '\n')
        state.file_handle.flush()
        state.header_written = True
        print(f"[MoE Patch] Header written to {state.log_path}")


def log_routing(layer_idx: int, topk_ids: torch.Tensor, 
                topk_weights: torch.Tensor):
    """Log routing decisions for a batch of tokens."""
    state = get_log_state()
    
    if not state.enabled or state.file_handle is None:
        return
    
    if layer_idx not in state.layers_to_log:
        return
    
    # Convert to lists
    try:
        topk_ids_list = topk_ids.detach().cpu().tolist()
        topk_weights_list = topk_weights.detach().cpu().tolist()
    except Exception as e:
        print(f"[MoE Patch] Error converting tensors: {e}")
        return
    
    with state.write_lock:
        for i, (ids, weights) in enumerate(zip(topk_ids_list, topk_weights_list)):
            record = {
                "type": "route",
                "req_id": "batch",
                "token_idx": state.token_counter + i,
                "layer": layer_idx,
                "topk_ids": ids if isinstance(ids, list) else [ids],
                "topk_weights": [round(w, 4) for w in (weights if isinstance(weights, list) else [weights])]
            }
            state.file_handle.write(json.dumps(record) + '\n')
        
        state.token_counter += len(topk_ids_list)
        state.file_handle.flush()


def reset_counters():
    """Reset counters for new generation batch."""
    state = get_log_state()
    with state.write_lock:
        state.token_counter = 0
        state.layer_counter = 0


def increment_layer():
    """Increment the layer counter."""
    state = get_log_state()
    state.layer_counter += 1


def get_current_layer() -> int:
    """Get current layer index."""
    return get_log_state().layer_counter


def close_log():
    """Close the log file."""
    state = get_log_state()
    with state.write_lock:
        if state.file_handle:
            state.file_handle.close()
            state.file_handle = None
            print(f"[MoE Patch] Log file closed: {state.log_path}")


def create_patched_fused_topk(original_func):
    """Create a patched version of fused_topk that logs routing."""
    
    def patched_fused_topk(
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        indices_type: torch.dtype = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Patched fused_topk that logs expert selections."""
        
        # Call original function
        topk_weights, topk_ids, token_expert_indices = original_func(
            hidden_states, gating_output, topk, renormalize, indices_type
        )
        
        # Log if enabled
        state = get_log_state()
        if state.enabled and state.file_handle:
            current_layer = get_current_layer()
            if current_layer in state.layers_to_log:
                log_routing(current_layer, topk_ids, topk_weights)
            increment_layer()
        
        return topk_weights, topk_ids, token_expert_indices
    
    return patched_fused_topk


def apply_patch() -> bool:
    """Apply the MoE logging patch to vLLM."""
    global _original_fused_topk, _patched
    
    if _patched:
        print("[MoE Patch] Already patched")
        return True
    
    # Initialize logging
    if not init_logging():
        print("[MoE Patch] Logging not enabled (VLLM_LOG_MOE not set)")
        return False
    
    try:
        # Import vLLM's fused_moe module
        import vllm.model_executor.layers.fused_moe.fused_moe as moe_module
        
        # Store original
        _original_fused_topk = moe_module.fused_topk
        
        # Apply patch
        moe_module.fused_topk = create_patched_fused_topk(_original_fused_topk)
        
        # Also need to update the reference in the parent module
        import vllm.model_executor.layers.fused_moe as parent_moe
        parent_moe.fused_topk = moe_module.fused_topk
        
        _patched = True
        print("[MoE Patch] Successfully patched fused_topk")
        return True
        
    except ImportError as e:
        print(f"[MoE Patch] Could not import vLLM fused_moe: {e}")
        return False
    except Exception as e:
        print(f"[MoE Patch] Error applying patch: {e}")
        import traceback
        traceback.print_exc()
        return False


def remove_patch():
    """Remove the MoE logging patch."""
    global _original_fused_topk, _patched
    
    if not _patched:
        return
    
    try:
        import vllm.model_executor.layers.fused_moe.fused_moe as moe_module
        import vllm.model_executor.layers.fused_moe as parent_moe
        
        if _original_fused_topk:
            moe_module.fused_topk = _original_fused_topk
            parent_moe.fused_topk = _original_fused_topk
        
        _patched = False
        close_log()
        print("[MoE Patch] Patch removed")
    except Exception as e:
        print(f"[MoE Patch] Error removing patch: {e}")


def setup_moe_logging(log_path: str, layers: List[int] = None):
    """
    Set up MoE logging with the given parameters.
    
    Call this BEFORE importing vLLM or after setting env vars but before
    creating an LLM instance.
    """
    os.environ['VLLM_LOG_MOE'] = log_path
    if layers:
        os.environ['VLLM_LOG_MOE_LAYER'] = ','.join(map(str, layers))


if __name__ == "__main__":
    print("This module patches vLLM's FusedMoE for logging.")
    print("Usage: import patched_fused_moe; patched_fused_moe.apply_patch()")
