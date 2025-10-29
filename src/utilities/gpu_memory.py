"""
GPU Memory Management Utilities

This module provides comprehensive GPU memory management functions for
handling multiple model loading and cleanup scenarios, particularly useful
when processing large language models sequentially.

Key Functions:
- cleanup_gpu_memory: Thorough cleanup of GPU memory after model processing
- print_gpu_memory_status: Monitor and display current GPU memory usage

Dependencies:
- torch: For CUDA operations and memory management
- transformers: For Pipeline type hints
"""

import gc
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import Pipeline


def cleanup_gpu_memory(pipe: "Pipeline") -> None:
    """
    Comprehensive GPU memory cleanup after model processing.
    
    This function ensures thorough cleanup of GPU memory when processing
    multiple models sequentially to prevent out-of-memory errors. It's
    particularly important when transitioning from smaller to larger models
    (e.g., 3B -> 70B parameters).
    
    The cleanup process:
    1. Moves model from GPU to CPU
    2. Explicitly deletes model components
    3. Forces garbage collection
    4. Clears CUDA cache and synchronizes
    
    Args:
        pipe: The transformers Pipeline to clean up
        
    Example:
        >>> pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-3B")
        >>> # ... use the pipeline ...
        >>> cleanup_gpu_memory(pipe)
        GPU memory cleanup completed
    """
    try:
        # Clear model from GPU memory
        if hasattr(pipe, 'model') and pipe.model is not None:
            # Move model to CPU first (if it was on GPU)
            if hasattr(pipe.model, 'cpu'):
                pipe.model.cpu()
            # Delete model reference
            del pipe.model
        
        # Clear tokenizer
        if hasattr(pipe, 'tokenizer'):
            del pipe.tokenizer
            
        # Delete the entire pipeline
        del pipe
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Synchronize to ensure all operations are complete
            torch.cuda.synchronize()
            
        print("GPU memory cleanup completed")
        
    except Exception as e:
        print(f"Warning: Error during GPU cleanup: {e}")
        # Still attempt basic cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def print_gpu_memory_status() -> None:
    """
    Print current GPU memory usage for monitoring.
    
    Displays allocated and cached memory for each available GPU device.
    Useful for tracking memory usage before/after model loading and cleanup.
    
    Output format:
        GPU 0: 2.45 GB allocated, 2.50 GB cached
        GPU 1: 0.00 GB allocated, 0.00 GB cached
        
    Example:
        >>> print_gpu_memory_status()
        GPU 0: 6.45 GB allocated, 8.50 GB cached
    """
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
            cached = torch.cuda.memory_reserved(i) / (1024 ** 3)  # GB
            print(f"GPU {i}: {allocated:.2f} GB allocated, {cached:.2f} GB cached")
    else:
        print("CUDA not available - no GPU memory to monitor")


def get_gpu_memory_info() -> dict:
    """
    Get GPU memory information as a dictionary.
    
    Returns:
        Dictionary with GPU memory information for each device
        
    Example:
        >>> info = get_gpu_memory_info()
        >>> print(info)
        {
            'cuda_available': True,
            'device_count': 1,
            'devices': {
                0: {'allocated_gb': 2.45, 'cached_gb': 2.50}
            }
        }
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': 0,
        'devices': {}
    }
    
    if torch.cuda.is_available():
        info['device_count'] = torch.cuda.device_count()
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
            cached = torch.cuda.memory_reserved(i) / (1024 ** 3)  # GB
            info['devices'][i] = {
                'allocated_gb': round(allocated, 2),
                'cached_gb': round(cached, 2)
            }
    
    return info
