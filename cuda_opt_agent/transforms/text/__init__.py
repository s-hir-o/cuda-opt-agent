"""Text-based kernel transforms — regex/string manipulation on CUDA source."""

from cuda_opt_agent.transforms.text.shared_memory_tiling import SharedMemoryTiling
from cuda_opt_agent.transforms.text.loop_unrolling import LoopUnrolling
from cuda_opt_agent.transforms.text.vectorized_loads import VectorizedLoads
from cuda_opt_agent.transforms.text.register_blocking import RegisterBlocking
from cuda_opt_agent.transforms.text.thread_coarsening import ThreadCoarsening
from cuda_opt_agent.transforms.text.warp_shuffle import WarpShuffleReduction
from cuda_opt_agent.transforms.text.kernel_fusion import KernelFusion

__all__ = [
    "SharedMemoryTiling",
    "LoopUnrolling",
    "VectorizedLoads",
    "RegisterBlocking",
    "ThreadCoarsening",
    "WarpShuffleReduction",
    "KernelFusion",
]