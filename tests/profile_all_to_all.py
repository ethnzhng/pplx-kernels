# ruff: noqa: T201

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import torch

from pplx_kernels import AllToAll
from pplx_kernels.nvshmem import (
    nvshmem_alloc_empty_unique_id,
    nvshmem_alltoall,
    nvshmem_barrier_all_on_current_stream,
    nvshmem_finalize,
    nvshmem_get_unique_id,
    nvshmem_init,
    nvshmem_malloc,
)

from .all_to_all_utils import MoEConfig, RankTestData
from .distributed_utils import (
    ProcessGroupInfo,
    parallel_launch,
    parallel_launch_from_env,
)

logger = logging.getLogger(__name__)


@torch.inference_mode()
def bench_all_to_all(
    pgi: ProcessGroupInfo,
    dp_size: int,
    moe: MoEConfig,
    selected_kernel: str,
) -> tuple[tuple[int, ...], torch.Tensor]:
    device = pgi.device
    num_dp = pgi.world_size // dp_size
    dp_rank = pgi.rank // dp_size

    # Generate the same rank data for each DP group
    rng = torch.Generator()
    rng.manual_seed(dp_rank + 1)
    rank_data = RankTestData(moe, rng, use_max_tokens=True)

    # Allocate symmetric memory
    num_local_experts = moe.num_experts // pgi.world_size

    hidden_dim_scale_bytes = (
        0
        if moe.in_dtype.itemsize != 1
        else (
            (moe.hidden_dim + moe.block_size - 1)
            // moe.block_size
            * torch.float32.itemsize
        )
    )

    ata = AllToAll(
        max_num_tokens=moe.max_num_tokens,
        num_experts=moe.num_experts,
        experts_per_token=moe.experts_per_token,
        rank=pgi.rank,
        world_size=pgi.world_size,
        dp_size=dp_size,
        hidden_dim=moe.hidden_dim,
        hidden_dim_bytes=moe.hidden_dim * moe.in_dtype.itemsize,
        hidden_dim_scale_bytes=hidden_dim_scale_bytes,
    )

    # Allocate input and output tensors
    expert_num_tokens = torch.empty(
        num_local_experts,
        dtype=torch.int32,
        device=device,
    )
    expert_x = torch.empty(
        (num_local_experts, moe.max_num_tokens * num_dp, moe.hidden_dim),
        dtype=moe.in_dtype,
        device=device,
    )
    expert_x_scale: torch.Tensor | None = None
    if moe.in_dtype.itemsize == 1:
        expert_x_scale = torch.empty(
            (
                expert_x.size(0),
                expert_x.size(1),
                (expert_x.size(2) + moe.block_size - 1) // moe.block_size,
            ),
            dtype=torch.float32,
            device=device,
        )
    expert_y = expert_x.to(moe.out_dtype)
    dp_x = rank_data.x.to(device)
    dp_x_scale = rank_data.x_scale.to(device) if rank_data.x_scale is not None else None
    indices = rank_data.indices.to(device).to(torch.uint32)
    bound_m = torch.tensor([rank_data.num_tokens], dtype=torch.uint32, device=device)
    weights = rank_data.weights.to(device)
    y = torch.full(
        (moe.max_num_tokens, moe.hidden_dim),
        torch.nan,
        dtype=moe.out_dtype,
        device=device,
    )

    hidden_dim_bytes_with_scale = moe.hidden_dim // dp_size * moe.in_dtype.itemsize
    if moe.in_dtype.itemsize == 1:
        hidden_dim_bytes_with_scale += (
            (moe.hidden_dim // dp_size + moe.block_size - 1)
            // moe.block_size
            * torch.float32.itemsize
        )

    a2a_shape = (
        pgi.world_size,
        num_local_experts,
        moe.max_num_tokens * hidden_dim_bytes_with_scale,
    )
    a2a_tensor = torch.empty(
        a2a_shape,
        dtype=torch.uint8,
        device=device,
    )
    a2a_out_tensor = torch.empty_like(a2a_tensor)

    nvshmem_in = nvshmem_malloc(a2a_shape, torch.uint8, device)
    nvshmem_out = nvshmem_malloc(a2a_shape, torch.uint8, device)

    # Compute stats
    dispatch_bytes = (
        rank_data.num_tokens * moe.experts_per_token * hidden_dim_bytes_with_scale
    )
    combine_bytes = (
        rank_data.num_tokens
        * moe.experts_per_token
        * (moe.hidden_dim // dp_size)
        * moe.out_dtype.itemsize
    )
    a2a_bytes = a2a_tensor.numel() * a2a_tensor.element_size()
    nvshmem_bytes = nvshmem_in.numel() * nvshmem_in.element_size()

    is_local_rank_0 = pgi.local_rank == 0

    # Benchmark launcher
    def run(is_warmup: bool = False) -> tuple[float, ...]:
        events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        stream = torch.cuda.current_stream()

        nvshmem_barrier_all_on_current_stream()
        events[0].record(stream)

        # Only use NVTX range for non-warmup runs
        if is_local_rank_0 and not is_warmup:
            torch.cuda.nvtx.range_push(selected_kernel)

        if selected_kernel == "pplx_dispatch":
            ata.dispatch(
                out_expert_num_tokens=expert_num_tokens,
                out_expert_x=expert_x,
                out_expert_x_scale=expert_x_scale,
                dp_x=dp_x,
                dp_x_scale=dp_x_scale,
                indices=indices,
                bound_m=bound_m,
            )
        elif selected_kernel == "pplx_combine":
            ata.combine(
                out_tokens=y,
                indices=indices,
                weights=weights,
                expert_y=expert_y,
                bound_m=bound_m,
            )
        elif selected_kernel == "torch_all_to_all":
            torch.distributed.all_to_all_single(a2a_out_tensor, a2a_tensor)
        elif selected_kernel == "nvshmem_all_to_all":
            nvshmem_alltoall(nvshmem_out, nvshmem_in)

        if is_local_rank_0 and not is_warmup:
            torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize()
            # torch.distributed.barrier()

        events[1].record(stream)

        # Get latency
        stream.synchronize()
        elapsed_us = events[0].elapsed_time(events[1]) * 1e3

        if selected_kernel == "pplx_dispatch":
            bytes_transferred = dispatch_bytes
        elif selected_kernel == "pplx_combine":
            bytes_transferred = combine_bytes
        elif selected_kernel == "torch_all_to_all":
            bytes_transferred = a2a_bytes
        elif selected_kernel == "nvshmem_all_to_all":
            bytes_transferred = nvshmem_bytes

        gbps = bytes_transferred / elapsed_us / 1e3

        return (elapsed_us, gbps)

    # Warmup
    num_warmup = 20
    with torch.cuda.nvtx.range(
        f"[Warmup] Exp={moe.num_experts}, Tok={moe.max_num_tokens}, HidDim={moe.hidden_dim}"
    ):
        for _ in range(num_warmup):
            run(is_warmup=True)

    # Benchmark
    torch.distributed.barrier()
    num_repeat = 1
    with torch.cuda.nvtx.range(
        f"[Benchmark] Exp={moe.num_experts}, Tok={moe.max_num_tokens}, HidDim={moe.hidden_dim}"
    ):
        result = torch.tensor([run(is_warmup=False) for _ in range(num_repeat)])

    # Cleanup
    ata.destroy()

    return (
        (dispatch_bytes, combine_bytes, a2a_bytes, nvshmem_bytes),
        result,
    )


def _worker_profile_all_to_all(
    pgi: ProcessGroupInfo,
    dp_size: int,
    in_dtype_str: str,
    out_dtype_str: str,
    selected_kernel: str,
) -> None:
    uid = nvshmem_get_unique_id() if pgi.rank == 0 else nvshmem_alloc_empty_unique_id()
    torch.distributed.broadcast(uid, src=0)
    nvshmem_init(uid, pgi.rank, pgi.world_size)

    in_dtype = getattr(torch, in_dtype_str)
    out_dtype = getattr(torch, out_dtype_str)
    assert isinstance(in_dtype, torch.dtype)
    configs = [
        # V2-Lite:  64 Experts, 6 Experts per Token, 2048 Hidden Dim
        # MoEConfig(64, 6, 2048, 1, in_dtype, out_dtype),
        # MoEConfig(64, 6, 2048, 4, in_dtype, out_dtype),
        # MoEConfig(64, 6, 2048, 8, in_dtype, out_dtype),
        # MoEConfig(64, 6, 2048, 16, in_dtype, out_dtype),
        # MoEConfig(64, 6, 2048, 32, in_dtype, out_dtype),
        # MoEConfig(64, 6, 2048, 64, in_dtype, out_dtype),
        MoEConfig(64, 6, 2048, 128, in_dtype, out_dtype),
        # R1     : 256 Experts, 8 Experts per Token, 7168 Hidden Dim
        # MoEConfig(256, 8, 7168, 1, in_dtype, out_dtype),
        # MoEConfig(256, 8, 7168, 4, in_dtype, out_dtype),
        # MoEConfig(256, 8, 7168, 8, in_dtype, out_dtype),
        # MoEConfig(256, 8, 7168, 16, in_dtype, out_dtype),
        # MoEConfig(256, 8, 7168, 32, in_dtype, out_dtype),
        # MoEConfig(256, 8, 7168, 64, in_dtype, out_dtype),
        # MoEConfig(256, 8, 7168, 128, in_dtype, out_dtype),
    ]

    header = [
        "E",
        "E/tok",
        "tok",
        "dim",
        f"{selected_kernel.capitalize()}_lat",
        f"{selected_kernel.capitalize()}_bw",
        f"{selected_kernel.capitalize()}_bytes",
    ]

    outpath = (
        Path(__file__).resolve().parents[1]
        / "data"
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_all_to_all.tsv"
    )
    f_out = None
    if pgi.rank == 0:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        f_out = outpath.open("w")

        line = f"EP={pgi.world_size} DP={pgi.world_size // dp_size}"
        print(line)
        f_out.write(line + "\n")

        line = "\t".join(header)
        print(line)
        f_out.write(line + "\n")

    for config in configs:
        if pgi.world_size > config.num_experts:
            continue
        meta, result = bench_all_to_all(pgi, dp_size, config, selected_kernel)
        bytes_transferred = meta[
            [
                "pplx_dispatch",
                "pplx_combine",
                "torch_all_to_all",
                "nvshmem_all_to_all",
            ].index(selected_kernel)
        ]
        if pgi.rank == 0:
            row: dict[str, str] = {
                "E": f"{config.num_experts}",
                "E/tok": f"{config.experts_per_token}",
                "tok": f"{config.max_num_tokens}",
                "dim": f"{config.hidden_dim}",
                f"{selected_kernel.capitalize()}_lat": f"{result[:, 0].mean():4.1f}μs"
                + (f" ± {result[:, 0].std():4.1f}μs" if len(result) > 1 else ""),
                f"{selected_kernel.capitalize()}_bw": f"{result[:, 1].mean():2.3f}GB/s",
                f"{selected_kernel.capitalize()}_bytes": f"{bytes_transferred}",
            }
            assert list(row.keys()) == header
            line = "\t".join(row[h] for h in header)
            print(line)
            assert f_out is not None
            f_out.write(line + "\n")
            f_out.flush()

    if f_out is not None:
        f_out.close()
        print("Saved to", outpath)

    nvshmem_finalize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument(
        "--in-dtype",
        choices=["bfloat16", "float16", "float8_e4m3fn"],
        default="float8_e4m3fn",
    )
    parser.add_argument(
        "--out-dtype",
        choices=["bfloat16", "float16"],
        default="bfloat16",
    )
    parser.add_argument(
        "--kernel",
        choices=[
            "pplx_dispatch",
            "pplx_combine",
            "torch_all_to_all",
            "nvshmem_all_to_all",
        ],
        required=True,
    )
    args = parser.parse_args()
    dp_size = int(args.dp_size)
    in_dtype = str(args.in_dtype)
    out_dtype = str(args.out_dtype)
    selected_kernel = str(args.kernel)

    if "MASTER_ADDR" in os.environ:
        parallel_launch_from_env(
            _worker_profile_all_to_all, dp_size, in_dtype, out_dtype, selected_kernel
        )
    else:
        world_size = torch.cuda.device_count()
        parallel_launch(
            world_size,
            _worker_profile_all_to_all,
            dp_size,
            in_dtype,
            out_dtype,
            selected_kernel,
        )


if __name__ == "__main__":
    main()
