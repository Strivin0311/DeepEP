import argparse
import os
import time
from typing import Callable

import torch
import torch.distributed as dist

# noinspection PyUnresolvedReferences
import deep_ep
from utils import bench, bench_kineto, calc_diff, create_grouped_scores, inplace_unique, per_token_cast_to_fp8, per_token_cast_back

# Test compatibility with low latency functions
import test_low_latency


def setup_dist_env(
    backend: str = "nccl",
    base_seed: int | None = None,
    seed_bias: Callable = lambda rank: 0,
) -> tuple[int, int, int, dist.ProcessGroup, int, int | None]:
    """set up distributed environment with the specified process group backend,
    NOTE: the test script using this func to set up should be executed through torchrun

    Args:
        backend (str, optional): the process group backend. Defaults to "nccl".
        base_seed (int | None, optional): the base seed. Defaults to None to not set seed.
        seed_bias (Callable, optional): the seed bias func for each rank. Defaults to lambda rank: 0, i.e., no bias.

    Returns:
        rank, local_rank, world_size, world_group, device, seed
    """
    num_nodes = int(os.getenv('NNODES'))
    num_local_ranks = int(os.getenv('NPROC_PER_NODE'))
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )

    seed = None
    if base_seed is not None:
        seed = base_seed + seed_bias(rank)
        torch.manual_seed(seed)

    return (
        num_nodes,
        num_local_ranks,
        world_size, # num_ranks
        rank,
        local_rank,
        dist.group.WORLD,
        device,
        seed,
    )  # noqa: E231


# noinspection PyShadowingNames
def test_main(args: argparse.Namespace, num_sms: int,
              local_rank: int, num_local_ranks: int, num_ranks: int, num_nodes: int, rank: int,
              buffer: deep_ep.Buffer, group: dist.ProcessGroup):
    # Settings
    num_tokens, hidden = args.num_tokens, args.hidden
    num_topk_groups, num_topk, num_experts = args.num_topk_groups, args.num_topk, args.num_experts

    assert num_experts % num_ranks == 0 and num_local_ranks == 8
    
    num_max_nvl_chunked_send_tokens = 8
    nvl_buffer_size = num_max_nvl_chunked_recv_tokens = (720 if num_ranks in (144, 160) else 512)
    
    num_max_rdma_chunked_send_tokens = 16
    rdma_buffer_size = num_max_rdma_chunked_recv_tokens = 128
    
    if local_rank == 0:
        print(
            (
                f"[config] {num_max_nvl_chunked_send_tokens=} | {num_max_nvl_chunked_recv_tokens=} | {nvl_buffer_size=}\n"
                f"{num_max_rdma_chunked_send_tokens=} | {num_max_rdma_chunked_recv_tokens=} | {rdma_buffer_size=}\n"
            ), 
            flush=True
        )
    
    # Config
    config = deep_ep.Config(
        num_sms,  # num_sms, default 20
        num_max_nvl_chunked_send_tokens, # num_max_nvl_chunked_send_tokens (nvl_chunk_size), default 6
        num_max_nvl_chunked_recv_tokens, # num_max_nvl_chunked_recv_tokens (nvl_buffer_size), default 256
        num_max_rdma_chunked_send_tokens, # num_max_rdma_chunked_send_tokens, default 6
        num_max_rdma_chunked_recv_tokens, # num_max_rdma_chunked_recv_tokens, default 256
    )

    # Random data
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * rank
    x_pure_rand = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    x_e4m3 = per_token_cast_to_fp8(x)
    x_e4m3 = (x_e4m3[0], x_e4m3[1].T.contiguous().T)
    
    # Random score
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    group_scores = scores.view(num_tokens, num_nodes, -1).amax(dim=-1)
    group_idx = torch.topk(group_scores, k=num_topk_groups, dim=-1, sorted=False).indices
    masked_scores = create_grouped_scores(scores, group_idx, num_nodes)
    assert torch.equal(scores, masked_scores) # since we guarantee num_nodes == num_topk_groups, thus scores == masked_scores
    
    topk_idx = torch.topk(masked_scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device='cuda') * rank
    topk_weights_pure_rand = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='cuda')
    rank_idx = topk_idx // (num_experts // num_ranks)
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)
    print(f"[RANK {rank}]: {rank_idx=} | {rank_idx.shape=}\n", flush=True)
    
    rdma_rank_idx = rank_idx // num_local_ranks
    rdma_rank_idx.masked_fill_(rank_idx == -1, -1)
    inplace_unique(rdma_rank_idx, num_nodes)

    # RDMA dispatch counts
    rdma_idx = topk_idx // (num_experts // num_nodes)
    rdma_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rdma_idx, num_nodes)
    num_rdma_token_sent = rdma_idx.ne(-1).sum().item()
    assert torch.equal(rdma_idx, rdma_rank_idx)
    print(f"[RANK {rank}]: {rdma_idx=} | {rdma_idx.shape=} | {num_rdma_token_sent=}\n", flush=True)

    # Expert meta
    num_tokens_per_expert = torch.zeros((num_experts, ), dtype=torch.int, device='cuda')
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()
    gbl_num_tokens_per_expert = num_tokens_per_expert.clone()
    dist.all_reduce(gbl_num_tokens_per_expert, group=group)
    if local_rank == 0:
        print(f"{gbl_num_tokens_per_expert=} | {gbl_num_tokens_per_expert.shape=}\n", flush=True)
    print(f"[RANK {rank}]: {num_tokens_per_expert=} | {num_tokens_per_expert.shape=}\n", flush=True)

    # Rank layout meta
    num_tokens_per_rank = torch.empty((num_ranks, ), dtype=torch.int, device='cuda')
    num_tokens_per_rdma_rank = torch.empty((num_nodes, ), dtype=torch.int, device='cuda')
    token_idx_in_rank = torch.full((num_ranks, num_tokens), -1, dtype=torch.long, device='cuda')
    for i in range(num_ranks):
        num_tokens_per_rank[i] = (rank_idx == i).sum()
        token_sel = (rank_idx == i).max(dim=-1)[0]
        count = token_sel.sum().item()
        tokens = torch.sort(token_sel.to(torch.int), descending=True)[1]
        tokens[:count] = torch.sort(tokens[:count])[0]
        token_idx_in_rank[i][tokens[:count]] = torch.arange(count, dtype=torch.long, device='cuda')
    for i in range(num_nodes):
        num_tokens_per_rdma_rank[i] = (rdma_rank_idx == i).sum()
    token_idx_in_rank = token_idx_in_rank.T.contiguous().to(torch.int)
    is_token_in_rank = token_idx_in_rank >= 0
    gbl_num_tokens_per_rank = num_tokens_per_rank.clone()
    dist.all_reduce(gbl_num_tokens_per_rank, group=group)
    if local_rank == 0:
        print(f"{gbl_num_tokens_per_rank=} | {gbl_num_tokens_per_rank.shape=}\n", flush=True)
    print(f"[RANK {rank}]: {num_tokens_per_rank=} | {num_tokens_per_rank.shape=}\n", flush=True)
    print(f"[RANK {rank}]: {num_tokens_per_rdma_rank=} | {num_tokens_per_rdma_rank.shape=}\n", flush=True)

    # get dispatch layout from buffer
    ref_num_tokens_per_rank, ref_num_tokens_per_rdma_rank, ref_num_tokens_per_expert, ref_is_token_in_rank, _ = \
        buffer.get_dispatch_layout(topk_idx, num_experts)
        
    # assert close to layout ref
    assert torch.allclose(ref_num_tokens_per_rank, num_tokens_per_rank)
    assert torch.allclose(ref_num_tokens_per_rdma_rank, num_tokens_per_rdma_rank)
    assert torch.allclose(ref_num_tokens_per_expert, num_tokens_per_expert)
    assert torch.allclose(ref_is_token_in_rank, is_token_in_rank)
    
    # benchmark dispatch layout
    t = bench(lambda: buffer.get_dispatch_layout(topk_idx, num_experts))[0]
    if local_rank == 0:
        print(f'[layout] Kernel performance: {t * 1000:.3f} ms', flush=True)
        print('', flush=True)
    group.barrier()
    time.sleep(1)

    # Test dispatch
    # noinspection PyShadowingNames
    def check_data(check_x, recv_gbl_rank_prefix_sum):
        assert torch.allclose(check_x.amin(dim=1), check_x.amax(dim=1))
        check_start = 0
        for i in range(num_ranks):
            check_end = recv_gbl_rank_prefix_sum[i].item()
            assert (check_x[check_start:check_end, :].int() - i).sum().item() == 0
            check_start = check_end

    for previous_mode in (True,): # (False, True):
        for async_mode in (True,): # (False, True):
            for current_x in (x,): # (x_pure_rand, x, x_e4m3):
                for with_topk in (True,): # (False, True):
                    if local_rank == 0:
                        print("\n# ------    Test Internode Dispatch   ------ #\n", flush=True)
                    
                    # prepare dispatch args
                    if local_rank == 0:
                        print(f'[testing] Running with {"FP8" if isinstance(current_x, tuple) else "BF16"}, {"with" if with_topk else "without"} top-k (async={async_mode}, previous={previous_mode}) ...', flush=True, end='')
                    dispatch_args = {'x': current_x, 'num_tokens_per_rank': num_tokens_per_rank, 'num_tokens_per_rdma_rank': num_tokens_per_rdma_rank,  'is_token_in_rank': is_token_in_rank,
                                     'num_tokens_per_expert': num_tokens_per_expert, 'config': config, 'async_finish': async_mode}
                    if with_topk:
                        dispatch_args.update({'topk_idx': topk_idx, 'topk_weights': topk_weights_pure_rand if current_x is x_pure_rand else topk_weights})
                    if previous_mode:
                        dispatch_args.update({'previous_event': buffer.capture()})
                        
                    # dispatch
                    # recv_x: shape=[num_recv_tokens, hidden_dim]: the recv tokens for this rank (in rank order just like a2a output, while the boundary is indicated by rank_prefix_matrix)
                    # recv_topk_idx: shape=[num_recv_tokens, topk]: the local expert idx for this rank w.r.t. each recv token's topk list (-1 means not sent to this rank)
                    # recv_topk_weights: shape=[num_recv_tokens, topk]: the corr. weight for each recv token's topk list (if idx = -1, then weight = 0.)
                    # recv_num_tokens_per_expert_list: shape=[num_local_experts,]: the number of tokens to recv for each local expert in this rank
                    # handle: the tuple of some meta tensors that will be passed to combine or cached dispatch
                    # handle[0] (is_token_in_rank_handle)
                    # handle[1] (rdma_channel_prefix_matrix): shape=[num_rdma_ranks, num_channels]: TODO: what's this ?
                    # handle[2] (gbl_channel_prefix_matrix): shape=[num_ranks, num_channels]: TODO: what's this ?
                    # handle[3] (recv_rdma_channel_prefix_matrix): shape=[num_rdma_ranks, num_channels]: TODO: what's this ?
                    # handle[4] (recv_rdma_rank_prefix_sum): shape=[num_rdma_ranks,]: TODO: what's this ?
                    # handle[5] (recv_gbl_channel_prefix_matrix): shape=[num_ranks, num_channels]: TODO: what's this ?
                    # handle[6] (recv_gbl_rank_prefix_sum): shape=[num_ranks,]: TODO: what's this ?
                    # handle[7] (recv_src_meta): shape=[num_recv_tokens, sizeof(internode::SourceMeta)=8]: TODO: what's this ?
                    # handle[8] (send_rdma_head): shape=[num_tokens, num_rdma_ranks]: TODO: what's this ?
                    # handle[9] (send_nvl_head): shape=[num_rdma_recv_tokens, num_local_ranks]: TODO: what's this ?
                    recv_x, recv_topk_idx, recv_topk_weights, recv_num_tokens_per_expert_list, handle, event = buffer.dispatch(**dispatch_args)
                    
                    # wait
                    event.current_stream_wait() if async_mode else ()
                    
                    # print
                    (
                        is_token_in_rank_handle, # handle[0]
                        rdma_channel_prefix_matrix, # handle[1]
                        gbl_channel_prefix_matrix, # handle[2]
                        recv_rdma_channel_prefix_matrix, # handle[3]
                        recv_rdma_rank_prefix_sum, # handle[4]
                        recv_gbl_channel_prefix_matrix, # handle[5]
                        recv_gbl_rank_prefix_sum, # handle[6]
                        recv_src_meta, # handle[7]
                        send_rdma_head, # handle[8]
                        send_nvl_head, # handle[9]
                    ) = handle
                    if with_topk:
                        print(
                            (
                                f"\n[RANK {rank}]: {recv_x.shape=}\n"
                                f"{recv_topk_idx.shape=} | {recv_topk_idx=}\n"
                                f"{recv_topk_weights.shape=} | {recv_topk_weights=}\n"
                                f"{len(recv_num_tokens_per_expert_list)=} | {recv_num_tokens_per_expert_list=}\n"
                                f"{is_token_in_rank_handle.shape=} | {is_token_in_rank_handle=}\n" # handle[0]
                                f"{rdma_channel_prefix_matrix.shape=} | {rdma_channel_prefix_matrix=}\n" # handle[1]
                                f"{gbl_channel_prefix_matrix.shape=} | {gbl_channel_prefix_matrix=}\n" # handle[2]
                                f"{recv_rdma_channel_prefix_matrix.shape=} | {recv_rdma_channel_prefix_matrix=}\n" # handle[3]
                                f"{recv_rdma_rank_prefix_sum.shape=} | {recv_rdma_rank_prefix_sum=}\n" # handle[4]
                                f"{recv_gbl_channel_prefix_matrix.shape=} | {recv_gbl_channel_prefix_matrix=}\n" # handle[5]
                                f"{recv_gbl_rank_prefix_sum.shape=} | {recv_gbl_rank_prefix_sum=}\n" # handle[6]
                                f"{recv_src_meta.shape=} | {recv_src_meta=}\n" # handle[7]
                                f"{send_rdma_head.shape=} | {send_rdma_head=}\n" # handle[8]
                                f"{send_nvl_head.shape=} | {send_nvl_head=}\n" # handle[9]
                            )
                            , flush=True
                        )
                    else:
                        print(
                            (
                                f"\n[RANK {rank}]: {recv_x.shape=}\n"
                                f"{recv_topk_idx=}\n"
                                f"{recv_topk_weights=}\n"
                                f"{len(recv_num_tokens_per_expert_list)=} | {recv_num_tokens_per_expert_list=}\n"
                                
                            )
                            , flush=True
                        )
                    
                    # cast back from fp8
                    recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x

                    # check
                    recv_gbl_rank_prefix_sum = handle[-4]
                    assert gbl_num_tokens_per_rank[rank].item() == recv_x.size(0), f'{gbl_num_tokens_per_rank[rank].item()} != {recv_x.size(0)}'
                    assert gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist() == recv_num_tokens_per_expert_list
                    if current_x is not x_pure_rand:
                        check_data(recv_x, recv_gbl_rank_prefix_sum)
                    if with_topk:
                        # Check `topk_idx`
                        assert (recv_topk_idx.eq(-1) | ((recv_topk_idx >= 0) & (recv_topk_idx < (num_experts // num_ranks)))).sum().item() == recv_topk_idx.numel()
                        for i, count in enumerate(recv_num_tokens_per_expert_list):
                            assert recv_topk_idx.eq(i).sum().item() == count

                        # Check `topk_weights`
                        if current_x is not x_pure_rand:
                            recv_topk_weights[recv_topk_idx.eq(-1)] = recv_topk_weights.amax(dim=1, keepdim=True).expand_as(recv_topk_weights)[recv_topk_idx.eq(-1)]
                            check_data(recv_topk_weights, recv_gbl_rank_prefix_sum)
                            
                    if local_rank == 0:
                        print("\n# ------    Test Internode Cached Dispatch   ------ #\n", flush=True)

                    # Test cached dispatch (must without top-k staffs)
                    if not with_topk:
                        dispatch_args = {'x': current_x, 'handle': handle, 'config': config, 'async_finish': async_mode}
                        if previous_mode:
                            dispatch_args.update({'previous_event': buffer.capture()})
                        recv_x, _, _, _, _, event = buffer.dispatch(**dispatch_args)
                        event.current_stream_wait() if async_mode else ()
                        recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x
                        if current_x is not x_pure_rand:
                            check_data(recv_x, recv_gbl_rank_prefix_sum)

                    if local_rank == 0:
                        print("\n# ------    Test Internode Combine   ------ #\n", flush=True)

                    # prepare combine args
                    bias_0 = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
                    bias_1 = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
                    combine_args = {'x': recv_x, 'bias': (bias_0, bias_1), 'handle': handle, 'config': config, 'async_finish': async_mode}
                    if with_topk:
                        combine_args.update({'topk_weights': recv_topk_weights})
                    if previous_mode:
                        combine_args.update({'previous_event': buffer.capture()})
                        
                    # combine
                    # combined_x: shape=[num_tokens, hidden_size]: combined_x[i]: the ith token's sum-reduction result of top-k experts
                    # NOTE: the combined_x is assumed to be already scaled by topk_weights before combining, thus in kernel we don't have to multiply topk_weights
                    # combined_topk_weights: shape=[num_tokens, topk]: combined_topk_weights[i]: the ith token's sum-reduction weights
                    # NOTE: the topk_weights might not a valid probability distribution, thus here we might need combined_topk_weights to be normalized
                    combined_x, combined_topk_weights, event = buffer.combine(**combine_args)
                    
                    # wait
                    event.current_stream_wait() if async_mode else ()
                    
                    # print
                    if with_topk:
                        print(
                            (
                                f"\n[RANK {rank}]: {combined_x.shape=}\n"
                                f"{combined_topk_weights.shape=} | {combined_topk_weights=}\n"
                            )
                            , flush=True
                        )
                    else:
                        print(
                            (
                                f"\n[RANK {rank}]: {combined_x.shape=}\n"
                                f"{combined_topk_weights=}\n"
                            )
                            , flush=True
                        )
                    
                    # check
                    check_x = (combined_x.float() - bias_0.float() - bias_1.float()) / is_token_in_rank.sum(dim=1).unsqueeze(1)
                    ref_x = x_pure_rand if current_x is x_pure_rand else x
                    assert calc_diff(check_x, ref_x) < 5e-6
                    if with_topk:
                        check_topk_weights = combined_topk_weights if (current_x is x_pure_rand) else (combined_topk_weights / is_token_in_rank.sum(dim=1).unsqueeze(1))
                        ref_topk_weights = topk_weights_pure_rand if current_x is x_pure_rand else topk_weights
                        assert calc_diff(check_topk_weights, ref_topk_weights) < 1e-9

                    # For later tuning
                    dispatch_bf16_rdma_send_bytes = num_rdma_token_sent * hidden * 2
                    dispatch_bf16_nvl_recv_bytes = recv_x.numel() * 2
                    combine_bf16_nvl_send_bytes = dispatch_bf16_nvl_recv_bytes
                    combine_bf16_rdma_recv_bytes = dispatch_bf16_rdma_send_bytes

                    if local_rank == 0:
                        print(' passed', flush=True)
    if local_rank == 0:
        print('', flush=True)

    # sync before tuning
    torch.cuda.synchronize()
    dist.barrier()

    # Tune dispatch performance
    best_dispatch_results = None
    fp8_factor = (1 + 4 / 128) / 2
    for current_x in (x_e4m3, x):
        best_time, best_results = 1e10, None
        rdma_send_bytes = (dispatch_bf16_rdma_send_bytes * fp8_factor) if isinstance(current_x, tuple) else dispatch_bf16_rdma_send_bytes
        nvl_recv_bytes = (dispatch_bf16_nvl_recv_bytes * fp8_factor) if isinstance(current_x, tuple) else dispatch_bf16_nvl_recv_bytes
        for nvl_chunk_size in range(4, 45, 4):
            for rdma_chunk_size in range(4, 33, 4):
                config = deep_ep.Config(num_sms, nvl_chunk_size, nvl_buffer_size, rdma_chunk_size, rdma_buffer_size)
                tune_args = {'x': current_x, 'handle': handle, 'config': config}
                t, notify_t = bench_kineto(lambda: buffer.dispatch(**tune_args), ('dispatch', 'notify'))
                if t < best_time:
                    best_time, best_results = t, (num_sms, nvl_chunk_size, rdma_chunk_size, notify_t)
                if local_rank == 0:
                    print(f'[tuning] SMs {num_sms}, NVL chunk {nvl_chunk_size}, RDMA chunk {rdma_chunk_size}, transmit: {t * 1e6:.2f} us, notify: {notify_t * 1e6:.2f} us, BW: {rdma_send_bytes / 1e9 / t:.2f} GB/s (RDMA), {nvl_recv_bytes / 1e9 / t:.2f} GB/s (NVL) ', flush=True)
        if local_rank == 0:
            print(f'[tuning] Best dispatch ({"FP8" if isinstance(current_x, tuple) else "BF16"}): SMs {best_results[0]}, NVL chunk {best_results[1]}, RDMA chunk {best_results[2]}, transmit: {best_time * 1e6:.2f} us, notify: {best_results[3] * 1e6:.2f} us, BW: {rdma_send_bytes / 1e9 / best_time:.2f} GB/s (RDMA), {nvl_recv_bytes / 1e9 / best_time:.2f} GB/s (NVL)', flush=True)
            print('', flush=True)

        if isinstance(current_x, tuple):
            # Gather FP8 the best config from rank 0
            best_dispatch_results = torch.tensor([best_results[0], best_results[1], best_results[2]], dtype=torch.int32, device='cuda')
            all_best_fp8_results_list = [torch.zeros_like(best_dispatch_results) for _ in range(torch.distributed.get_world_size())]
            dist.all_gather(all_best_fp8_results_list, best_dispatch_results, group=group)
            best_dispatch_results = all_best_fp8_results_list[0].tolist()
    dispatch_config = deep_ep.Config(best_dispatch_results[0], best_dispatch_results[1], nvl_buffer_size, best_dispatch_results[2], rdma_buffer_size)

    dispatch_args = {'x': x, 'num_tokens_per_rank': num_tokens_per_rank, 'num_tokens_per_rdma_rank': num_tokens_per_rdma_rank,
                     'is_token_in_rank': is_token_in_rank, 'num_tokens_per_expert': num_tokens_per_expert,
                     'config': dispatch_config if dispatch_config is not None else config}
    recv_x, _, _, _, handle, _ = buffer.dispatch(**dispatch_args)

    # Tune combine performance
    best_time, best_results = 1e10, None
    for nvl_chunk_size in range(1, 8, 1):
        for rdma_chunk_size in range(12 if num_nodes == 2 else 8, 33, 4):
            config = deep_ep.Config(num_sms, nvl_chunk_size, nvl_buffer_size, rdma_chunk_size, rdma_buffer_size)
            tune_args = {'x': recv_x, 'handle': handle, 'config': config}
            t, notify_t = bench_kineto(lambda: buffer.combine(**tune_args), ('combine', 'notify'))
            if local_rank == 0:
                print(f'[tuning] SMs {num_sms}, NVL chunk {nvl_chunk_size}, RDMA chunk {rdma_chunk_size}, transmit: {t * 1e6:.2f} us, notify: {notify_t * 1e6:.2f} us, BW: {combine_bf16_rdma_recv_bytes / 1e9 / t:.2f} GB/s (RDMA), {combine_bf16_nvl_send_bytes / 1e9 / t:.2f} GB/s (NVL) ', flush=True)
                if t < best_time:
                    best_time, best_results = t, (num_sms, nvl_chunk_size, rdma_chunk_size, notify_t)

    if local_rank == 0:
        print(f'[tuning] Best combine: SMs {best_results[0]}, NVL chunk {best_results[1]}, RDMA chunk {best_results[2]}, transmit: {best_time * 1e6:.2f} us, notify: {best_results[3] * 1e6:.2f} us, BW: {combine_bf16_rdma_recv_bytes / 1e9 / best_time:.2f} GB/s (RDMA), {combine_bf16_nvl_send_bytes / 1e9 / best_time:.2f} GB/s (NVL)', flush=True)
        print('', flush=True)


# noinspection PyUnboundLocalVariable,PyShadowingNames
def test_loop(args: argparse.Namespace):
    num_tokens, hidden = args.num_tokens, args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts
    
    # init dist
    num_nodes, num_local_ranks, num_ranks, rank, local_rank, group, device, seed = setup_dist_env(seed_bias=lambda rank: rank)
    
    if args.test_ll_compatibility:
        ll_num_tokens, ll_hidden, ll_num_experts, ll_num_topk = 16, 5120, 256, 9

    num_sms = 24
    num_qps_per_rank = max(num_sms, ll_num_experts // num_ranks if args.test_ll_compatibility else 0)
    args.num_topk_groups = num_topk_groups = num_nodes
    
    num_nvl_bytes = int(2e9)
    num_rdma_bytes = int(1e9)
    
    if local_rank == 0:
        print(
            (
                f"[config] {num_nvl_bytes=} ({num_nvl_bytes / 1e9:.2f} GB) | {num_rdma_bytes=} ({num_rdma_bytes / 1e9:.2f} GB) | "
                f"{num_nodes=} | {num_ranks=} | {num_local_ranks=} | {group.size()=} | "
                f" {num_sms=} | {num_qps_per_rank=} | "
                f"{num_tokens=} | {hidden=} | {num_topk=} | {num_experts=} | {num_topk_groups=}\n\n\n"
            )
            , flush=True
        )

    buffer = deep_ep.Buffer(
        group, 
        num_nvl_bytes, 
        num_rdma_bytes, 
        low_latency_mode=args.test_ll_compatibility,
        num_qps_per_rank=num_qps_per_rank,
        explicitly_destroy=True
    )
    assert num_local_ranks == 8 and num_ranks > 8
    torch.manual_seed(rank)

    for i in (num_sms, ):
        test_main(args, i, local_rank, num_local_ranks, num_ranks, num_nodes, rank, buffer, group)
        if local_rank == 0:
            print('', flush=True)

    # Test compatibility with low latency functions
    if args.test_ll_compatibility:
        buffer.clean_low_latency_buffer(ll_num_tokens, ll_hidden, ll_num_experts)
        test_low_latency.test_main(ll_num_tokens, ll_hidden, ll_num_experts, ll_num_topk, rank, num_ranks, group, buffer, seed=1)

    # Destroy the buffer runtime and communication group
    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test internode EP kernels')
    parser.add_argument('--num-processes', type=int, default=8,
                       help='Number of processes to spawn (default: 8)')
    parser.add_argument('--num-tokens', type=int, default=4096,
                       help='Number of tokens (default: 4096)')
    parser.add_argument('--hidden', type=int, default=7168,
                       help='Hidden dimension size (default: 7168)')
    parser.add_argument('--num-topk-groups', type=int, default=None,
                       help='Number of top-k groups (default: `min(num_nodes, 4)`)')
    parser.add_argument('--num-topk', type=int, default=8,
                       help='Number of top-k experts (default: 8)')
    parser.add_argument('--num-experts', type=int, default=256,
                       help='Number of experts (default: 256')
    parser.add_argument('--test-ll-compatibility', action='store_true',
                        help='whether to test compatibility with low-latency kernels')
    args = parser.parse_args()
        
    args.test_ll_compatibility = os.environ.get('DEEPEP_TEST_INTERNODE_LL_COMPATIBILITY', args.test_ll_compatibility) == "1"

    num_processes = args.num_processes
    
    # torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)
    
    # launch using torchrun
    test_loop(args)