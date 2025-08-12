
import random
import torch
import torch.distributed as dist


def get_random_split_size_list(
    total_seqlen: int,
    num_splits: int,
) -> list[int]:
    cu_seqlens = [0] + sorted(random.sample(range(1, total_seqlen-1), num_splits - 1)) + [total_seqlen]
    seqlens = torch.tensor(cu_seqlens, dtype=torch.int).diff().tolist()
    return seqlens


def get_random_dst_indices_list(
    num_splits: int,
    num_ranks: int,
    allow_empty_dst: bool = False
) -> list[list[int]]:
    dst_indices_list = [[] for _ in range(num_splits)]
    num_dst_ranks_per_split = torch.randint(0 if allow_empty_dst else 1, num_ranks+1, (num_splits,)).tolist()
    
    for dst_indices, num_dst_ranks in zip(dst_indices_list, num_dst_ranks_per_split):
        dst_indices.extend(sorted(random.sample(range(num_ranks), num_dst_ranks)))
    return dst_indices_list


def get_output_split_size_list_and_src_index_list(
    input_split_size_list: list[int],
    dst_indices_list: list[list[int]],
    group: dist.ProcessGroup,
) -> tuple[list[int], list[int]]:
    my_rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    input_split_size_list_per_rank = [None] * world_size
    dst_indices_list_per_rank = [None] * world_size
    dist.all_gather_object(input_split_size_list_per_rank, input_split_size_list, group=group)
    dist.all_gather_object(dst_indices_list_per_rank, dst_indices_list, group=group)
    
    output_split_size_list, src_index_list = [], []
    
    output_src_rank_map = {}
    for src_rank in range(world_size):
        input_split_size_list_this_rank = input_split_size_list_per_rank[src_rank]
        dst_indices_list_this_rank = dst_indices_list_per_rank[src_rank]
        assert len(input_split_size_list_this_rank) == len(dst_indices_list_this_rank)
        
        for input_split_size, dst_indices in zip(input_split_size_list_this_rank, dst_indices_list_this_rank):
            if my_rank in dst_indices:
                output_src_rank_map.setdefault(src_rank, []).append(input_split_size)
        
    for src_rank in range(world_size):
        split_sizes = output_src_rank_map.get(src_rank, [])
        if split_sizes:
            output_split_size_list.extend(split_sizes)
            src_index_list.extend([src_rank] * len(split_sizes))
    
    return output_split_size_list, src_index_list


def transfer_group_cast_meta_to_dispatch_meta(
    rank: int,
    num_ranks: int,
    num_local_experts: int,
    input_split_size_list: list[int],
    dst_indices_list: list[list[int]],
    device: str = "cuda",
    dtype: torch.dtype = torch.int64,
    use_topk: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = sum(input_split_size_list)
    num_splits = len(input_split_size_list)
    
    if use_topk:
        assert num_local_experts == num_ranks
        topk_idxs = torch.full((num_splits, num_ranks), fill_value=-1, dtype=dtype, device="cpu")
        topk_weights = torch.ones((num_tokens, num_ranks), dtype=torch.float32, device=device) * rank
        for split_idx in range(num_splits):
            num_dst_ranks = len(dst_indices_list[split_idx])
            assert num_dst_ranks > 0, "For now, we only support non-empty dst_indices_list"
            num_dst_local_experts = num_ranks // num_dst_ranks
            num_last_dst_local_experts = num_ranks - num_dst_local_experts * (num_dst_ranks-1)
            is_last_dst_rank = lambda r: r == dst_indices_list[split_idx][-1]
            
            start = 0
            for dst_rank in dst_indices_list[split_idx]:
                num = num_last_dst_local_experts if is_last_dst_rank(dst_rank) else num_dst_local_experts
                end = start + num
                topk_idxs[split_idx][start:end] = dst_rank * num_ranks + torch.arange(num, dtype=dtype)
                start = end
        topk_idxs = topk_idxs.to(device).repeat_interleave(torch.tensor(input_split_size_list), dim=0, output_size=num_tokens) # shape=(num_tokens, num_ranks)
    else:
        assert num_local_experts == 1
        topk_idxs, topk_weights = None, None

    rank_idx = torch.full((num_splits, num_ranks), fill_value=-1, dtype=dtype, device="cpu")
    for split_idx in range(num_splits):
        num_dst_ranks = len(dst_indices_list[split_idx])
        assert num_dst_ranks > 0, "For now, we only support non-empty dst_indices_list"
        rank_idx[split_idx, :num_dst_ranks] = torch.tensor(sorted(dst_indices_list[split_idx], reverse=True))
    rank_idx = rank_idx.to(device).repeat_interleave(torch.tensor(input_split_size_list), dim=0, output_size=num_tokens) # shape=(num_tokens, num_ranks)

    return rank_idx, topk_idxs, topk_weights