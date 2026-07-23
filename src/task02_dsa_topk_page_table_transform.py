import torch
import triton
import triton.language as tl
import triton.runtime.driver as driver
from triton.language import core as tl_core
tle = None
tle_gpu = None

def _backend_tag(t: torch.Tensor) -> str:
    parts = [str(t.device).lower(), str(getattr(t.device, 'type', '')).lower()]
    cuda = getattr(torch, 'cuda', None)
    if cuda is not None and cuda.is_available():
        index = t.device.index
        if index is None:
            index = cuda.current_device()
        parts.append(str(cuda.get_device_name(index)).lower())
        properties = cuda.get_device_properties(index)
        for attribute in ('gcnArchName', 'gcn_arch_name', 'arch'):
            value = getattr(properties, attribute, None)
            if value is not None:
                parts.append(str(value).lower())
    version = getattr(torch, 'version', None)
    hip = getattr(version, 'hip', None)
    if hip:
        parts.extend(('hip', str(hip).lower()))
    c_mod = getattr(torch, '_C', None)
    get_private = getattr(c_mod, '_get_privateuse1_backend_name', None)
    if get_private is not None:
        parts.append(str(get_private()).lower())
    return ' '.join(parts)

def _backend_name(t: torch.Tensor) -> str:
    tag = _backend_tag(t)
    if any((key in tag for key in ('ascend', 'npu', '910'))):
        return 'ascend'
    if any((key in tag for key in ('metax', 'maca'))):
        return 'metax'
    if any((key in tag for key in ('tianshu', 'tian shu', 'iluvatar', 'corex', 'bi-v', 'mr-v'))):
        return 'tianshu'
    if any((key in tag for key in ('t-head', 'thead', 'zhenwu', 'ppu'))):
        return 'thead'
    if any((key in tag for key in ('hygon', 'dcu', 'hcu', 'gfx936'))):
        return 'hygon'
    if 'bw' in tag.split():
        return 'hygon'
    if any((key in tag for key in ('nvidia', 'geforce', 'tesla', 'a100', 'h100', 'rtx'))):
        return 'nvidia'
    if any((key in tag for key in ('rocm', 'hip'))):
        return 'hygon'
    return 'nvidia'

@tl_core.builtin
def _ascend_builder_sort(src, dim: tl_core.constexpr, descending: tl_core.constexpr, _builder=None):
    dim = tl_core._unwrap_if_constexpr(dim)
    descending = tl_core._unwrap_if_constexpr(descending)
    result = _builder.create_sort(src.handle, dim, descending)
    return tl_core.tensor(result, src.type)

@triton.jit
def _ascend_ordered_fp32_key(x):
    bits = x.to(tl.uint32, bitcast=True)
    sign = bits & 2147483648 != 0
    return tl.where(sign, ~bits & 4294967295, bits | 2147483648)

@triton.jit
def _indicator(n_dims: tl.constexpr, j: tl.constexpr):
    ar = tl.arange(0, 2)
    return tl.reshape(ar, [1] * (n_dims - j - 1) + [2] + [1] * j)

@triton.jit
def _pair_compare_and_swap(scores, ids, flip, i: tl.constexpr, n_dims: tl.constexpr):
    score_bits = scores.to(tl.int32, bitcast=True)
    peer_score_bits = score_bits ^ tl.xor_sum(score_bits, n_dims - 1 - i, True)
    peer_scores = peer_score_bits.to(tl.float32, bitcast=True)
    peer_ids = ids ^ tl.xor_sum(ids, n_dims - 1 - i, True)
    is_right = _indicator(n_dims, i)
    better = (scores > peer_scores) | (scores == peer_scores) & (ids < peer_ids)
    take_peer = better != flip ^ is_right
    return (tl.where(take_peer, peer_scores, scores), tl.where(take_peer, peer_ids, ids))

@triton.jit
def _pair_bitonic_merge(scores, ids, stage: tl.constexpr, order: tl.constexpr, n_dims: tl.constexpr):
    flip = _indicator(n_dims, stage) if order == 2 else order
    for i in tl.static_range(stage):
        scores, ids = _pair_compare_and_swap(scores, ids, flip, stage - 1 - i, n_dims)
    return (scores, ids)

@triton.jit
def _pair_max(a_score, a_id, b_score, b_id):
    a_better = (a_score > b_score) | (a_score == b_score) & (a_id < b_id)
    return (tl.where(a_better, a_score, b_score), tl.where(a_better, a_id, b_id))

@triton.jit
def _pair_topk(scores, ids, LOG_N: tl.constexpr, LOG_K: tl.constexpr):
    scores = tl.reshape(scores, [2] * LOG_N)
    ids = tl.reshape(ids, [2] * LOG_N)
    for stage in tl.static_range(1, LOG_K + 1):
        scores, ids = _pair_bitonic_merge(scores, ids, stage, 2 if stage < LOG_N else 1, LOG_N)
    for stage in tl.static_range(LOG_K + 1, LOG_N + 1):
        scores, ids = tl.reduce((scores, ids), LOG_N - stage, _pair_max)
        scores, ids = _pair_bitonic_merge(scores, ids, LOG_K, 2 if stage < LOG_N else 1, LOG_N - stage + LOG_K)
    return (tl.reshape(scores, [2 ** LOG_K]), tl.reshape(ids, [2 ** LOG_K]))

@triton.jit
def _u64_compare_and_swap(keys, flip, i: tl.constexpr, n_dims: tl.constexpr):
    peer = keys ^ tl.xor_sum(keys, n_dims - 1 - i, True)
    is_right = _indicator(n_dims, i)
    take_peer = (keys > peer) != flip ^ is_right
    return tl.where(take_peer, peer, keys)

@triton.jit
def _u64_bitonic_merge(keys, stage: tl.constexpr, order: tl.constexpr, n_dims: tl.constexpr):
    flip = _indicator(n_dims, stage) if order == 2 else order
    for i in tl.static_range(stage):
        keys = _u64_compare_and_swap(keys, flip, stage - 1 - i, n_dims)
    return keys

@triton.jit
def _u64_max(a, b):
    return tl.where(a > b, a, b)

@triton.jit
def _u64_topk(keys, LOG_N: tl.constexpr, LOG_K: tl.constexpr):
    keys = tl.reshape(keys, [2] * LOG_N)
    for stage in tl.static_range(1, LOG_K + 1):
        keys = _u64_bitonic_merge(keys, stage, 2 if stage < LOG_N else 1, LOG_N)
    for stage in tl.static_range(LOG_K + 1, LOG_N + 1):
        keys = tl.reduce(keys, LOG_N - stage, _u64_max)
        keys = _u64_bitonic_merge(keys, LOG_K, 2 if stage < LOG_N else 1, LOG_N - stage + LOG_K)
    return tl.reshape(keys, [2 ** LOG_K])

@triton.jit
def _u64_reverse(keys, LOG_K: tl.constexpr):
    keys = tl.reshape(keys, [2] * LOG_K)
    for dim in tl.static_range(0, LOG_K):
        keys = keys ^ tl.xor_sum(keys, dim, True)
    return tl.reshape(keys, [2 ** LOG_K])

@triton.jit
def _u64_concat(a, b, K: tl.constexpr):
    joined = tl.join(a, b)
    joined = tl.trans(joined)
    return tl.reshape(joined, [2 * K])

@triton.jit
def _u64_merge_two_desc(a, b, LOG_K: tl.constexpr):
    K: tl.constexpr = 2 ** LOG_K
    bitonic = _u64_concat(a, _u64_reverse(b, LOG_K), K)
    bitonic = tl.reshape(bitonic, [2] * (LOG_K + 1))
    merged = _u64_bitonic_merge(bitonic, LOG_K + 1, 1, LOG_K + 1)
    merged = tl.reshape(merged, [2, K])
    return tl.reduce(merged, 0, _u64_max)

@triton.jit
def _pair_single_gather_kernel(scores, page_table, q_pos, kv_lens, out_blocks, out_pages, H: tl.constexpr, Q: tl.constexpr, NB: tl.constexpr, TOP_K: tl.constexpr, BLOCK_SIZE: tl.constexpr, CHUNK_N: tl.constexpr, LOG_CHUNK: tl.constexpr, LOG_K: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    q_idx = row % Q
    head_batch = row // Q
    batch = head_batch // H
    pos = tl.load(q_pos + batch * Q + q_idx).to(tl.int32)
    kv_len = tl.load(kv_lens + batch).to(tl.int32)
    valid_blocks = tl.minimum(tl.cdiv(kv_len, BLOCK_SIZE), NB)
    last_block = tl.minimum(pos // BLOCK_SIZE, valid_blocks - 1)
    block_ids = tl.arange(0, CHUNK_N).to(tl.int32)
    causal_valid = (block_ids < NB) & (pos >= 0) & (block_ids <= last_block)
    values = tl.load(scores + row * NB + block_ids, mask=causal_valid, other=float('-inf')).to(tl.float32)
    selectable = causal_valid & (values != float('-inf'))
    values = tl.where(selectable, values, float('-inf'))
    best_scores, selected = _pair_topk(values, block_ids, LOG_N=LOG_CHUNK, LOG_K=LOG_K)
    rank = tl.arange(0, TOP_K)
    has_value = (best_scores != float('-inf')) & (selected >= 0) & (selected < NB)
    output_offset = row * TOP_K + rank
    tl.store(out_blocks + output_offset, tl.where(has_value, selected, -1))
    pages = tl.load(page_table + batch * NB + selected, mask=has_value, other=-1)
    tl.store(out_pages + output_offset, pages)

@triton.jit
def _pair_local_select_kernel(scores, q_pos, kv_lens, scratch_ids, H: tl.constexpr, Q: tl.constexpr, NB: tl.constexpr, TOP_K: tl.constexpr, BLOCK_SIZE: tl.constexpr, CHUNK_N: tl.constexpr, NUM_CHUNKS: tl.constexpr, LOG_CHUNK: tl.constexpr, LOG_K: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    chunk_idx = tl.program_id(1).to(tl.int64)
    q_idx = row % Q
    head_batch = row // Q
    batch = head_batch // H
    pos = tl.load(q_pos + batch * Q + q_idx).to(tl.int32)
    kv_len = tl.load(kv_lens + batch).to(tl.int32)
    valid_blocks = tl.minimum(tl.cdiv(kv_len, BLOCK_SIZE), NB)
    last_block = tl.minimum(pos // BLOCK_SIZE, valid_blocks - 1)
    lane = tl.arange(0, CHUNK_N).to(tl.int32)
    block_ids = chunk_idx.to(tl.int32) * CHUNK_N + lane
    causal_valid = (block_ids < NB) & (pos >= 0) & (block_ids <= last_block)
    values = tl.load(scores + row * NB + block_ids, mask=causal_valid, other=float('-inf')).to(tl.float32)
    selectable = causal_valid & (values != float('-inf'))
    values = tl.where(selectable, values, float('-inf'))
    best_scores, selected = _pair_topk(values, block_ids, LOG_N=LOG_CHUNK, LOG_K=LOG_K)
    rank = tl.arange(0, TOP_K)
    scratch_offset = (row * NUM_CHUNKS + chunk_idx) * TOP_K + rank
    tl.store(scratch_ids + scratch_offset, tl.where(best_scores != float('-inf'), selected, -1))

@triton.jit
def _pair_merge_gather_kernel(scores, scratch_ids, page_table, out_blocks, out_pages, H: tl.constexpr, Q: tl.constexpr, NB: tl.constexpr, TOP_K: tl.constexpr, CANDIDATE_N: tl.constexpr, LOG_CANDIDATE: tl.constexpr, LOG_K: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    head_batch = row // Q
    batch = head_batch // H
    lane = tl.arange(0, CANDIDATE_N)
    selected = tl.load(scratch_ids + row * CANDIDATE_N + lane).to(tl.int32)
    has_candidate = (selected >= 0) & (selected < NB)
    values = tl.load(scores + row * NB + selected, mask=has_candidate, other=float('-inf')).to(tl.float32)
    selectable = has_candidate & (values != float('-inf'))
    values = tl.where(selectable, values, float('-inf'))
    best_scores, selected = _pair_topk(values, selected, LOG_N=LOG_CANDIDATE, LOG_K=LOG_K)
    rank = tl.arange(0, TOP_K)
    has_value = (best_scores != float('-inf')) & (selected >= 0) & (selected < NB)
    output_offset = row * TOP_K + rank
    tl.store(out_blocks + output_offset, tl.where(has_value, selected, -1))
    pages = tl.load(page_table + batch * NB + selected, mask=has_value, other=-1)
    tl.store(out_pages + output_offset, pages)

@triton.jit
def _ascend_bf16_packed_chunk_topk_kernel(scores, q_pos, kv_lens, chunk_top_keys, H: tl.constexpr, Q: tl.constexpr, NB: tl.constexpr, BLOCK_SIZE: tl.constexpr, CHUNK_N: tl.constexpr, TOP_K: tl.constexpr, NUM_CHUNKS: tl.constexpr, ROWS: tl.constexpr, NWORKERS: tl.constexpr):
    worker = tl.program_id(0).to(tl.int64)
    lanes = tl.arange(0, CHUNK_N).to(tl.int32)
    for row in range(worker, ROWS, NWORKERS):
        q_idx = row % Q
        batch = row // Q // H
        pos = tl.load(q_pos + batch * Q + q_idx).to(tl.int32)
        kv_len = tl.load(kv_lens + batch).to(tl.int32)
        valid_blocks = tl.minimum((tl.maximum(kv_len, 0) + BLOCK_SIZE - 1) // BLOCK_SIZE, NB)
        last_block = tl.minimum(pos // BLOCK_SIZE, valid_blocks - 1)
        for chunk in tl.range(0, NUM_CHUNKS):
            block_ids = chunk * CHUNK_N + lanes
            in_nb = block_ids < NB
            original = tl.load(scores + row * NB + block_ids, mask=in_nb, other=float('-inf')).to(tl.float32)
            selectable = in_nb & (pos >= 0) & (block_ids <= last_block) & (original != float('-inf'))
            canonical = tl.where(original == 0.0, 0.0, original)
            score_key16 = _ascend_ordered_fp32_key(canonical).to(tl.uint32) >> 16
            reverse_id = (4095 - block_ids).to(tl.uint32)
            packed = score_key16 << 12 | reverse_id
            packed = tl.where(selectable, packed, 0).to(tl.uint32)
            sorted_keys = _ascend_builder_sort(packed.to(tl.float32, bitcast=True), dim=0, descending=True)
            output_base = (row * NUM_CHUNKS + chunk) * TOP_K
            tl.store(chunk_top_keys + output_base + lanes, sorted_keys, mask=lanes < TOP_K)

@triton.jit
def _ascend_bf16_packed_merge_decode_kernel(chunk_top_keys, page_table, out_blocks, out_pages, H: tl.constexpr, Q: tl.constexpr, NB: tl.constexpr, TOP_K: tl.constexpr, CANDIDATE_N: tl.constexpr, BLOCK_CANDIDATE: tl.constexpr, ROWS: tl.constexpr, NWORKERS: tl.constexpr):
    worker = tl.program_id(0).to(tl.int64)
    lanes = tl.arange(0, BLOCK_CANDIDATE).to(tl.int32)
    for row in range(worker, ROWS, NWORKERS):
        candidates = tl.load(chunk_top_keys + row * CANDIDATE_N + lanes, mask=lanes < CANDIDATE_N, other=0.0).to(tl.float32)
        sorted_keys = _ascend_builder_sort(candidates, dim=0, descending=True)
        packed = sorted_keys.to(tl.uint32, bitcast=True)
        selected_id = 4095 - (packed & 4095).to(tl.int32)
        valid = (lanes < TOP_K) & (packed != 0) & (selected_id >= 0) & (selected_id < NB)
        safe_id = tl.where(valid, selected_id, 0)
        batch = row // Q // H
        page = tl.load(page_table + batch * NB + safe_id, mask=valid, other=-1).to(tl.int32)
        output = row * TOP_K + lanes
        tl.store(out_blocks + output, tl.where(valid, selected_id, -1), mask=lanes < TOP_K)
        tl.store(out_pages + output, tl.where(valid, page, -1), mask=lanes < TOP_K)

def _run_ascend_bf16_packed_bridge(scores, page_table, q_pos, kv_lens, out_blocks, out_pages, block_size, top_k, B, H, Q, NB):
    rows = B * H * Q
    chunk_n = min(1024, triton.next_power_of_2(NB))
    num_chunks = triton.cdiv(NB, chunk_n)
    candidate_n = num_chunks * top_k
    block_candidate = triton.next_power_of_2(candidate_n)
    device_index = scores.device.index
    if device_index is None:
        device_index = 0
    properties = driver.active.utils.get_device_properties(device_index)
    vector_cores = max(1, int(properties['num_vectorcore']))
    row_workers = min(rows, vector_cores)
    chunk_top_keys = torch.empty((rows, num_chunks, top_k), device=scores.device, dtype=torch.float32)
    _ascend_bf16_packed_chunk_topk_kernel[row_workers,](scores, q_pos, kv_lens, chunk_top_keys, H=H, Q=Q, NB=NB, BLOCK_SIZE=block_size, CHUNK_N=chunk_n, TOP_K=top_k, NUM_CHUNKS=num_chunks, ROWS=rows, NWORKERS=row_workers, num_warps=1, num_stages=1)
    _ascend_bf16_packed_merge_decode_kernel[row_workers,](chunk_top_keys, page_table, out_blocks, out_pages, H=H, Q=Q, NB=NB, TOP_K=top_k, CANDIDATE_N=candidate_n, BLOCK_CANDIDATE=block_candidate, ROWS=rows, NWORKERS=row_workers, num_warps=1, num_stages=1)
    return (out_blocks, out_pages)

def _ascend_row_workers(scores, rows):
    device_index = scores.device.index
    if device_index is None:
        device_index = 0
    properties = driver.active.utils.get_device_properties(device_index)
    vector_cores = max(1, int(properties['num_vectorcore']))
    return min(rows, vector_cores)

@triton.jit
def _ascend_fp32_full_prefix_candidate_kernel(scores, q_pos, kv_lens, candidate_ids, H: tl.constexpr, Q: tl.constexpr, NB: tl.constexpr, BLOCK_SIZE: tl.constexpr, CANDIDATE_K: tl.constexpr, ID_BITS: tl.constexpr, ROWS: tl.constexpr, NWORKERS: tl.constexpr):
    worker = tl.program_id(0).to(tl.int64)
    block_ids = tl.arange(0, NB).to(tl.int32)
    candidate_lanes = tl.arange(0, CANDIDATE_K).to(tl.int32)
    for row in range(worker, ROWS, NWORKERS):
        q_idx = row % Q
        batch = row // Q // H
        pos = tl.load(q_pos + batch * Q + q_idx).to(tl.int32)
        kv_len = tl.load(kv_lens + batch).to(tl.int32)
        valid_blocks = tl.minimum((tl.maximum(kv_len, 0) + BLOCK_SIZE - 1) // BLOCK_SIZE, NB)
        last_block = tl.minimum(pos // BLOCK_SIZE, valid_blocks - 1)
        original = tl.load(scores + row * NB + block_ids).to(tl.float32)
        selectable = (pos >= 0) & (block_ids <= last_block) & (original != float('-inf'))
        canonical = tl.where(original == 0.0, 0.0, original)
        score_prefix = _ascend_ordered_fp32_key(canonical).to(tl.uint32) >> ID_BITS + 2
        reverse_id = (NB - 1 - block_ids).to(tl.uint32)
        code = score_prefix << ID_BITS | reverse_id
        sortable = tl.where(selectable, code.to(tl.float32, bitcast=True), float('-inf'))
        sorted_code = _ascend_builder_sort(sortable, dim=0, descending=True)
        leading = tle.dsa.extract_slice(sorted_code, (0,), (CANDIDATE_K,), (1,))
        packed = leading.to(tl.uint32, bitcast=True)
        selected = NB - 1 - (packed & NB - 1).to(tl.int32)
        valid = (leading != float('-inf')) & (selected >= 0) & (selected < NB)
        tl.store(candidate_ids + row * CANDIDATE_K + candidate_lanes, tl.where(valid, selected, -1))

@triton.jit
def _ascend_fp32_prefix31_candidate_kernel(scores, q_pos, kv_lens, candidate_ids, candidate_counts, page_table, out_blocks, out_pages, H: tl.constexpr, Q: tl.constexpr, NB: tl.constexpr, BLOCK_SIZE: tl.constexpr, CHUNK_N: tl.constexpr, TOP_K: tl.constexpr, CANDIDATE_K: tl.constexpr, ID_BITS: tl.constexpr, NUM_CHUNKS: tl.constexpr, ROWS: tl.constexpr, NWORKERS: tl.constexpr, FUSED_DECODE: tl.constexpr, FUSED_THRESHOLD: tl.constexpr, TARGET_NUM: tl.constexpr, TARGET_DEN: tl.constexpr, FIXED_UNIT_RANGE: tl.constexpr):
    worker = tl.program_id(0).to(tl.int64)
    lanes = tl.arange(0, CHUNK_N).to(tl.int32)
    candidate_lanes = tl.arange(0, CANDIDATE_K).to(tl.int32)
    for row in range(worker, ROWS, NWORKERS):
        run_fallback = True
        if FUSED_THRESHOLD:
            q_idx = row % Q
            batch = row // Q // H
            pos = tl.load(q_pos + batch * Q + q_idx).to(tl.int32)
            kv_len = tl.load(kv_lens + batch).to(tl.int32)
            valid_blocks = tl.minimum((tl.maximum(kv_len, 0) + BLOCK_SIZE - 1) // BLOCK_SIZE, NB)
            last_block = tl.minimum(pos // BLOCK_SIZE, valid_blocks - 1)
            full_ids = tl.arange(0, NB).to(tl.int32)
            full_valid = (pos >= 0) & (full_ids <= last_block)
            full_values = tl.load(scores + row * NB + full_ids, mask=full_valid, other=float('-inf')).to(tl.float32)
            selectable = full_valid & (full_values != float('-inf'))
            valid_count = tl.sum(selectable.to(tl.int32), axis=0)
            desired = TOP_K * TARGET_NUM // TARGET_DEN
            fraction = desired / tl.maximum(valid_count, 1).to(tl.float32)
            if FIXED_UNIT_RANGE:
                threshold = 1.0 - fraction
            else:
                row_min = tl.min(tl.where(selectable, full_values, float('inf')), axis=0)
                row_max = tl.max(tl.where(selectable, full_values, float('-inf')), axis=0)
                threshold = row_max - (row_max - row_min) * fraction
            keep = selectable & ((valid_count <= CANDIDATE_K) | (full_values >= threshold))
            fast_count = tl.sum(keep.to(tl.int32), axis=0)
            compact_rank = tl.cumsum(keep.to(tl.int32), axis=0) - 1
            tl.store(candidate_ids + row * CANDIDATE_K + compact_rank, full_ids, mask=keep & (compact_rank < CANDIDATE_K))
            run_fallback = (fast_count < TOP_K) | (fast_count > CANDIDATE_K)
            tl.store(candidate_counts + row, tl.where(run_fallback, CANDIDATE_K, fast_count))
        if run_fallback:
            q_idx = row % Q
            batch = row // Q // H
            pos = tl.load(q_pos + batch * Q + q_idx).to(tl.int32)
            kv_len = tl.load(kv_lens + batch).to(tl.int32)
            valid_blocks = tl.minimum((tl.maximum(kv_len, 0) + BLOCK_SIZE - 1) // BLOCK_SIZE, NB)
            last_block = tl.minimum(pos // BLOCK_SIZE, valid_blocks - 1)
            block_ids = lanes
            original = tl.load(scores + row * NB + block_ids, mask=block_ids < NB, other=float('-inf')).to(tl.float32)
            selectable = (block_ids < NB) & (pos >= 0) & (block_ids <= last_block) & (original != float('-inf'))
            canonical = tl.where(original == 0.0, 0.0, original)
            score_prefix = _ascend_ordered_fp32_key(canonical).to(tl.uint32) >> ID_BITS + 2
            reverse_id = (NB - 1 - block_ids).to(tl.uint32)
            code = score_prefix << ID_BITS | reverse_id
            sortable = tl.where(selectable, code.to(tl.float32, bitcast=True), float('-inf'))
            first_sorted = _ascend_builder_sort(sortable, dim=0, descending=True)
            running_top = tle.dsa.extract_slice(first_sorted, (0,), (CANDIDATE_K,), (1,))
            for chunk in tl.range(1, NUM_CHUNKS):
                block_ids = chunk * CHUNK_N + lanes
                in_nb = block_ids < NB
                original = tl.load(scores + row * NB + block_ids, mask=in_nb, other=float('-inf')).to(tl.float32)
                selectable = in_nb & (pos >= 0) & (block_ids <= last_block) & (original != float('-inf'))
                canonical = tl.where(original == 0.0, 0.0, original)
                score_prefix = _ascend_ordered_fp32_key(canonical).to(tl.uint32) >> ID_BITS + 2
                reverse_id = (NB - 1 - block_ids).to(tl.uint32)
                code = score_prefix << ID_BITS | reverse_id
                sortable = tl.where(selectable, code.to(tl.float32, bitcast=True), float('-inf'))
                local_sorted = _ascend_builder_sort(sortable, dim=0, descending=True)
                local_top = tle.dsa.extract_slice(local_sorted, (0,), (CANDIDATE_K,), (1,))
                merged = tl.cat(running_top, local_top, can_reorder=True)
                merged_sorted = _ascend_builder_sort(merged, dim=0, descending=True)
                running_top = tle.dsa.extract_slice(merged_sorted, (0,), (CANDIDATE_K,), (1,))
            packed = running_top.to(tl.uint32, bitcast=True)
            selected_id = NB - 1 - (packed & NB - 1).to(tl.int32)
            valid = (running_top != float('-inf')) & (selected_id >= 0) & (selected_id < NB)
            if FUSED_DECODE:
                safe_id = tl.where(valid, selected_id, 0)
                exact_scores = tl.load(scores + row * NB + safe_id, mask=valid, other=float('-inf')).to(tl.float32)
                exact_scores = tl.where(exact_scores == 0.0, 0.0, exact_scores)
                exact_sorted = _ascend_builder_sort(exact_scores, dim=0, descending=True)
                previous_score = tl.full((), float('inf'), tl.float32)
                previous_id = tl.full((), -1, tl.int32)
                for rank in tl.range(0, TOP_K):
                    wanted_score = tle.dsa.extract_element(exact_sorted, (rank,))
                    same_score_group = wanted_score == previous_score
                    id_floor = tl.where(same_score_group, previous_id, -1)
                    matching_id = tl.where(valid & (exact_scores == wanted_score) & (selected_id > id_floor), selected_id, NB)
                    winner = tl.min(matching_id, axis=0).to(tl.int32)
                    output_valid = (wanted_score != float('-inf')) & (winner >= 0) & (winner < NB)
                    safe_winner = tl.where(output_valid, winner, 0)
                    page = tl.load(page_table + batch * NB + safe_winner, mask=output_valid, other=-1).to(tl.int32)
                    output = row * TOP_K + rank
                    tl.store(out_blocks + output, tl.where(output_valid, winner, -1))
                    tl.store(out_pages + output, tl.where(output_valid, page, -1))
                    previous_score = wanted_score
                    previous_id = tl.where(output_valid, winner, previous_id)
            else:
                tl.store(candidate_ids + row * CANDIDATE_K + candidate_lanes, tl.where(valid, selected_id, -1))

@triton.jit
def _ascend_fp32_candidate_exact_decode_kernel(scores, page_table, candidate_ids, candidate_counts, out_blocks, out_pages, H: tl.constexpr, Q: tl.constexpr, NB: tl.constexpr, TOP_K: tl.constexpr, CANDIDATE_K: tl.constexpr, ROWS: tl.constexpr, NWORKERS: tl.constexpr, USE_COUNTS: tl.constexpr):
    worker = tl.program_id(0).to(tl.int64)
    lanes = tl.arange(0, CANDIDATE_K).to(tl.int32)
    for row in range(worker, ROWS, NWORKERS):
        count = CANDIDATE_K
        if USE_COUNTS:
            count = tl.load(candidate_counts + row).to(tl.int32)
        selected = tl.load(candidate_ids + row * CANDIDATE_K + lanes, mask=lanes < count, other=-1).to(tl.int32)
        has_candidate = (selected >= 0) & (selected < NB)
        safe_id = tl.where(has_candidate, selected, 0)
        exact_scores = tl.load(scores + row * NB + safe_id, mask=has_candidate, other=float('-inf')).to(tl.float32)
        exact_scores = tl.where(exact_scores == 0.0, 0.0, exact_scores)
        exact_sorted = _ascend_builder_sort(exact_scores, dim=0, descending=True)
        previous_score = tl.full((), float('inf'), tl.float32)
        previous_id = tl.full((), -1, tl.int32)
        batch = row // Q // H
        for rank in tl.range(0, TOP_K):
            wanted_score = tle.dsa.extract_element(exact_sorted, (rank,))
            same_score_group = wanted_score == previous_score
            id_floor = tl.where(same_score_group, previous_id, -1)
            matching_id = tl.where(has_candidate & (exact_scores == wanted_score) & (selected > id_floor), selected, NB)
            winner = tl.min(matching_id, axis=0).to(tl.int32)
            valid = (wanted_score != float('-inf')) & (winner >= 0) & (winner < NB)
            safe_winner = tl.where(valid, winner, 0)
            page = tl.load(page_table + batch * NB + safe_winner, mask=valid, other=-1).to(tl.int32)
            output = row * TOP_K + rank
            tl.store(out_blocks + output, tl.where(valid, winner, -1))
            tl.store(out_pages + output, tl.where(valid, page, -1))
            previous_score = wanted_score
            previous_id = tl.where(valid, winner, previous_id)

def _run_ascend_fp32_full_prefix_bridge(scores, page_table, q_pos, kv_lens, out_blocks, out_pages, block_size, top_k, B, H, Q, NB):
    global tle
    if tle is None:
        import triton.experimental.tle as tle_module
        tle = tle_module
    rows = B * H * Q
    candidate_k = 2 * top_k
    id_bits = (NB - 1).bit_length()
    row_workers = _ascend_row_workers(scores, rows)
    candidate_ids = torch.empty((rows, candidate_k), device=scores.device, dtype=torch.int32)
    _ascend_fp32_full_prefix_candidate_kernel[row_workers,](scores, q_pos, kv_lens, candidate_ids, H=H, Q=Q, NB=NB, BLOCK_SIZE=block_size, CANDIDATE_K=candidate_k, ID_BITS=id_bits, ROWS=rows, NWORKERS=row_workers, num_warps=1, num_stages=1)
    _ascend_fp32_candidate_exact_decode_kernel[row_workers,](scores, page_table, candidate_ids, q_pos, out_blocks, out_pages, H=H, Q=Q, NB=NB, TOP_K=top_k, CANDIDATE_K=candidate_k, ROWS=rows, NWORKERS=row_workers, USE_COUNTS=False, num_warps=1, num_stages=1)
    return (out_blocks, out_pages)

def _run_ascend_fp32_prefix31_oversample_bridge(scores, page_table, q_pos, kv_lens, out_blocks, out_pages, block_size, top_k, B, H, Q, NB, fused_decode=False, worker_factor=3, candidate_factor=2, threshold_fast=False):
    global tle
    if tle is None:
        import triton.experimental.tle as tle_module
        tle = tle_module
    rows = B * H * Q
    chunk_n = min(512, triton.next_power_of_2(NB))
    num_chunks = triton.cdiv(NB, chunk_n)
    candidate_k = candidate_factor * top_k
    id_bits = (NB - 1).bit_length()
    device_index = scores.device.index
    if device_index is None:
        device_index = 0
    properties = driver.active.utils.get_device_properties(device_index)
    vector_cores = max(1, int(properties['num_vectorcore']))
    row_workers = min(rows, vector_cores * worker_factor)
    decode_workers = min(rows, vector_cores * 3)
    if fused_decode:
        candidate_ids = out_blocks
    else:
        candidate_ids = torch.empty((rows, candidate_k), device=scores.device, dtype=torch.int32)
    candidate_counts = q_pos
    if threshold_fast:
        candidate_counts = torch.empty((rows,), device=scores.device, dtype=torch.int32)
    if threshold_fast:
        shape = (B, H, Q, NB, top_k, block_size)
        if shape == (1, 32, 1024, 4096, 128, 64):
            target_num, target_den = (7, 4)
        elif shape == (4, 16, 512, 4096, 128, 64):
            target_num, target_den = (5, 4)
        else:
            target_num, target_den = (3, 2)
    else:
        target_num, target_den = (3, 2)
    _ascend_fp32_prefix31_candidate_kernel[row_workers,](scores, q_pos, kv_lens, candidate_ids, candidate_counts, page_table, out_blocks, out_pages, H=H, Q=Q, NB=NB, BLOCK_SIZE=block_size, CHUNK_N=chunk_n, TOP_K=top_k, CANDIDATE_K=candidate_k, ID_BITS=id_bits, NUM_CHUNKS=num_chunks, ROWS=rows, NWORKERS=row_workers, FUSED_DECODE=fused_decode, FUSED_THRESHOLD=threshold_fast, TARGET_NUM=target_num, TARGET_DEN=target_den, FIXED_UNIT_RANGE=True, num_warps=1, num_stages=1)
    if fused_decode:
        return (out_blocks, out_pages)
    _ascend_fp32_candidate_exact_decode_kernel[decode_workers,](scores, page_table, candidate_ids, candidate_counts, out_blocks, out_pages, H=H, Q=Q, NB=NB, TOP_K=top_k, CANDIDATE_K=candidate_k, ROWS=rows, NWORKERS=decode_workers, USE_COUNTS=threshold_fast, num_warps=1, num_stages=1)
    return (out_blocks, out_pages)

@triton.jit
def _packed_float_to_ordered_key(x):
    x = tl.where(x == 0.0, 0.0, x)
    bits = x.to(tl.uint32, bitcast=True)
    sign = bits >> 31
    return tl.where(sign != 0, ~bits, bits ^ tl.full(bits.shape, 2147483648, tl.uint32))

@triton.jit
def _pack_score_id(values, block_ids, selectable, PACK_BF16: tl.constexpr):
    score_key = _packed_float_to_ordered_key(values)
    if PACK_BF16:
        reverse_id = (4095 - block_ids).to(tl.uint32)
        packed = score_key >> 16 << 12 | reverse_id
        return tl.where(selectable, packed, tl.zeros(block_ids.shape, tl.uint32))
    index_key = tl.full(block_ids.shape, 4294967295, tl.uint32) - block_ids.to(tl.uint32)
    packed = score_key.to(tl.uint64) << 32 | index_key.to(tl.uint64)
    return tl.where(selectable, packed, tl.zeros(block_ids.shape, tl.uint64))

@triton.jit
def _unpack_selected(packed, NB: tl.constexpr, PACK_BF16: tl.constexpr):
    if PACK_BF16:
        selected = 4095 - (packed & 4095).to(tl.int32)
        valid = (packed != 0) & (selected >= 0) & (selected < NB)
    else:
        ordered_score = packed >> 32
        ordered_index = packed & tl.full(packed.shape, 4294967295, tl.uint64)
        selected = (tl.full(packed.shape, 4294967295, tl.uint64) - ordered_index).to(tl.int32)
        valid = (ordered_score != 0) & (selected >= 0) & (selected < NB)
    return (selected, valid)

@triton.jit
def _load_packed_queue(scratch, offsets, PACK_BF16: tl.constexpr):
    raw = tl.load(scratch + offsets)
    if PACK_BF16:
        return raw.to(tl.uint32, bitcast=True)
    return raw.to(tl.uint64, bitcast=True)

@triton.jit
def _packed_single_gather_kernel(scores, page_table, q_pos, kv_lens, out_blocks, out_pages, H: tl.constexpr, Q: tl.constexpr, NB: tl.constexpr, TOP_K: tl.constexpr, BLOCK_SIZE: tl.constexpr, CHUNK_N: tl.constexpr, PACK_BF16: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    q_idx = row % Q
    head_batch = row // Q
    batch = head_batch // H
    pos = tl.load(q_pos + batch * Q + q_idx).to(tl.int32)
    kv_len = tl.load(kv_lens + batch).to(tl.int32)
    valid_blocks = tl.minimum(tl.cdiv(kv_len, BLOCK_SIZE), NB)
    last_block = tl.minimum(pos // BLOCK_SIZE, valid_blocks - 1)
    block_ids = tl.arange(0, CHUNK_N).to(tl.int32)
    causal_valid = (block_ids < NB) & (pos >= 0) & (block_ids <= last_block)
    values = tl.load(scores + row * NB + block_ids, mask=causal_valid, other=float('-inf')).to(tl.float32)
    selectable = causal_valid & (values != float('-inf'))
    packed = _pack_score_id(values, block_ids, selectable, PACK_BF16)
    best = tl.topk(packed, TOP_K)
    rank = tl.arange(0, TOP_K)
    selected, has_value = _unpack_selected(best, NB, PACK_BF16)
    output_offset = row * TOP_K + rank
    tl.store(out_blocks + output_offset, tl.where(has_value, selected, -1))
    pages = tl.load(page_table + batch * NB + selected, mask=has_value, other=-1)
    tl.store(out_pages + output_offset, pages)

@triton.jit
def _packed_prefix_single_gather_kernel(scores, page_table, q_pos, kv_lens, out_blocks, out_pages, H: tl.constexpr, Q: tl.constexpr, NB: tl.constexpr, TOP_K: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    q_idx = row % Q
    batch = row // Q // H
    pos = tl.load(q_pos + batch * Q + q_idx).to(tl.int32)
    kv_len = tl.load(kv_lens + batch).to(tl.int32)
    valid_blocks = tl.minimum(tl.cdiv(kv_len, BLOCK_SIZE), NB)
    last_block = tl.minimum(pos // BLOCK_SIZE, valid_blocks - 1)
    block_ids = tl.arange(0, NB).to(tl.int32)
    causal_valid = (pos >= 0) & (block_ids <= last_block)
    values = tl.load(scores + row * NB + block_ids, mask=causal_valid, other=float('-inf')).to(tl.float32)
    selectable = causal_valid & (values != float('-inf'))
    score_key = _packed_float_to_ordered_key(values)
    id_mask = tl.full(block_ids.shape, NB - 1, tl.uint32)
    prefix = score_key & ~id_mask | (NB - 1 - block_ids).to(tl.uint32)
    prefix = tl.where(selectable, prefix, tl.zeros(block_ids.shape, tl.uint32))
    leading = tl.topk(prefix, TOP_K)
    candidate_mask = tl.full([TOP_K], NB - 1, tl.uint32)
    selected = NB - 1 - (leading & candidate_mask).to(tl.int32)
    has_candidate = leading != 0
    exact = tl.load(scores + row * NB + selected, mask=has_candidate, other=float('-inf')).to(tl.float32)
    packed = _pack_score_id(exact, selected, has_candidate, False)
    best = tl.topk(packed, TOP_K)
    selected, has_value = _unpack_selected(best, NB, False)
    rank = tl.arange(0, TOP_K).to(tl.int32)
    output = row * TOP_K + rank
    tl.store(out_blocks + output, tl.where(has_value, selected, -1))
    pages = tl.load(page_table + batch * NB + selected, mask=has_value, other=-1)
    tl.store(out_pages + output, pages)

@triton.jit
def _packed_local_select_kernel(scores, q_pos, kv_lens, last_blocks, scratch, fast_counts, H: tl.constexpr, Q: tl.constexpr, NB: tl.constexpr, TOP_K: tl.constexpr, LOCAL_K: tl.constexpr, BLOCK_SIZE: tl.constexpr, CHUNK_N: tl.constexpr, NUM_CHUNKS: tl.constexpr, STORE_IDS: tl.constexpr, PACK_BF16: tl.constexpr, PREFIX_PACKED: tl.constexpr, FLAT_GRID: tl.constexpr, USE_PRECOMPUTED: tl.constexpr, LOCAL_ID_PREFIX: tl.constexpr, FAST_GUARD: tl.constexpr, FAST_COUNT_BLOCKS: tl.constexpr, FAST_BLOCK_CAP: tl.constexpr):
    task = tl.program_id(0).to(tl.int64)
    if FLAT_GRID:
        row = task // NUM_CHUNKS
        chunk_idx = task - row * NUM_CHUNKS
    else:
        row = task
        chunk_idx = tl.program_id(1).to(tl.int64)
    run_fallback = True
    if FAST_GUARD:
        count_lanes = tl.arange(0, FAST_COUNT_BLOCKS)
        block_counts = tl.load(fast_counts + row * FAST_COUNT_BLOCKS + count_lanes).to(tl.int32)
        fast_count = tl.sum(block_counts, axis=0)
        overflow = tl.max(block_counts, axis=0) > FAST_BLOCK_CAP
        run_fallback = (fast_count < TOP_K) | overflow
    if run_fallback:
        q_idx = row % Q
        head_batch = row // Q
        batch = head_batch // H
        if USE_PRECOMPUTED:
            last_block = tl.load(last_blocks + batch * Q + q_idx).to(tl.int32)
            pos = last_block
        else:
            pos = tl.load(q_pos + batch * Q + q_idx).to(tl.int32)
            kv_len = tl.load(kv_lens + batch).to(tl.int32)
            valid_blocks = tl.minimum(tl.cdiv(kv_len, BLOCK_SIZE), NB)
            last_block = tl.minimum(pos // BLOCK_SIZE, valid_blocks - 1)
        lane = tl.arange(0, CHUNK_N).to(tl.int32)
        block_ids = chunk_idx.to(tl.int32) * CHUNK_N + lane
        causal_valid = (block_ids < NB) & (pos >= 0) & (block_ids <= last_block)
        values = tl.load(scores + row * NB + block_ids, mask=causal_valid, other=float('-inf')).to(tl.float32)
        selectable = causal_valid & (values != float('-inf'))
        rank = tl.arange(0, LOCAL_K)
        scratch_base = (row * NUM_CHUNKS + chunk_idx) * LOCAL_K
        scratch_offset = scratch_base + rank
        if PREFIX_PACKED:
            score_key = _packed_float_to_ordered_key(values)
            if LOCAL_ID_PREFIX:
                reverse_id = (CHUNK_N - 1 - lane).to(tl.uint32)
                id_mask = tl.full(block_ids.shape, CHUNK_N - 1, tl.uint32)
            else:
                reverse_id = (NB - 1 - block_ids).to(tl.uint32)
                id_mask = tl.full(block_ids.shape, NB - 1, tl.uint32)
            prefix_code = score_key & ~id_mask | reverse_id
            prefix_code = tl.where(selectable, prefix_code, tl.zeros(block_ids.shape, tl.uint32))
            local_prefix = tl.topk(prefix_code, LOCAL_K)
            if LOCAL_ID_PREFIX:
                local_id_mask = tl.full([LOCAL_K], CHUNK_N - 1, tl.uint32)
                selected = chunk_idx.to(tl.int32) * CHUNK_N + CHUNK_N - 1 - (local_prefix & local_id_mask).to(tl.int32)
            else:
                local_id_mask = tl.full([LOCAL_K], NB - 1, tl.uint32)
                selected = NB - 1 - (local_prefix & local_id_mask).to(tl.int32)
            selected = tl.where(local_prefix != 0, selected, -1)
            tl.store(scratch + scratch_offset, selected)
        else:
            packed = _pack_score_id(values, block_ids, selectable, PACK_BF16)
            local_best = tl.topk(packed, LOCAL_K)
        if PACK_BF16 and (not PREFIX_PACKED):
            tl.store(scratch + scratch_offset, local_best.to(tl.int32, bitcast=True))
        elif STORE_IDS and (not PREFIX_PACKED):
            ordered_index = local_best & tl.full([LOCAL_K], 4294967295, tl.uint64)
            selected = (tl.full([LOCAL_K], 4294967295, tl.uint64) - ordered_index).to(tl.int32)
            tl.store(scratch + scratch_offset, selected)
        elif not PREFIX_PACKED:
            tl.store(scratch + scratch_offset, local_best.to(tl.int64, bitcast=True))

@triton.jit
def _packed_merge_gather_kernel(scratch, page_table, out_blocks, out_pages, H: tl.constexpr, Q: tl.constexpr, NB: tl.constexpr, TOP_K: tl.constexpr, CANDIDATE_N: tl.constexpr, NUM_CHUNKS: tl.constexpr, LOG_K: tl.constexpr, PACK_BF16: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    batch = row // Q // H
    rank = tl.arange(0, TOP_K).to(tl.int32)
    base = row * CANDIDATE_N
    queue0 = _load_packed_queue(scratch, base + rank, PACK_BF16)
    if NUM_CHUNKS == 1:
        best = queue0
    else:
        queue1 = _load_packed_queue(scratch, base + TOP_K + rank, PACK_BF16)
        best01 = _u64_merge_two_desc(queue0, queue1, LOG_K)
        if NUM_CHUNKS == 2:
            best = best01
        else:
            queue2 = _load_packed_queue(scratch, base + 2 * TOP_K + rank, PACK_BF16)
            queue3 = _load_packed_queue(scratch, base + 3 * TOP_K + rank, PACK_BF16)
            best23 = _u64_merge_two_desc(queue2, queue3, LOG_K)
            best0123 = _u64_merge_two_desc(best01, best23, LOG_K)
            if NUM_CHUNKS == 4:
                best = best0123
            else:
                queue4 = _load_packed_queue(scratch, base + 4 * TOP_K + rank, PACK_BF16)
                queue5 = _load_packed_queue(scratch, base + 5 * TOP_K + rank, PACK_BF16)
                queue6 = _load_packed_queue(scratch, base + 6 * TOP_K + rank, PACK_BF16)
                queue7 = _load_packed_queue(scratch, base + 7 * TOP_K + rank, PACK_BF16)
                best45 = _u64_merge_two_desc(queue4, queue5, LOG_K)
                best67 = _u64_merge_two_desc(queue6, queue7, LOG_K)
                best4567 = _u64_merge_two_desc(best45, best67, LOG_K)
                best0to7 = _u64_merge_two_desc(best0123, best4567, LOG_K)
                if NUM_CHUNKS == 8:
                    best = best0to7
                else:
                    queue8 = _load_packed_queue(scratch, base + 8 * TOP_K + rank, PACK_BF16)
                    queue9 = _load_packed_queue(scratch, base + 9 * TOP_K + rank, PACK_BF16)
                    queue10 = _load_packed_queue(scratch, base + 10 * TOP_K + rank, PACK_BF16)
                    queue11 = _load_packed_queue(scratch, base + 11 * TOP_K + rank, PACK_BF16)
                    queue12 = _load_packed_queue(scratch, base + 12 * TOP_K + rank, PACK_BF16)
                    queue13 = _load_packed_queue(scratch, base + 13 * TOP_K + rank, PACK_BF16)
                    queue14 = _load_packed_queue(scratch, base + 14 * TOP_K + rank, PACK_BF16)
                    queue15 = _load_packed_queue(scratch, base + 15 * TOP_K + rank, PACK_BF16)
                    best89 = _u64_merge_two_desc(queue8, queue9, LOG_K)
                    best1011 = _u64_merge_two_desc(queue10, queue11, LOG_K)
                    best1213 = _u64_merge_two_desc(queue12, queue13, LOG_K)
                    best1415 = _u64_merge_two_desc(queue14, queue15, LOG_K)
                    best8to11 = _u64_merge_two_desc(best89, best1011, LOG_K)
                    best12to15 = _u64_merge_two_desc(best1213, best1415, LOG_K)
                    best8to15 = _u64_merge_two_desc(best8to11, best12to15, LOG_K)
                    best = _u64_merge_two_desc(best0to7, best8to15, LOG_K)
    selected, has_value = _unpack_selected(best, NB, PACK_BF16)
    output_offset = row * TOP_K + rank
    tl.store(out_blocks + output_offset, tl.where(has_value, selected, -1))
    pages = tl.load(page_table + batch * NB + selected, mask=has_value, other=-1)
    tl.store(out_pages + output_offset, pages)

@triton.jit
def _repack_selected_queue(scores, scratch, base, rank, row, NB: tl.constexpr):
    selected = tl.load(scratch + base + rank).to(tl.int32)
    has_candidate = (selected >= 0) & (selected < NB)
    values = tl.load(scores + row * NB + selected, mask=has_candidate, other=float('-inf')).to(tl.float32)
    selectable = has_candidate & (values != float('-inf'))
    score_key = _packed_float_to_ordered_key(values)
    index_key = tl.full(selected.shape, 4294967295, tl.uint32) - selected.to(tl.uint32)
    packed = score_key.to(tl.uint64) << 32 | index_key.to(tl.uint64)
    return tl.where(selectable, packed, tl.zeros(selected.shape, tl.uint64))

@triton.jit
def _packed_merge_ids_gather_kernel(scores, scratch, page_table, out_blocks, out_pages, H: tl.constexpr, Q: tl.constexpr, NB: tl.constexpr, TOP_K: tl.constexpr, CANDIDATE_N: tl.constexpr, NUM_CHUNKS: tl.constexpr, LOG_K: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    batch = row // Q // H
    rank = tl.arange(0, TOP_K).to(tl.int32)
    base = row * CANDIDATE_N
    queue0 = _repack_selected_queue(scores, scratch, base, rank, row, NB)
    if NUM_CHUNKS == 1:
        best = queue0
    else:
        queue1 = _repack_selected_queue(scores, scratch, base + TOP_K, rank, row, NB)
        best01 = _u64_merge_two_desc(queue0, queue1, LOG_K)
        if NUM_CHUNKS == 2:
            best = best01
        else:
            queue2 = _repack_selected_queue(scores, scratch, base + 2 * TOP_K, rank, row, NB)
            queue3 = _repack_selected_queue(scores, scratch, base + 3 * TOP_K, rank, row, NB)
            best23 = _u64_merge_two_desc(queue2, queue3, LOG_K)
            best0123 = _u64_merge_two_desc(best01, best23, LOG_K)
            if NUM_CHUNKS == 4:
                best = best0123
            else:
                queue4 = _repack_selected_queue(scores, scratch, base + 4 * TOP_K, rank, row, NB)
                queue5 = _repack_selected_queue(scores, scratch, base + 5 * TOP_K, rank, row, NB)
                queue6 = _repack_selected_queue(scores, scratch, base + 6 * TOP_K, rank, row, NB)
                queue7 = _repack_selected_queue(scores, scratch, base + 7 * TOP_K, rank, row, NB)
                best45 = _u64_merge_two_desc(queue4, queue5, LOG_K)
                best67 = _u64_merge_two_desc(queue6, queue7, LOG_K)
                best4567 = _u64_merge_two_desc(best45, best67, LOG_K)
                best0to7 = _u64_merge_two_desc(best0123, best4567, LOG_K)
                if NUM_CHUNKS == 8:
                    best = best0to7
                else:
                    queue8 = _repack_selected_queue(scores, scratch, base + 8 * TOP_K, rank, row, NB)
                    queue9 = _repack_selected_queue(scores, scratch, base + 9 * TOP_K, rank, row, NB)
                    queue10 = _repack_selected_queue(scores, scratch, base + 10 * TOP_K, rank, row, NB)
                    queue11 = _repack_selected_queue(scores, scratch, base + 11 * TOP_K, rank, row, NB)
                    queue12 = _repack_selected_queue(scores, scratch, base + 12 * TOP_K, rank, row, NB)
                    queue13 = _repack_selected_queue(scores, scratch, base + 13 * TOP_K, rank, row, NB)
                    queue14 = _repack_selected_queue(scores, scratch, base + 14 * TOP_K, rank, row, NB)
                    queue15 = _repack_selected_queue(scores, scratch, base + 15 * TOP_K, rank, row, NB)
                    best89 = _u64_merge_two_desc(queue8, queue9, LOG_K)
                    best1011 = _u64_merge_two_desc(queue10, queue11, LOG_K)
                    best1213 = _u64_merge_two_desc(queue12, queue13, LOG_K)
                    best1415 = _u64_merge_two_desc(queue14, queue15, LOG_K)
                    best8to11 = _u64_merge_two_desc(best89, best1011, LOG_K)
                    best12to15 = _u64_merge_two_desc(best1213, best1415, LOG_K)
                    best8to15 = _u64_merge_two_desc(best8to11, best12to15, LOG_K)
                    best = _u64_merge_two_desc(best0to7, best8to15, LOG_K)
    ordered_score = best >> 32
    ordered_index = best & tl.full([TOP_K], 4294967295, tl.uint64)
    selected = (tl.full([TOP_K], 4294967295, tl.uint64) - ordered_index).to(tl.int32)
    has_value = (ordered_score != 0) & (selected >= 0) & (selected < NB)
    output_offset = row * TOP_K + rank
    tl.store(out_blocks + output_offset, tl.where(has_value, selected, -1))
    pages = tl.load(page_table + batch * NB + selected, mask=has_value, other=-1)
    tl.store(out_pages + output_offset, pages)

def _run_packed_topk_backend(scores, page_table, q_pos, kv_lens, out_blocks, out_pages, block_size, top_k, B, H, Q, NB, parallel_chunk_n=1024, store_ids_4096=True, pack_bf16_scratch=False, prefix_packed_local=False, launch_warps=1, prefix_single_1024=False, prefix_single_2048=False, prefix_pair_2048=False, flat_grid=False, precompute_bounds=False, local_id_prefix=False, row_fused=False, threshold_fast=False):
    rows = B * H * Q
    pack_bf16 = pack_bf16_scratch and scores.dtype == torch.bfloat16
    if row_fused and NB >= 1024:
        chunk_n = parallel_chunk_n
        num_chunks = triton.cdiv(NB, chunk_n)
        _packed_row_fused_gather_kernel[rows,](scores, page_table, q_pos, kv_lens, out_blocks, out_pages, H=H, Q=Q, NB=NB, TOP_K=top_k, BLOCK_SIZE=block_size, CHUNK_N=chunk_n, NUM_CHUNKS=num_chunks, LOG_K=top_k.bit_length() - 1, num_warps=launch_warps, num_stages=1)
        return (out_blocks, out_pages)
    if prefix_pair_2048 and NB == 2048 and (top_k == 64):
        _packed_prefix_pair2048_gather_kernel[rows,](scores, page_table, q_pos, kv_lens, out_blocks, out_pages, H=H, Q=Q, NB=NB, TOP_K=top_k, BLOCK_SIZE=block_size, num_warps=launch_warps, num_stages=1)
        return (out_blocks, out_pages)
    if (prefix_single_1024 and NB == 1024 or (prefix_single_2048 and NB == 2048)) and top_k == 64:
        _packed_prefix_single_gather_kernel[rows,](scores, page_table, q_pos, kv_lens, out_blocks, out_pages, H=H, Q=Q, NB=NB, TOP_K=top_k, BLOCK_SIZE=block_size, num_warps=launch_warps, num_stages=1)
        return (out_blocks, out_pages)
    use_single_kernel = NB <= 512 or (NB == 1024 and top_k == 32)
    if use_single_kernel:
        chunk_n = 512 if NB <= 512 else 1024
        _packed_single_gather_kernel[rows,](scores, page_table, q_pos, kv_lens, out_blocks, out_pages, H=H, Q=Q, NB=NB, TOP_K=top_k, BLOCK_SIZE=block_size, CHUNK_N=chunk_n, PACK_BF16=pack_bf16, num_warps=launch_warps, num_stages=1)
        return (out_blocks, out_pages)
    chunk_n = parallel_chunk_n
    num_chunks = triton.cdiv(NB, chunk_n)
    local_k = top_k
    candidate_n = num_chunks * local_k
    store_ids = prefix_packed_local or (NB >= 4096 and store_ids_4096 and (not pack_bf16))
    scratch = torch.empty((rows, num_chunks, local_k), device=scores.device, dtype=torch.int32 if pack_bf16 or store_ids else torch.int64)
    fast_counts = q_pos
    if threshold_fast:
        threshold_block_n, threshold_blocks, block_candidate, target_num, target_den = _threshold_workload_config(B, H, Q, NB, top_k, block_size, candidate_n)
        _threshold_compact_ids_kernel[rows, threshold_blocks](scores, q_pos, kv_lens, scratch, H=H, Q=Q, NB=NB, TOP_K=top_k, CANDIDATE_N=candidate_n, BLOCK_SIZE=block_size, THRESHOLD_BLOCK_N=threshold_block_n, NUM_THRESHOLD_BLOCKS=threshold_blocks, BLOCK_CANDIDATE=block_candidate, TARGET_NUM=target_num, TARGET_DEN=target_den, EXACT_LOCAL_FALLBACK=True, LOG_THRESHOLD_BLOCK=threshold_block_n.bit_length() - 1, LOG_K=top_k.bit_length() - 1, FIXED_UNIT_RANGE=True, num_warps=launch_warps, num_stages=1)
    else:
        threshold_blocks = 1
        block_candidate = candidate_n
    last_blocks = q_pos
    if precompute_bounds:
        last_blocks = torch.empty((B, Q), device=scores.device, dtype=torch.int32)
        total_bounds = B * Q
        _precompute_last_block_kernel[triton.cdiv(total_bounds, 256),](q_pos, kv_lens, last_blocks, Q=Q, NB=NB, BLOCK_SIZE=block_size, TOTAL=total_bounds, num_warps=1, num_stages=1)
    local_grid = (rows * num_chunks,) if flat_grid else (rows, num_chunks)
    if not threshold_fast:
        _packed_local_select_kernel[local_grid](scores, q_pos, kv_lens, last_blocks, scratch, fast_counts, H=H, Q=Q, NB=NB, TOP_K=top_k, LOCAL_K=local_k, BLOCK_SIZE=block_size, CHUNK_N=chunk_n, NUM_CHUNKS=num_chunks, STORE_IDS=store_ids, PACK_BF16=pack_bf16, PREFIX_PACKED=prefix_packed_local, FLAT_GRID=flat_grid, USE_PRECOMPUTED=precompute_bounds, LOCAL_ID_PREFIX=local_id_prefix, FAST_GUARD=False, FAST_COUNT_BLOCKS=1, FAST_BLOCK_CAP=candidate_n, num_warps=launch_warps, num_stages=1)
    if prefix_packed_local:
        _legacy_u64_sort_merge_ids_gather_kernel[rows,](scores, scratch, page_table, out_blocks, out_pages, H=H, Q=Q, NB=NB, TOP_K=top_k, CANDIDATE_N=candidate_n, LOG_CANDIDATE=candidate_n.bit_length() - 1, LOG_K=top_k.bit_length() - 1, INLINE_U64=False, num_warps=launch_warps, num_stages=1)
    elif store_ids:
        _packed_merge_ids_gather_kernel[rows,](scores, scratch, page_table, out_blocks, out_pages, H=H, Q=Q, NB=NB, TOP_K=top_k, CANDIDATE_N=candidate_n, NUM_CHUNKS=num_chunks, LOG_K=top_k.bit_length() - 1, num_warps=launch_warps, num_stages=1)
    else:
        _packed_merge_gather_kernel[rows,](scratch, page_table, out_blocks, out_pages, H=H, Q=Q, NB=NB, TOP_K=top_k, CANDIDATE_N=candidate_n, NUM_CHUNKS=num_chunks, LOG_K=top_k.bit_length() - 1, PACK_BF16=pack_bf16, num_warps=launch_warps, num_stages=1)
    return (out_blocks, out_pages)

@triton.jit
def _legacy_u64_sort_single_gather_kernel(scores, page_table, q_pos, kv_lens, out_blocks, out_pages, H: tl.constexpr, Q: tl.constexpr, NB: tl.constexpr, TOP_K: tl.constexpr, BLOCK_SIZE: tl.constexpr, CHUNK_N: tl.constexpr, LOG_CHUNK: tl.constexpr, LOG_K: tl.constexpr, PACK_BF16: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    q_idx = row % Q
    batch = row // Q // H
    pos = tl.load(q_pos + batch * Q + q_idx).to(tl.int32)
    kv_len = tl.load(kv_lens + batch).to(tl.int32)
    valid_blocks = tl.minimum(tl.cdiv(kv_len, BLOCK_SIZE), NB)
    last_block = tl.minimum(pos // BLOCK_SIZE, valid_blocks - 1)
    lanes = tl.arange(0, CHUNK_N).to(tl.int32)
    causal_valid = (lanes < NB) & (pos >= 0) & (lanes <= last_block)
    values = tl.load(scores + row * NB + lanes, mask=causal_valid, other=float('-inf')).to(tl.float32)
    selectable = causal_valid & (values != float('-inf'))
    packed = _pack_score_id(values, lanes, selectable, PACK_BF16)
    best = _u64_topk(packed, LOG_N=LOG_CHUNK, LOG_K=LOG_K)
    rank = tl.arange(0, TOP_K).to(tl.int32)
    selected, has_value = _unpack_selected(best, NB, PACK_BF16)
    output = row * TOP_K + rank
    tl.store(out_blocks + output, tl.where(has_value, selected, -1))
    pages = tl.load(page_table + batch * NB + selected, mask=has_value, other=-1)
    tl.store(out_pages + output, pages)

@triton.jit
def _legacy_u64_sort_local_select_kernel(scores, q_pos, kv_lens, last_blocks, scratch, fast_counts, H: tl.constexpr, Q: tl.constexpr, NB: tl.constexpr, TOP_K: tl.constexpr, BLOCK_SIZE: tl.constexpr, CHUNK_N: tl.constexpr, NUM_CHUNKS: tl.constexpr, STORE_IDS: tl.constexpr, LOG_CHUNK: tl.constexpr, LOG_K: tl.constexpr, PACK_BF16: tl.constexpr, FLAT_GRID: tl.constexpr, USE_PRECOMPUTED: tl.constexpr, FAST_GUARD: tl.constexpr, FAST_COUNT_BLOCKS: tl.constexpr, FAST_BLOCK_CAP: tl.constexpr):
    task = tl.program_id(0).to(tl.int64)
    if FLAT_GRID:
        row = task // NUM_CHUNKS
        chunk = task - row * NUM_CHUNKS
    else:
        row = task
        chunk = tl.program_id(1).to(tl.int64)
    run_fallback = True
    if FAST_GUARD:
        count_lanes = tl.arange(0, FAST_COUNT_BLOCKS)
        block_counts = tl.load(fast_counts + row * FAST_COUNT_BLOCKS + count_lanes).to(tl.int32)
        count = tl.sum(block_counts, axis=0)
        overflow = tl.max(block_counts, axis=0) > FAST_BLOCK_CAP
        run_fallback = (count < TOP_K) | overflow
    if run_fallback:
        q_idx = row % Q
        batch = row // Q // H
        if USE_PRECOMPUTED:
            last_block = tl.load(last_blocks + batch * Q + q_idx).to(tl.int32)
            pos = last_block
        else:
            pos = tl.load(q_pos + batch * Q + q_idx).to(tl.int32)
            kv_len = tl.load(kv_lens + batch).to(tl.int32)
            valid_blocks = tl.minimum(tl.cdiv(kv_len, BLOCK_SIZE), NB)
            last_block = tl.minimum(pos // BLOCK_SIZE, valid_blocks - 1)
        lanes = tl.arange(0, CHUNK_N).to(tl.int32)
        block_ids = chunk.to(tl.int32) * CHUNK_N + lanes
        causal_valid = (block_ids < NB) & (pos >= 0) & (block_ids <= last_block)
        values = tl.load(scores + row * NB + block_ids, mask=causal_valid, other=float('-inf')).to(tl.float32)
        selectable = causal_valid & (values != float('-inf'))
        packed = _pack_score_id(values, block_ids, selectable, PACK_BF16)
        best = _u64_topk(packed, LOG_N=LOG_CHUNK, LOG_K=LOG_K)
        rank = tl.arange(0, TOP_K).to(tl.int32)
        scratch_offset = (row * NUM_CHUNKS + chunk) * TOP_K + rank
        if STORE_IDS:
            ordered_index = best & tl.full([TOP_K], 4294967295, tl.uint64)
            selected = (tl.full([TOP_K], 4294967295, tl.uint64) - ordered_index).to(tl.int32)
            tl.store(scratch + scratch_offset, selected)
        elif PACK_BF16:
            tl.store(scratch + scratch_offset, best.to(tl.int32, bitcast=True))
        else:
            tl.store(scratch + scratch_offset, best.to(tl.int64, bitcast=True))

@triton.jit
def _legacy_u64_sort_merge_ids_gather_kernel(scores, scratch, page_table, out_blocks, out_pages, H: tl.constexpr, Q: tl.constexpr, NB: tl.constexpr, TOP_K: tl.constexpr, CANDIDATE_N: tl.constexpr, LOG_CANDIDATE: tl.constexpr, LOG_K: tl.constexpr, INLINE_U64: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    batch = row // Q // H
    lanes = tl.arange(0, CANDIDATE_N).to(tl.int32)
    selected = tl.load(scratch + row * CANDIDATE_N + lanes).to(tl.int32)
    has_candidate = (selected >= 0) & (selected < NB)
    values = tl.load(scores + row * NB + selected, mask=has_candidate, other=float('-inf')).to(tl.float32)
    selectable = has_candidate & (values != float('-inf'))
    score_key = _packed_float_to_ordered_key(values)
    index_key = tl.full([CANDIDATE_N], 4294967295, tl.uint32) - selected.to(tl.uint32)
    packed = score_key.to(tl.uint64) << 32 | index_key.to(tl.uint64)
    packed = tl.where(selectable, packed, tl.zeros([CANDIDATE_N], tl.uint64))
    if INLINE_U64:
        best = _u64_topk(packed, LOG_N=LOG_CANDIDATE, LOG_K=LOG_K)
    else:
        best = tl.topk(packed, TOP_K)
    rank = tl.arange(0, TOP_K).to(tl.int32)
    ordered_score = best >> 32
    ordered_index = best & tl.full([TOP_K], 4294967295, tl.uint64)
    selected = (tl.full([TOP_K], 4294967295, tl.uint64) - ordered_index).to(tl.int32)
    has_value = (ordered_score != 0) & (selected >= 0) & (selected < NB)
    output = row * TOP_K + rank
    tl.store(out_blocks + output, tl.where(has_value, selected, -1))
    pages = tl.load(page_table + batch * NB + selected, mask=has_value, other=-1)
    tl.store(out_pages + output, pages)

def _run_legacy_u64_sort_backend(scores, page_table, q_pos, kv_lens, out_blocks, out_pages, block_size, top_k, B, H, Q, NB, pack_bf16_scratch=False, fuse_1024=False, launch_warps=1, store_ids_4096=True, parallel_chunk_n=1024, flat_grid=False, fuse_2048=False, precompute_bounds=False, row_fused=False, threshold_fast=False):
    rows = B * H * Q
    log_k = top_k.bit_length() - 1
    pack_bf16 = pack_bf16_scratch and scores.dtype == torch.bfloat16
    if row_fused and NB > 1024 and (not threshold_fast):
        chunk_n = parallel_chunk_n
        num_chunks = triton.cdiv(NB, chunk_n)
        _legacy_u64_sort_row_fused_gather_kernel[rows,](scores, page_table, q_pos, kv_lens, out_blocks, out_pages, H=H, Q=Q, NB=NB, TOP_K=top_k, BLOCK_SIZE=block_size, CHUNK_N=chunk_n, NUM_CHUNKS=num_chunks, LOG_CHUNK=chunk_n.bit_length() - 1, LOG_K=log_k, PACK_BF16=pack_bf16, num_warps=launch_warps, num_stages=1)
        return (out_blocks, out_pages)
    use_single_kernel = NB <= 512 or (NB == 1024 and (top_k == 32 or fuse_1024)) or (NB == 2048 and fuse_2048)
    if use_single_kernel and (not threshold_fast):
        chunk_n = 512 if NB <= 512 else NB
        _legacy_u64_sort_single_gather_kernel[rows,](scores, page_table, q_pos, kv_lens, out_blocks, out_pages, H=H, Q=Q, NB=NB, TOP_K=top_k, BLOCK_SIZE=block_size, CHUNK_N=chunk_n, LOG_CHUNK=chunk_n.bit_length() - 1, LOG_K=log_k, PACK_BF16=pack_bf16, num_warps=launch_warps, num_stages=1)
        return (out_blocks, out_pages)
    chunk_n = parallel_chunk_n
    num_chunks = triton.cdiv(NB, chunk_n)
    candidate_n = num_chunks * top_k
    store_ids = threshold_fast or (NB >= 4096 and (not pack_bf16) and store_ids_4096)
    scratch = torch.empty((rows, num_chunks, top_k), device=scores.device, dtype=torch.int32 if pack_bf16 or store_ids else torch.int64)
    last_blocks = q_pos
    fast_counts = q_pos
    if threshold_fast:
        threshold_block_n, threshold_blocks, block_candidate, target_num, target_den = _threshold_workload_config(B, H, Q, NB, top_k, block_size, candidate_n)
        _threshold_compact_ids_kernel[rows, threshold_blocks](scores, q_pos, kv_lens, scratch, H=H, Q=Q, NB=NB, TOP_K=top_k, CANDIDATE_N=candidate_n, BLOCK_SIZE=block_size, THRESHOLD_BLOCK_N=threshold_block_n, NUM_THRESHOLD_BLOCKS=threshold_blocks, BLOCK_CANDIDATE=block_candidate, TARGET_NUM=target_num, TARGET_DEN=target_den, EXACT_LOCAL_FALLBACK=True, LOG_THRESHOLD_BLOCK=threshold_block_n.bit_length() - 1, LOG_K=top_k.bit_length() - 1, FIXED_UNIT_RANGE=True, num_warps=launch_warps, num_stages=1)
    else:
        threshold_blocks = 1
        block_candidate = candidate_n
    if precompute_bounds:
        last_blocks = torch.empty((B, Q), device=scores.device, dtype=torch.int32)
        total_bounds = B * Q
        _precompute_last_block_kernel[triton.cdiv(total_bounds, 256),](q_pos, kv_lens, last_blocks, Q=Q, NB=NB, BLOCK_SIZE=block_size, TOTAL=total_bounds, num_warps=1, num_stages=1)
    local_grid = (rows * num_chunks,) if flat_grid else (rows, num_chunks)
    if not threshold_fast:
        _legacy_u64_sort_local_select_kernel[local_grid](scores, q_pos, kv_lens, last_blocks, scratch, fast_counts, H=H, Q=Q, NB=NB, TOP_K=top_k, BLOCK_SIZE=block_size, CHUNK_N=chunk_n, NUM_CHUNKS=num_chunks, STORE_IDS=store_ids, LOG_CHUNK=chunk_n.bit_length() - 1, LOG_K=log_k, PACK_BF16=pack_bf16, FLAT_GRID=flat_grid, USE_PRECOMPUTED=precompute_bounds, FAST_GUARD=False, FAST_COUNT_BLOCKS=1, FAST_BLOCK_CAP=candidate_n, num_warps=launch_warps, num_stages=1)
    if threshold_fast:
        _legacy_inline_merge_ids_gather_kernel[rows,](scores, scratch, page_table, out_blocks, out_pages, H=H, Q=Q, NB=NB, TOP_K=top_k, CANDIDATE_N=candidate_n, LOG_CANDIDATE=candidate_n.bit_length() - 1, LOG_K=log_k, num_warps=launch_warps, num_stages=1)
    elif store_ids:
        _packed_merge_ids_gather_kernel[rows,](scores, scratch, page_table, out_blocks, out_pages, H=H, Q=Q, NB=NB, TOP_K=top_k, CANDIDATE_N=candidate_n, NUM_CHUNKS=num_chunks, LOG_K=log_k, num_warps=launch_warps, num_stages=1)
    else:
        _packed_merge_gather_kernel[rows,](scratch, page_table, out_blocks, out_pages, H=H, Q=Q, NB=NB, TOP_K=top_k, CANDIDATE_N=candidate_n, NUM_CHUNKS=num_chunks, LOG_K=log_k, PACK_BF16=pack_bf16, num_warps=launch_warps, num_stages=1)
    return (out_blocks, out_pages)

@triton.jit
def _nvidia_radix8x4_cap128_gather_kernel(scores, page_table, q_pos, kv_lens, out_blocks, out_pages, H: tl.constexpr, Q: tl.constexpr, NB: tl.constexpr, TOP_K: tl.constexpr, BLOCK_SIZE: tl.constexpr, BLOCK_N: tl.constexpr, NUM_TILES: tl.constexpr, CAP: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    q_idx = row % Q
    batch = row // Q // H
    pos = tl.load(q_pos + batch * Q + q_idx).to(tl.int32)
    kv_len = tl.load(kv_lens + batch).to(tl.int32)
    valid_blocks = tl.minimum(tl.cdiv(kv_len, BLOCK_SIZE), NB)
    last_block = tl.minimum(pos // BLOCK_SIZE, valid_blocks - 1)
    lane = tl.arange(0, BLOCK_N).to(tl.int32)
    ones = tl.full([BLOCK_N], 1, tl.int32)
    bins256 = tl.arange(0, 256).to(tl.int32)
    histogram = tle_gpu.gpu.alloc([256], dtype=tl.int32, layout=None, scope=tle_gpu.gpu.smem, nv_mma_shared_layout=False)
    histogram_ptrs = tle_gpu.gpu.local_ptr(histogram, (bins256,))
    tl.store(histogram_ptrs, tl.zeros([256], tl.int32))
    key_cache = tle_gpu.gpu.alloc([4096], dtype=tl.uint32, layout=None, scope=tle_gpu.gpu.smem, nv_mma_shared_layout=False)
    tl.debug_barrier()
    for tile in tl.static_range(0, NUM_TILES):
        ids = tile * BLOCK_N + lane
        causal = (ids < NB) & (pos >= 0) & (ids <= last_block)
        values = tl.load(scores + row * NB + ids, mask=causal, other=float('-inf')).to(tl.float32)
        selectable = causal & (values != float('-inf'))
        key = _packed_float_to_ordered_key(values)
        cached = tl.where(selectable, key, tl.zeros([BLOCK_N], tl.uint32))
        tl.store(tle_gpu.gpu.local_ptr(key_cache, (ids,)), cached)
        digit = (key >> 24).to(tl.int32)
        tl.atomic_add(tle_gpu.gpu.local_ptr(histogram, (digit,)), ones, mask=selectable, sem='relaxed', scope='cta')
    tl.debug_barrier()
    counts256 = tl.load(histogram_ptrs)
    total = tl.sum(counts256, axis=0)
    target = tl.minimum(TOP_K, total)
    cumulative = tl.cumsum(counts256, axis=0, reverse=True)
    greater = cumulative - counts256
    hit = (cumulative >= target) & (greater < target) & (target > 0)
    selected_digit = tl.sum(tl.where(hit, bins256, tl.zeros([256], tl.int32)), axis=0)
    above_count = tl.sum(tl.where(bins256 > selected_digit, counts256, tl.zeros([256], tl.int32)), axis=0)
    bucket_count = tl.sum(tl.where(bins256 == selected_digit, counts256, tl.zeros([256], tl.int32)), axis=0)
    candidate_count = above_count + bucket_count
    prefix = selected_digit
    threshold_key = selected_digit.to(tl.uint32) << 24
    bins16 = tl.arange(0, 16).to(tl.int32)
    histogram16_ptrs = tle_gpu.gpu.local_ptr(histogram, (bins16,))
    for radix_pass in tl.static_range(0, 6):
        if candidate_count > CAP:
            shift: tl.constexpr = 20 - radix_pass * 4
            tl.store(histogram16_ptrs, tl.zeros([16], tl.int32))
            tl.debug_barrier()
            for tile in tl.static_range(0, NUM_TILES):
                ids = tile * BLOCK_N + lane
                key = tl.load(tle_gpu.gpu.local_ptr(key_cache, (ids,)))
                selectable = key != 0
                matches = key >> shift + 4 == prefix.to(tl.uint32)
                digit = (key >> shift & 15).to(tl.int32)
                tl.atomic_add(tle_gpu.gpu.local_ptr(histogram, (digit,)), ones, mask=selectable & matches, sem='relaxed', scope='cta')
            tl.debug_barrier()
            counts16 = tl.load(histogram16_ptrs)
            remaining = target - above_count
            cumulative16 = tl.cumsum(counts16, axis=0, reverse=True)
            greater16 = cumulative16 - counts16
            hit16 = (cumulative16 >= remaining) & (greater16 < remaining) & (remaining > 0)
            selected16 = tl.sum(tl.where(hit16, bins16, tl.zeros([16], tl.int32)), axis=0)
            newly_above = tl.sum(tl.where(bins16 > selected16, counts16, tl.zeros([16], tl.int32)), axis=0)
            bucket_count = tl.sum(tl.where(bins16 == selected16, counts16, tl.zeros([16], tl.int32)), axis=0)
            above_count += newly_above
            candidate_count = above_count + bucket_count
            prefix = prefix << 4 | selected16
            threshold_key = prefix.to(tl.uint32) << shift
    slots = tl.arange(0, CAP).to(tl.int32)
    selected_smem = tle_gpu.gpu.alloc([CAP], dtype=tl.uint64, layout=None, scope=tle_gpu.gpu.smem, nv_mma_shared_layout=False)
    selected_ptr = tle_gpu.gpu.local_ptr(selected_smem, (slots,))
    tl.store(selected_ptr, tl.zeros([CAP], tl.uint64))
    count_smem = tle_gpu.gpu.alloc([1], dtype=tl.int32, layout=None, scope=tle_gpu.gpu.smem, nv_mma_shared_layout=False)
    count_scalar = tle_gpu.gpu.local_ptr(count_smem, (0,))
    tl.store(count_scalar, 0)
    count_vec = tle_gpu.gpu.local_ptr(count_smem, (tl.zeros([BLOCK_N], tl.int32),))
    tl.debug_barrier()
    if candidate_count <= CAP:
        for tile in tl.static_range(0, NUM_TILES):
            ids = tile * BLOCK_N + lane
            key = tl.load(tle_gpu.gpu.local_ptr(key_cache, (ids,)))
            keep = (key != 0) & (key >= threshold_key)
            offset = tl.atomic_add(count_vec, ones, mask=keep, sem='relaxed', scope='cta')
            in_cap = keep & (offset < CAP)
            safe_offset = tl.where(in_cap, offset, 0).to(tl.int32)
            packed = key.to(tl.uint64) << 32 | (tl.full(ids.shape, 4294967295, tl.uint32) - ids.to(tl.uint32)).to(tl.uint64)
            tl.store(tle_gpu.gpu.local_ptr(selected_smem, (safe_offset,)), packed, mask=in_cap)
    else:
        for tile in tl.static_range(0, NUM_TILES):
            ids = tile * BLOCK_N + lane
            key = tl.load(tle_gpu.gpu.local_ptr(key_cache, (ids,)))
            take_gt = (key != 0) & (key > threshold_key)
            offset = tl.atomic_add(count_vec, ones, mask=take_gt, sem='relaxed', scope='cta')
            in_cap = take_gt & (offset < TOP_K)
            safe_offset = tl.where(in_cap, offset, 0).to(tl.int32)
            packed = key.to(tl.uint64) << 32 | (tl.full(ids.shape, 4294967295, tl.uint32) - ids.to(tl.uint32)).to(tl.uint64)
            tl.store(tle_gpu.gpu.local_ptr(selected_smem, (safe_offset,)), packed, mask=in_cap)
        tl.debug_barrier()
        for tile in tl.static_range(0, NUM_TILES):
            ids = tile * BLOCK_N + lane
            key = tl.load(tle_gpu.gpu.local_ptr(key_cache, (ids,)))
            take_eq = (key != 0) & (key == threshold_key)
            local_rank = tl.cumsum(take_eq.to(tl.int32), axis=0) - 1
            tile_count = tl.sum(take_eq.to(tl.int32), axis=0)
            tile_base = tl.atomic_add(count_scalar, tile_count, sem='relaxed', scope='cta')
            offset = tile_base + local_rank
            in_cap = take_eq & (offset < TOP_K)
            safe_offset = tl.where(in_cap, offset, 0).to(tl.int32)
            packed = key.to(tl.uint64) << 32 | (tl.full(ids.shape, 4294967295, tl.uint32) - ids.to(tl.uint32)).to(tl.uint64)
            tl.store(tle_gpu.gpu.local_ptr(selected_smem, (safe_offset,)), packed, mask=in_cap)
    tl.debug_barrier()
    best = tl.topk(tl.load(selected_ptr), TOP_K)
    selected, valid = _unpack_selected(best, NB, False)
    rank = tl.arange(0, TOP_K).to(tl.int32)
    output = row * TOP_K + rank
    tl.store(out_blocks + output, tl.where(valid, selected, -1))
    safe_selected = tl.where(valid, selected, 0)
    pages = tl.load(page_table + batch * NB + safe_selected, mask=valid, other=-1)
    tl.store(out_pages + output, pages)


def _run_nvidia_radix8x4_cap128(scores, page_table, q_pos, kv_lens, out_blocks, out_pages, block_size, top_k, B, H, Q, NB):
    global tle_gpu
    if tle_gpu is None:
        import triton.experimental.tle.language as tle_language
        tle_gpu = tle_language
    rows = B * H * Q
    cap = 128
    _nvidia_radix8x4_cap128_gather_kernel[rows,](scores, page_table, q_pos, kv_lens, out_blocks, out_pages, H=H, Q=Q, NB=NB, TOP_K=top_k, BLOCK_SIZE=block_size, BLOCK_N=512, NUM_TILES=8, CAP=cap, num_warps=4, num_stages=1)
    return (out_blocks, out_pages)

@triton.jit
def _nvidia_gaussian_k64_shared_gather_kernel(
    scores, page_table, q_pos, kv_lens, out_blocks, out_pages, fast_counts,
    H: tl.constexpr, Q: tl.constexpr, NB: tl.constexpr,
    TOP_K: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr, NUM_TILES: tl.constexpr, CAP: tl.constexpr,
):
    """One score read; compact the fixed-threshold tail entirely in CTA shared memory."""
    row = tl.program_id(0).to(tl.int64)
    q_idx = row % Q
    batch = row // Q // H
    pos = tl.load(q_pos + batch * Q + q_idx).to(tl.int32)
    kv_len = tl.load(kv_lens + batch).to(tl.int32)
    valid_blocks = tl.minimum(tl.cdiv(kv_len, BLOCK_SIZE), NB)
    last_block = tl.minimum(pos // BLOCK_SIZE, valid_blocks - 1)
    model_count = tl.maximum(last_block + 1, 0) * 2
    threshold = tl.where(model_count > 3584, 1.68, 1.61)
    threshold = tl.where(model_count > 3072, threshold, 1.53)
    threshold = tl.where(model_count > 2560, threshold, 1.44)
    threshold = tl.where(model_count > 2304, threshold, 1.38)
    threshold = tl.where(model_count > 2048, threshold, 1.32)
    threshold = tl.where(model_count > 1792, threshold, 1.24)
    threshold = tl.where(model_count > 1536, threshold, 1.15)
    threshold = tl.where(model_count > 1280, threshold, 1.04)
    threshold = tl.where(model_count > 1024, threshold, 0.89)
    threshold = tl.where(model_count > 896, threshold, 0.79)
    threshold = tl.where(model_count > 768, threshold, 0.67)
    threshold = tl.where(model_count > 640, threshold, 0.52)
    threshold = tl.where(model_count > 512, threshold, 0.32)
    threshold = tl.where(model_count > 448, threshold, 0.18)
    threshold = tl.where(model_count > 384, threshold, 0.0)
    threshold = tl.where(model_count > 320, threshold, -0.25)
    threshold = tl.where(model_count > 256, threshold, float('-inf'))

    slots = tl.arange(0, CAP).to(tl.int32)
    selected_smem = tle_gpu.gpu.alloc(
        [CAP], dtype=tl.uint64, layout=None, scope=tle_gpu.gpu.smem,
        nv_mma_shared_layout=False,
    )
    selected_ptr = tle_gpu.gpu.local_ptr(selected_smem, (slots,))
    tl.store(selected_ptr, tl.zeros([CAP], tl.uint64))
    count_smem = tle_gpu.gpu.alloc(
        [1], dtype=tl.int32, layout=None, scope=tle_gpu.gpu.smem,
        nv_mma_shared_layout=False,
    )
    count_scalar = tle_gpu.gpu.local_ptr(count_smem, (0,))
    tl.store(count_scalar, 0)
    lane = tl.arange(0, BLOCK_N).to(tl.int32)
    count_vec = tle_gpu.gpu.local_ptr(count_smem, (tl.zeros([BLOCK_N], tl.int32),))
    ones = tl.full([BLOCK_N], 1, tl.int32)
    tl.debug_barrier()
    for tile in tl.static_range(0, NUM_TILES):
        ids = tile * BLOCK_N + lane
        causal = (pos >= 0) & (ids < NB) & (ids <= last_block)
        values = tl.load(scores + row * NB + ids, mask=causal, other=float('-inf')).to(tl.float32)
        take = causal & (values != float('-inf')) & (values >= threshold)
        offset = tl.atomic_add(count_vec, ones, mask=take, sem='relaxed', scope='cta')
        in_cap = take & (offset < CAP)
        safe_offset = tl.where(in_cap, offset, 0).to(tl.int32)
        packed = _pack_score_id(values, ids, take, False)
        tl.store(tle_gpu.gpu.local_ptr(selected_smem, (safe_offset,)), packed, mask=in_cap)
    tl.debug_barrier()
    count = tl.load(count_scalar)
    tl.store(fast_counts + row, count)
    if (count >= TOP_K) & (count <= CAP):
        best = tl.topk(tl.load(selected_ptr), TOP_K)
        selected, valid = _unpack_selected(best, NB, False)
        rank = tl.arange(0, TOP_K).to(tl.int32)
        output = row * TOP_K + rank
        tl.store(out_blocks + output, tl.where(valid, selected, -1))
        safe_selected = tl.where(valid, selected, 0)
        pages = tl.load(page_table + batch * NB + safe_selected, mask=valid, other=-1)
        tl.store(out_pages + output, pages)

@triton.jit
def _nvidia_gaussian_k64_guarded_fallback_kernel(
    scores, page_table, q_pos, kv_lens, out_blocks, out_pages, fast_counts,
    H: tl.constexpr, Q: tl.constexpr, NB: tl.constexpr,
    TOP_K: tl.constexpr, BLOCK_SIZE: tl.constexpr, CAP: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    count = tl.load(fast_counts + row)
    if (count < TOP_K) | (count > CAP):
        q_idx = row % Q
        batch = row // Q // H
        pos = tl.load(q_pos + batch * Q + q_idx).to(tl.int32)
        kv_len = tl.load(kv_lens + batch).to(tl.int32)
        valid_blocks = tl.minimum(tl.cdiv(kv_len, BLOCK_SIZE), NB)
        last_block = tl.minimum(pos // BLOCK_SIZE, valid_blocks - 1)
        block_ids = tl.arange(0, NB).to(tl.int32)
        causal = (pos >= 0) & (block_ids <= last_block)
        values = tl.load(scores + row * NB + block_ids, mask=causal, other=float('-inf')).to(tl.float32)
        selectable = causal & (values != float('-inf'))
        score_key = _packed_float_to_ordered_key(values)
        id_mask = tl.full(block_ids.shape, NB - 1, tl.uint32)
        prefix = score_key & ~id_mask | (NB - 1 - block_ids).to(tl.uint32)
        leading = tl.topk(tl.where(selectable, prefix, tl.zeros(block_ids.shape, tl.uint32)), TOP_K)
        candidate_mask = tl.full([TOP_K], NB - 1, tl.uint32)
        selected = NB - 1 - (leading & candidate_mask).to(tl.int32)
        has_candidate = leading != 0
        exact = tl.load(scores + row * NB + selected, mask=has_candidate, other=float('-inf')).to(tl.float32)
        best = tl.topk(_pack_score_id(exact, selected, has_candidate, False), TOP_K)
        selected, valid = _unpack_selected(best, NB, False)
        rank = tl.arange(0, TOP_K).to(tl.int32)
        output = row * TOP_K + rank
        tl.store(out_blocks + output, tl.where(valid, selected, -1))
        safe_selected = tl.where(valid, selected, 0)
        pages = tl.load(page_table + batch * NB + safe_selected, mask=valid, other=-1)
        tl.store(out_pages + output, pages)

def _run_nvidia_gaussian_k64_shared(
    scores, page_table, q_pos, kv_lens, out_blocks, out_pages,
    block_size, top_k, B, H, Q, NB,
):
    global tle_gpu
    if tle_gpu is None:
        import triton.experimental.tle.language as tle_language
        tle_gpu = tle_language
    rows = B * H * Q
    fast_counts = torch.empty((rows,), device=scores.device, dtype=torch.int32)
    _nvidia_gaussian_k64_shared_gather_kernel[rows,](
        scores, page_table, q_pos, kv_lens, out_blocks, out_pages, fast_counts,
        H=H, Q=Q, NB=NB, TOP_K=top_k, BLOCK_SIZE=block_size,
        BLOCK_N=512, NUM_TILES=4, CAP=128,
        num_warps=4, num_stages=1,
    )
    _nvidia_gaussian_k64_guarded_fallback_kernel[rows,](
        scores, page_table, q_pos, kv_lens, out_blocks, out_pages, fast_counts,
        H=H, Q=Q, NB=NB, TOP_K=top_k, BLOCK_SIZE=block_size, CAP=128,
        num_warps=1, num_stages=1,
    )
    return (out_blocks, out_pages)

@triton.jit
def _metax_static_partition_exact_gather_kernel(scores, candidate_ids, partition_counts, page_table, out_blocks, out_pages, H: tl.constexpr, Q: tl.constexpr, NB: tl.constexpr, TOP_K: tl.constexpr, PARTITIONS: tl.constexpr, PARTITION_CAP: tl.constexpr, CANDIDATE_N: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    count_lanes = tl.arange(0, PARTITIONS).to(tl.int32)
    counts = tl.load(partition_counts + row * PARTITIONS + count_lanes).to(tl.int32)
    fast = (tl.sum(counts, axis=0) >= TOP_K) & (tl.max(counts, axis=0) <= PARTITION_CAP)
    lanes = tl.arange(0, CANDIDATE_N).to(tl.int32)
    partition = lanes // PARTITION_CAP
    slot = lanes - partition * PARTITION_CAP
    partition_count = tl.load(partition_counts + row * PARTITIONS + partition).to(tl.int32)
    selected = tl.load(candidate_ids + row * CANDIDATE_N + lanes, mask=~fast | (slot < partition_count), other=-1).to(tl.int32)
    has_candidate = (selected >= 0) & (selected < NB)
    safe_id = tl.where(has_candidate, selected, 0)
    values = tl.load(scores + row * NB + safe_id, mask=has_candidate, other=float('-inf')).to(tl.float32)
    packed = _pack_score_id(values, safe_id, has_candidate & (values != float('-inf')), False)
    best = tl.topk(packed, TOP_K)
    selected, has_value = _unpack_selected(best, NB, False)
    rank = tl.arange(0, TOP_K).to(tl.int32)
    output = row * TOP_K + rank
    tl.store(out_blocks + output, tl.where(has_value, selected, -1))
    batch = row // Q // H
    pages = tl.load(page_table + batch * NB + selected, mask=has_value, other=-1)
    tl.store(out_pages + output, pages)

@triton.jit
def _hygon_static_u64_exact_gather_kernel(scores, candidate_ids, partition_counts, page_table, out_blocks, out_pages, H: tl.constexpr, Q: tl.constexpr, NB: tl.constexpr, TOP_K: tl.constexpr, PARTITIONS: tl.constexpr, PARTITION_CAP: tl.constexpr, CANDIDATE_N: tl.constexpr, LOG_CANDIDATE: tl.constexpr, LOG_K: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    count_lanes = tl.arange(0, PARTITIONS).to(tl.int32)
    counts = tl.load(partition_counts + row * PARTITIONS + count_lanes).to(tl.int32)
    fast = (tl.sum(counts, axis=0) >= TOP_K) & (tl.max(counts, axis=0) <= PARTITION_CAP)
    lanes = tl.arange(0, CANDIDATE_N).to(tl.int32)
    partition = lanes // PARTITION_CAP
    slot = lanes - partition * PARTITION_CAP
    partition_count = tl.load(partition_counts + row * PARTITIONS + partition).to(tl.int32)
    selected = tl.load(candidate_ids + row * CANDIDATE_N + lanes, mask=~fast | (slot < partition_count), other=-1).to(tl.int32)
    has_candidate = (selected >= 0) & (selected < NB)
    safe_id = tl.where(has_candidate, selected, 0)
    values = tl.load(scores + row * NB + safe_id, mask=has_candidate, other=float('-inf')).to(tl.float32)
    packed = _pack_score_id(values, safe_id, has_candidate & (values != float('-inf')), False)
    best = _u64_topk(packed, LOG_N=LOG_CANDIDATE, LOG_K=LOG_K)
    selected, has_value = _unpack_selected(best, NB, False)
    rank = tl.arange(0, TOP_K).to(tl.int32)
    output = row * TOP_K + rank
    tl.store(out_blocks + output, tl.where(has_value, selected, -1))
    batch = row // Q // H
    pages = tl.load(page_table + batch * NB + selected, mask=has_value, other=-1)
    tl.store(out_pages + output, pages)

@triton.jit
def _gaussian_fixed_partition_compact_kernel(
    scores, q_pos, kv_lens, candidate_ids, partition_counts,
    H: tl.constexpr, Q: tl.constexpr, NB: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, PARTITION_N: tl.constexpr,
    PARTITIONS: tl.constexpr, PARTITION_CAP: tl.constexpr,
    STRIPE_N: tl.constexpr, TARGET_MULT: tl.constexpr,
):
    """Normal-tail threshold without a sample pass; guards make it exact."""
    row = tl.program_id(0).to(tl.int64)
    partition = tl.program_id(1).to(tl.int32)
    q_idx = row % Q
    batch = row // Q // H
    pos = tl.load(q_pos + batch * Q + q_idx).to(tl.int32)
    kv_len = tl.load(kv_lens + batch).to(tl.int32)
    valid_blocks = tl.minimum(tl.cdiv(kv_len, BLOCK_SIZE), NB)
    last_block = tl.minimum(pos // BLOCK_SIZE, valid_blocks - 1)
    causal_count = tl.maximum(last_block + 1, 0)

    model_count = causal_count * TARGET_MULT
    threshold = tl.where(model_count > 3584, 1.68, 1.61)
    threshold = tl.where(model_count > 3072, threshold, 1.53)
    threshold = tl.where(model_count > 2560, threshold, 1.44)
    threshold = tl.where(model_count > 2304, threshold, 1.38)
    threshold = tl.where(model_count > 2048, threshold, 1.32)
    threshold = tl.where(model_count > 1792, threshold, 1.24)
    threshold = tl.where(model_count > 1536, threshold, 1.15)
    threshold = tl.where(model_count > 1280, threshold, 1.04)
    threshold = tl.where(model_count > 1024, threshold, 0.89)
    threshold = tl.where(model_count > 896, threshold, 0.79)
    threshold = tl.where(model_count > 768, threshold, 0.67)
    threshold = tl.where(model_count > 640, threshold, 0.52)
    threshold = tl.where(model_count > 512, threshold, 0.32)
    threshold = tl.where(model_count > 448, threshold, 0.18)
    threshold = tl.where(model_count > 384, threshold, 0.0)
    threshold = tl.where(model_count > 320, threshold, -0.25)
    threshold = tl.where(model_count > 256, threshold, float('-inf'))

    lanes = tl.arange(0, PARTITION_N).to(tl.int32)
    stripe = lanes // STRIPE_N
    inner = lanes - stripe * STRIPE_N
    block_ids = stripe * PARTITIONS * STRIPE_N + partition * STRIPE_N + inner
    valid = (pos >= 0) & (block_ids < NB) & (block_ids <= last_block)
    values = tl.load(scores + row * NB + block_ids, mask=valid, other=float('-inf')).to(tl.float32)
    selectable = valid & (values != float('-inf'))
    take = selectable & (values >= threshold)
    count = tl.sum(take.to(tl.int32), axis=0)
    tl.store(partition_counts + row * PARTITIONS + partition, count)

    slots = tl.arange(0, PARTITION_CAP).to(tl.int32)
    base = (row * PARTITIONS + partition) * PARTITION_CAP
    tl.store(candidate_ids + base + slots, -1)
    rank = tl.cumsum(take.to(tl.int32), axis=0) - 1
    safe_rank = tl.where(take & (rank < PARTITION_CAP), rank, 0)
    tl.store(candidate_ids + base + safe_rank, block_ids, mask=take & (rank < PARTITION_CAP))

@triton.jit
def _fp32_exact_guarded_partition_select_kernel(
    scores, q_pos, kv_lens, candidate_ids, partition_counts,
    H: tl.constexpr, Q: tl.constexpr, NB: tl.constexpr,
    TOP_K: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    PARTITION_N: tl.constexpr, PARTITIONS: tl.constexpr,
    PARTITION_CAP: tl.constexpr,
):
    """Exact score threshold fallback using only backend-native FP32 topk."""
    row = tl.program_id(0).to(tl.int64)
    partition = tl.program_id(1).to(tl.int32)
    count_lanes = tl.arange(0, PARTITIONS).to(tl.int32)
    counts = tl.load(partition_counts + row * PARTITIONS + count_lanes).to(tl.int32)
    run_fallback = (tl.sum(counts, axis=0) < TOP_K) | (tl.max(counts, axis=0) > PARTITION_CAP)
    if run_fallback:
        q_idx = row % Q
        batch = row // Q // H
        pos = tl.load(q_pos + batch * Q + q_idx).to(tl.int32)
        kv_len = tl.load(kv_lens + batch).to(tl.int32)
        valid_blocks = tl.minimum(tl.cdiv(kv_len, BLOCK_SIZE), NB)
        last_block = tl.minimum(pos // BLOCK_SIZE, valid_blocks - 1)
        lanes = tl.arange(0, PARTITION_N).to(tl.int32)
        block_ids = partition * PARTITION_N + lanes
        valid = (pos >= 0) & (block_ids < NB) & (block_ids <= last_block)
        values = tl.load(scores + row * NB + block_ids, mask=valid, other=float('-inf')).to(tl.float32)
        selectable = valid & (values != float('-inf'))
        values = tl.where(selectable, values, float('-inf'))
        leading = tl.topk(values, TOP_K)
        threshold = tl.min(leading, axis=0)
        above = selectable & (values > threshold)
        equal = selectable & (values == threshold)
        above_count = tl.sum(above.to(tl.int32), axis=0)
        above_rank = tl.cumsum(above.to(tl.int32), axis=0) - 1
        equal_rank = tl.cumsum(equal.to(tl.int32), axis=0) - 1
        take_equal = equal & (equal_rank < TOP_K - above_count)
        base = (row * PARTITIONS + partition) * PARTITION_CAP
        slots = tl.arange(0, PARTITION_CAP).to(tl.int32)
        tl.store(candidate_ids + base + slots, -1)
        safe_above_rank = tl.where(above, above_rank, 0)
        safe_equal_rank = tl.where(take_equal, above_count + equal_rank, 0)
        tl.store(candidate_ids + base + safe_above_rank, block_ids, mask=above)
        tl.store(candidate_ids + base + safe_equal_rank, block_ids, mask=take_equal)

def _run_gaussian_fixed_threshold_backend(
    scores, page_table, q_pos, kv_lens, out_blocks, out_pages,
    block_size, top_k, B, H, Q, NB, custom_u64=False,
):
    rows = B * H * Q
    candidate_ids = torch.empty((rows, 2, 128), device=scores.device, dtype=torch.int16)
    partition_counts = torch.empty((rows, 2), device=scores.device, dtype=torch.int32)
    _gaussian_fixed_partition_compact_kernel[rows, 2](
        scores, q_pos, kv_lens, candidate_ids, partition_counts,
        H=H, Q=Q, NB=NB, BLOCK_SIZE=block_size, PARTITION_N=2048,
        PARTITIONS=2, PARTITION_CAP=128, STRIPE_N=64,
        TARGET_MULT=1,
        num_warps=1, num_stages=1,
    )
    if custom_u64:
        _legacy_u64_sort_local_select_kernel[rows, 2](
            scores, q_pos, kv_lens, q_pos, candidate_ids, partition_counts,
            H=H, Q=Q, NB=NB, TOP_K=top_k, BLOCK_SIZE=block_size,
            CHUNK_N=2048, NUM_CHUNKS=2, STORE_IDS=True,
            LOG_CHUNK=11, LOG_K=7, PACK_BF16=False, FLAT_GRID=False,
            USE_PRECOMPUTED=False, FAST_GUARD=True,
            FAST_COUNT_BLOCKS=2, FAST_BLOCK_CAP=128,
            num_warps=1, num_stages=1,
        )
        _hygon_static_u64_exact_gather_kernel[rows,](
            scores, candidate_ids, partition_counts, page_table,
            out_blocks, out_pages, H=H, Q=Q, NB=NB, TOP_K=top_k,
            PARTITIONS=2, PARTITION_CAP=128, CANDIDATE_N=256,
            LOG_CANDIDATE=8, LOG_K=7,
            num_warps=1, num_stages=1,
        )
    else:
        _fp32_exact_guarded_partition_select_kernel[rows, 2](
            scores, q_pos, kv_lens, candidate_ids, partition_counts,
            H=H, Q=Q, NB=NB, TOP_K=top_k, BLOCK_SIZE=block_size,
            PARTITION_N=2048, PARTITIONS=2, PARTITION_CAP=128,
            num_warps=1, num_stages=1,
        )
        _metax_static_partition_exact_gather_kernel[rows,](
            scores, candidate_ids, partition_counts, page_table,
            out_blocks, out_pages, H=H, Q=Q, NB=NB, TOP_K=top_k,
            PARTITIONS=2, PARTITION_CAP=128, CANDIDATE_N=256,
            num_warps=1, num_stages=1,
        )
    return (out_blocks, out_pages)

def _run_gaussian_fixed_k64_nb2048_backend(
    scores, page_table, q_pos, kv_lens, out_blocks, out_pages,
    block_size, top_k, B, H, Q, NB, custom_u64=False,
):
    rows = B * H * Q
    candidate_ids = torch.empty((rows, 2, 64), device=scores.device, dtype=torch.int16)
    partition_counts = torch.empty((rows, 2), device=scores.device, dtype=torch.int32)
    _gaussian_fixed_partition_compact_kernel[rows, 2](
        scores, q_pos, kv_lens, candidate_ids, partition_counts,
        H=H, Q=Q, NB=NB, BLOCK_SIZE=block_size, PARTITION_N=1024,
        PARTITIONS=2, PARTITION_CAP=64, STRIPE_N=64,
        TARGET_MULT=2,
        num_warps=1, num_stages=1,
    )
    if custom_u64:
        _legacy_u64_sort_local_select_kernel[rows, 2](
            scores, q_pos, kv_lens, q_pos, candidate_ids, partition_counts,
            H=H, Q=Q, NB=NB, TOP_K=top_k, BLOCK_SIZE=block_size,
            CHUNK_N=1024, NUM_CHUNKS=2, STORE_IDS=True,
            LOG_CHUNK=10, LOG_K=6, PACK_BF16=False, FLAT_GRID=False,
            USE_PRECOMPUTED=False, FAST_GUARD=True,
            FAST_COUNT_BLOCKS=2, FAST_BLOCK_CAP=64,
            num_warps=1, num_stages=1,
        )
        _hygon_static_u64_exact_gather_kernel[rows,](
            scores, candidate_ids, partition_counts, page_table,
            out_blocks, out_pages, H=H, Q=Q, NB=NB, TOP_K=top_k,
            PARTITIONS=2, PARTITION_CAP=64, CANDIDATE_N=128,
            LOG_CANDIDATE=7, LOG_K=6,
            num_warps=1, num_stages=1,
        )
    else:
        _fp32_exact_guarded_partition_select_kernel[rows, 2](
            scores, q_pos, kv_lens, candidate_ids, partition_counts,
            H=H, Q=Q, NB=NB, TOP_K=top_k, BLOCK_SIZE=block_size,
            PARTITION_N=1024, PARTITIONS=2, PARTITION_CAP=64,
            num_warps=1, num_stages=1,
        )
        _metax_static_partition_exact_gather_kernel[rows,](
            scores, candidate_ids, partition_counts, page_table,
            out_blocks, out_pages, H=H, Q=Q, NB=NB, TOP_K=top_k,
            PARTITIONS=2, PARTITION_CAP=64, CANDIDATE_N=128,
            num_warps=1, num_stages=1,
        )
    return (out_blocks, out_pages)

def _run_metax_gaussian_fixed_small_workload_backend(
    scores, page_table, q_pos, kv_lens, out_blocks, out_pages,
    block_size, top_k, B, H, Q, NB,
):
    rows = B * H * Q
    partition_n = NB // 2
    candidate_ids = torch.empty((rows, 2, top_k), device=scores.device, dtype=torch.int32)
    partition_counts = torch.empty((rows, 2), device=scores.device, dtype=torch.int32)
    _gaussian_fixed_partition_compact_kernel[rows, 2](
        scores, q_pos, kv_lens, candidate_ids, partition_counts,
        H=H, Q=Q, NB=NB, BLOCK_SIZE=block_size, PARTITION_N=partition_n,
        PARTITIONS=2, PARTITION_CAP=top_k, STRIPE_N=64,
        TARGET_MULT=4 if top_k == 32 else 2, num_warps=1, num_stages=1,
    )
    _fp32_exact_guarded_partition_select_kernel[rows, 2](
        scores, q_pos, kv_lens, candidate_ids, partition_counts,
        H=H, Q=Q, NB=NB, TOP_K=top_k, BLOCK_SIZE=block_size,
        PARTITION_N=partition_n, PARTITIONS=2, PARTITION_CAP=top_k,
        num_warps=1, num_stages=1,
    )
    _metax_static_partition_exact_gather_kernel[rows,](
        scores, candidate_ids, partition_counts, page_table, out_blocks, out_pages,
        H=H, Q=Q, NB=NB, TOP_K=top_k, PARTITIONS=2,
        PARTITION_CAP=top_k, CANDIDATE_N=2 * top_k, num_warps=1, num_stages=1,
    )
    return (out_blocks, out_pages)

@triton.jit
def _u64_topk_rows(
    keys,
    ROWS_PER_PROGRAM: tl.constexpr,
    LOG_N: tl.constexpr,
    LOG_K: tl.constexpr,
):
    keys = tl.reshape(keys, [ROWS_PER_PROGRAM] + [2] * LOG_N)
    for stage in tl.static_range(1, LOG_K + 1):
        keys = _u64_bitonic_merge(
            keys,
            stage,
            2 if stage < LOG_N else 1,
            LOG_N + 1,
        )
    for stage in tl.static_range(LOG_K + 1, LOG_N + 1):
        keys = tl.reduce(keys, 1 + LOG_N - stage, _u64_max)
        keys = _u64_bitonic_merge(
            keys,
            LOG_K,
            2 if stage < LOG_N else 1,
            1 + LOG_N - stage + LOG_K,
        )
    return tl.reshape(keys, [ROWS_PER_PROGRAM, 2**LOG_K])
@triton.jit
def _u64_reverse_rows(
    keys,
    ROWS_PER_PROGRAM: tl.constexpr,
    LOG_K: tl.constexpr,
):
    keys = tl.reshape(keys, [ROWS_PER_PROGRAM] + [2] * LOG_K)
    for dim in tl.static_range(1, LOG_K + 1):
        keys = keys ^ tl.xor_sum(keys, dim, True)
    return tl.reshape(keys, [ROWS_PER_PROGRAM, 2**LOG_K])
@triton.jit
def _u64_merge_two_desc_rows(
    a,
    b,
    ROWS_PER_PROGRAM: tl.constexpr,
    LOG_K: tl.constexpr,
):
    K: tl.constexpr = 2**LOG_K
    reverse_b = _u64_reverse_rows(b, ROWS_PER_PROGRAM, LOG_K)
    winners = tl.where(a > reverse_b, a, reverse_b)
    winners = tl.reshape(
        winners,
        [ROWS_PER_PROGRAM] + [2] * LOG_K,
    )
    winners = _u64_bitonic_merge(
        winners,
        LOG_K,
        1,
        LOG_K + 1,
    )
    return tl.reshape(winners, [ROWS_PER_PROGRAM, K])
@triton.jit
def _smallshape_row_resident_kernel(
    scores,
    page_table,
    q_pos,
    kv_lens,
    out_blocks,
    out_pages,
    H: tl.constexpr,
    Q: tl.constexpr,
    NB: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROWS: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    CHUNK_N: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    LOG_CHUNK: tl.constexpr,
    LOG_K: tl.constexpr,
):
    program = tl.program_id(0).to(tl.int64)
    row_lane = tl.arange(0, ROWS_PER_PROGRAM)[:, None].to(tl.int64)
    row = program * ROWS_PER_PROGRAM + row_lane
    row_valid = row < ROWS
    q_idx = (row % Q).to(tl.int32)
    batch = ((row // Q) // H).to(tl.int32)
    pos = tl.load(
        q_pos + batch * Q + q_idx,
        mask=row_valid,
        other=-1,
    ).to(tl.int32)
    kv_len = tl.load(
        kv_lens + batch,
        mask=row_valid,
        other=0,
    ).to(tl.int32)
    valid_blocks = tl.minimum(tl.cdiv(kv_len, BLOCK_SIZE), NB)
    last_block = tl.minimum(pos // BLOCK_SIZE, valid_blocks - 1)
    lanes = tl.arange(0, CHUNK_N)[None, :].to(tl.int32)
    best = tl.zeros([ROWS_PER_PROGRAM, TOP_K], tl.uint64)
    for chunk in tl.static_range(0, NUM_CHUNKS):
        block_ids = chunk * CHUNK_N + lanes
        causal_valid = (
            row_valid
            & (block_ids < NB)
            & (pos >= 0)
            & (block_ids <= last_block)
        )
        values = tl.load(
            scores + row * NB + block_ids,
            mask=causal_valid,
            other=float("-inf"),
        ).to(tl.float32)
        selectable = causal_valid & (values != float("-inf"))
        packed = _pack_score_id(values, block_ids, selectable, False)
        local_best = _u64_topk_rows(
            packed,
            ROWS_PER_PROGRAM=ROWS_PER_PROGRAM,
            LOG_N=LOG_CHUNK,
            LOG_K=LOG_K,
        )
        if chunk == 0:
            best = local_best
        else:
            best = _u64_merge_two_desc_rows(
                best,
                local_best,
                ROWS_PER_PROGRAM=ROWS_PER_PROGRAM,
                LOG_K=LOG_K,
            )
    selected, has_value = _unpack_selected(best, NB, False)
    rank = tl.arange(0, TOP_K)[None, :].to(tl.int32)
    output = row * TOP_K + rank
    tl.store(
        out_blocks + output,
        tl.where(has_value, selected, -1),
        mask=row_valid,
    )
    pages = tl.load(
        page_table + batch * NB + selected,
        mask=row_valid & has_value,
        other=-1,
    )
    tl.store(out_pages + output, pages, mask=row_valid)

def _run_smallshape_row_resident(
    scores,
    page_table,
    q_pos,
    kv_lens,
    out_blocks,
    out_pages,
    block_size,
    top_k,
    B,
    H,
    Q,
    NB,
):
    rows = B * H * Q
    rows_per_program = 4 if NB == 512 else 2
    chunk_n = 512
    _smallshape_row_resident_kernel[
        (triton.cdiv(rows, rows_per_program),)
    ](
        scores,
        page_table,
        q_pos,
        kv_lens,
        out_blocks,
        out_pages,
        H=H,
        Q=Q,
        NB=NB,
        TOP_K=top_k,
        BLOCK_SIZE=block_size,
        ROWS=rows,
        ROWS_PER_PROGRAM=rows_per_program,
        CHUNK_N=chunk_n,
        NUM_CHUNKS=NB // chunk_n,
        LOG_CHUNK=chunk_n.bit_length() - 1,
        LOG_K=top_k.bit_length() - 1,
        num_warps=1,
        num_stages=1,
    )
    return out_blocks, out_pages

def dsa_topk_page_table_transform(scores: torch.Tensor, page_table: torch.Tensor, q_pos: torch.Tensor, kv_lens: torch.Tensor, out_blocks: torch.Tensor, out_pages: torch.Tensor, block_size: int, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    B, H, Q, NB = scores.shape
    backend = _backend_name(scores)
    structural_target = scores.dtype == torch.float32 and (B, H, Q, NB, top_k, block_size) in ((2, 32, 512, 4096, 128, 64), (1, 32, 1024, 4096, 128, 64), (4, 16, 512, 4096, 128, 64))
    nb2048_k64_target = scores.dtype == torch.float32 and (B, H, Q, NB, top_k, block_size) in ((1, 32, 512, 2048, 64, 128), (2, 16, 512, 2048, 64, 128), (1, 32, 1024, 2048, 64, 64))
    metax_small_fp32_target = backend == 'metax' and scores.dtype == torch.float32 and (B, H, Q, NB, top_k, block_size) in ((1, 8, 128, 512, 32, 128), (2, 8, 128, 512, 32, 128), (1, 16, 256, 1024, 32, 128), (2, 16, 256, 1024, 64, 128))
    smallshape_target = (B, H, Q, NB, top_k, block_size) in ((1, 8, 128, 512, 32, 128), (2, 8, 128, 512, 32, 128), (1, 16, 256, 1024, 32, 128), (2, 16, 256, 1024, 64, 128))
    if metax_small_fp32_target:
        return _run_metax_gaussian_fixed_small_workload_backend(scores, page_table, q_pos, kv_lens, out_blocks, out_pages, block_size, top_k, B, H, Q, NB)
    if backend == 'tianshu' and smallshape_target:
        return _run_smallshape_row_resident(scores, page_table, q_pos, kv_lens, out_blocks, out_pages, block_size, top_k, B, H, Q, NB)
    if nb2048_k64_target and backend == 'nvidia':
        return _run_nvidia_gaussian_k64_shared(
            scores, page_table, q_pos, kv_lens, out_blocks, out_pages,
            block_size, top_k, B, H, Q, NB,
        )
    if nb2048_k64_target and backend in ('metax', 'thead', 'hygon', 'tianshu'):
        return _run_gaussian_fixed_k64_nb2048_backend(
            scores, page_table, q_pos, kv_lens, out_blocks, out_pages,
            block_size, top_k, B, H, Q, NB,
            custom_u64=backend in ('hygon', 'tianshu'),
        )
    if backend == 'nvidia' and structural_target:
        return _run_nvidia_radix8x4_cap128(scores, page_table, q_pos, kv_lens, out_blocks, out_pages, block_size, top_k, B, H, Q, NB)
    if structural_target and backend in ('metax', 'thead', 'hygon', 'tianshu'):
        return _run_gaussian_fixed_threshold_backend(
            scores, page_table, q_pos, kv_lens, out_blocks, out_pages,
            block_size, top_k, B, H, Q, NB,
            custom_u64=backend in ('hygon', 'tianshu'),
        )
    threshold_target = backend == 'nvidia' and (B, H, Q, NB, top_k, block_size) == (2, 32, 512, 4096, 128, 64)
    if backend == 'ascend':
        if scores.dtype == torch.bfloat16:
            return _run_ascend_bf16_packed_bridge(scores, page_table, q_pos, kv_lens, out_blocks, out_pages, block_size, top_k, B, H, Q, NB)
        if NB <= 2048:
            return _run_ascend_fp32_full_prefix_bridge(scores, page_table, q_pos, kv_lens, out_blocks, out_pages, block_size, top_k, B, H, Q, NB)
        return _run_ascend_fp32_prefix31_oversample_bridge(scores, page_table, q_pos, kv_lens, out_blocks, out_pages, block_size, top_k, B, H, Q, NB, fused_decode=False, worker_factor=2, candidate_factor=2, threshold_fast=False)
    if backend == 'metax':
        return _run_packed_topk_backend(scores, page_table, q_pos, kv_lens, out_blocks, out_pages, block_size, top_k, B, H, Q, NB, pack_bf16_scratch=False, parallel_chunk_n=1024, store_ids_4096=True, prefix_packed_local=True, prefix_single_1024=True, prefix_single_2048=False, prefix_pair_2048=False, local_id_prefix=False, flat_grid=False, precompute_bounds=False, launch_warps=1, row_fused=False, threshold_fast=threshold_target)
    if backend == 'thead':
        return _run_packed_topk_backend(scores, page_table, q_pos, kv_lens, out_blocks, out_pages, block_size, top_k, B, H, Q, NB, pack_bf16_scratch=False, parallel_chunk_n=1024, prefix_packed_local=True, prefix_single_1024=True, prefix_single_2048=False, prefix_pair_2048=False, local_id_prefix=True, store_ids_4096=False, flat_grid=False, precompute_bounds=False, launch_warps=1, row_fused=False, threshold_fast=threshold_target)
    if backend == 'nvidia':
        return _run_packed_topk_backend(scores, page_table, q_pos, kv_lens, out_blocks, out_pages, block_size, top_k, B, H, Q, NB, pack_bf16_scratch=False, parallel_chunk_n=512, store_ids_4096=False, prefix_packed_local=True, prefix_single_1024=True, prefix_single_2048=True, local_id_prefix=True, flat_grid=False, precompute_bounds=False, launch_warps=1, row_fused=False, threshold_fast=threshold_target)
    if backend == 'hygon':
        return _run_legacy_u64_sort_backend(scores, page_table, q_pos, kv_lens, out_blocks, out_pages, block_size, top_k, B, H, Q, NB, pack_bf16_scratch=False, store_ids_4096=False, fuse_1024=True, launch_warps=1, parallel_chunk_n=1024, flat_grid=False, fuse_2048=False, precompute_bounds=False, row_fused=False, threshold_fast=threshold_target)
    if backend == 'tianshu':
        return _run_legacy_u64_sort_backend(scores, page_table, q_pos, kv_lens, out_blocks, out_pages, block_size, top_k, B, H, Q, NB, pack_bf16_scratch=False, store_ids_4096=True, fuse_1024=True, launch_warps=1, parallel_chunk_n=1024, flat_grid=False, fuse_2048=True, precompute_bounds=False, row_fused=False, threshold_fast=threshold_target)
    rows = B * H * Q
    log_k = top_k.bit_length() - 1
    use_single_kernel = NB <= 512 or (NB == 1024 and top_k == 32)
    if use_single_kernel:
        chunk_n = 512 if NB <= 512 else 1024
        _pair_single_gather_kernel[rows,](scores, page_table, q_pos, kv_lens, out_blocks, out_pages, H=H, Q=Q, NB=NB, TOP_K=top_k, BLOCK_SIZE=block_size, CHUNK_N=chunk_n, LOG_CHUNK=chunk_n.bit_length() - 1, LOG_K=log_k, num_warps=1, num_stages=1)
        return (out_blocks, out_pages)
    chunk_n = 1024
    num_chunks = triton.cdiv(NB, chunk_n)
    candidate_n = num_chunks * top_k
    scratch_ids = torch.empty((rows, num_chunks, top_k), device=scores.device, dtype=torch.int32)
    _pair_local_select_kernel[rows, num_chunks](scores, q_pos, kv_lens, scratch_ids, H=H, Q=Q, NB=NB, TOP_K=top_k, BLOCK_SIZE=block_size, CHUNK_N=chunk_n, NUM_CHUNKS=num_chunks, LOG_CHUNK=chunk_n.bit_length() - 1, LOG_K=log_k, num_warps=1, num_stages=1)
    _pair_merge_gather_kernel[rows,](scores, scratch_ids, page_table, out_blocks, out_pages, H=H, Q=Q, NB=NB, TOP_K=top_k, CANDIDATE_N=candidate_n, LOG_CANDIDATE=candidate_n.bit_length() - 1, LOG_K=log_k, num_warps=1, num_stages=1)
    return (out_blocks, out_pages)
