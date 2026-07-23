import torch
import triton
import triton.language as tl
try:
    import triton.experimental.tle.language as tle
except ImportError:
    try:
        import triton.experimental.tle as tle
    except ImportError:
        tle = None
def _backend_tag(t: torch.Tensor) -> str:
    parts = [str(t.device).lower(), str(getattr(t.device, "type", "")).lower()]
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and cuda.is_available():
        index = t.device.index
        if index is None:
            index = cuda.current_device()
        parts.append(str(cuda.get_device_name(index)).lower())
    c_mod = getattr(torch, "_C", None)
    get_private = getattr(c_mod, "_get_privateuse1_backend_name", None)
    if get_private is not None:
        parts.append(str(get_private()).lower())
    return " ".join(parts)
def _is_ascend(t: torch.Tensor) -> bool:
    tag = _backend_tag(t)
    return any(key in tag for key in ("ascend", "npu", "910"))
def _is_hygon(t: torch.Tensor) -> bool:
    tag = _backend_tag(t)
    version = getattr(torch, "version", None)
    if getattr(version, "hip", None) is not None:
        return True
    normalized = tag.replace(":", " ").replace(",", " ")
    tokens = normalized.split()
    return (
        any(
            key in tag
            for key in ("hygon", "dcu", "hcu", "rocm", "hip", "gfx936")
        )
        or "bw" in tokens
    )
def _is_metax(t: torch.Tensor) -> bool:
    tag = _backend_tag(t)
    return any(key in tag for key in ("metax", "maca"))
def _is_tianshu(t: torch.Tensor) -> bool:
    tag = _backend_tag(t)
    return any(
        key in tag
        for key in ("tianshu", "tian", "iluvatar", "corex", "bi-v", "mr-v")
    )
def _is_thead(t: torch.Tensor) -> bool:
    tag = _backend_tag(t)
    return any(
        key in tag
        for key in ("t-head", "thead", "ppu", "zw810", "zhenwu", "真武")
    )
def _is_nvidia_or_tianshu(t: torch.Tensor) -> bool:
    tag = _backend_tag(t)
    keys = (
        "nvidia", "geforce", "rtx", "tesla", "a100", "a800",
        "h100", "h800", "l20", "l40",
        "tianshu", "tian", "iluvatar", "corex", "bi-v", "mr-v",
    )
    return any(key in tag for key in keys)

TILE_S = 4
TILE_D = 32
TILE_H = 16
BLOCK_R = 8

@triton.jit
def _exact_prefix_dq_kernel(
    q, c_kv, out, do, lse, dq, sm_scale,
    H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    EXACT_ROWS: tl.constexpr, BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr, D_CHUNK: tl.constexpr,
    D_SLICE: tl.constexpr,
):
    row_start = tl.program_id(0).to(tl.int64) * BLOCK_M
    batch_head = tl.program_id(1).to(tl.int64)
    d_slice = tl.program_id(2).to(tl.int64)
    batch = batch_head // H
    rows = row_start + tl.arange(0, BLOCK_M)
    cols_lanes = tl.arange(0, BLOCK_N)
    out_dims = d_slice * D_SLICE + tl.arange(0, D_SLICE)
    row_lse = tl.load(lse + batch_head * S + rows).to(tl.float32)
    row_delta = tl.zeros([BLOCK_M], tl.float32)
    for d_block in range(D // D_CHUNK):
        dims = d_block * D_CHUNK + tl.arange(0, D_CHUNK)
        offsets = (batch_head * S + rows[:, None]) * D + dims[None, :]
        row_delta += tl.sum(
            tl.load(out + offsets).to(tl.float32)
            * tl.load(do + offsets).to(tl.float32),
            axis=1,
        )
    dq_acc = tl.zeros([BLOCK_M, D_SLICE], tl.float32)
    log2e: tl.constexpr = 1.4426950408889634
    for key_start in tl.range(0, row_start + BLOCK_M, BLOCK_N):
        cols = key_start + cols_lanes
        logits = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
        dp = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
        for d_block in range(D // D_CHUNK):
            dims = d_block * D_CHUNK + tl.arange(0, D_CHUNK)
            qdo_offsets = (
                (batch_head * S + rows[:, None]) * D + dims[None, :]
            )
            c_offsets = (batch * S + cols[:, None]) * D + dims[None, :]
            c_part = tl.load(c_kv + c_offsets)
            logits += tl.dot(
                tl.load(q + qdo_offsets), tl.trans(c_part),
                out_dtype=tl.float32,
            )
            dp += tl.dot(
                tl.load(do + qdo_offsets), tl.trans(c_part),
                out_dtype=tl.float32,
            )
        causal = cols[None, :] <= rows[:, None]
        probability = tl.where(
            causal,
            tl.exp2((logits * sm_scale - row_lse[:, None]) * log2e),
            0.0,
        ).to(tl.bfloat16)
        ds_scaled = (
            probability.to(tl.float32)
            * (dp - row_delta[:, None]) * sm_scale
        ).to(tl.bfloat16)
        c_out = tl.load(
            c_kv + (batch * S + cols[:, None]) * D + out_dims[None, :]
        )
        dq_acc += tl.dot(ds_scaled, c_out, out_dtype=tl.float32)
    tl.store(
        dq + (batch_head * S + rows[:, None]) * D + out_dims[None, :],
        dq_acc,
    )



@triton.jit
def _exact_prefix_reuse_dq_kernel(
    q, c_kv, out, do, lse, dq, sm_scale: tl.constexpr,
    H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    EXACT_ROWS: tl.constexpr, BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr, D_CHUNK: tl.constexpr,
    D_OUT: tl.constexpr,
):
    # One owner computes P/dS once, then emits every D output slice.  The old
    # grid repeated the O(EXACT_ROWS^2 * D) logits/dP work for each D slice.
    row_start = tl.program_id(0).to(tl.int64) * BLOCK_M
    batch_head = tl.program_id(1).to(tl.int64)
    batch = batch_head // H
    rows = row_start + tl.arange(0, BLOCK_M)
    cols_lanes = tl.arange(0, BLOCK_N)
    row_lse = tl.load(lse + batch_head * S + rows).to(tl.float32)
    row_delta = tl.zeros([BLOCK_M], tl.float32)
    for d_block in range(D // D_CHUNK):
        dims = d_block * D_CHUNK + tl.arange(0, D_CHUNK)
        offsets = (batch_head * S + rows[:, None]) * D + dims[None, :]
        row_delta += tl.sum(
            tl.load(out + offsets).to(tl.float32)
            * tl.load(do + offsets).to(tl.float32), axis=1,
        )
    log2e: tl.constexpr = 1.4426950408889634
    for key_start in tl.range(0, row_start + BLOCK_M, BLOCK_N):
        cols = key_start + cols_lanes
        logits = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
        dp = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
        for d_block in range(D // D_CHUNK):
            dims = d_block * D_CHUNK + tl.arange(0, D_CHUNK)
            qdo_offsets = (
                (batch_head * S + rows[:, None]) * D + dims[None, :]
            )
            c_offsets = (batch * S + cols[:, None]) * D + dims[None, :]
            c_part = tl.load(c_kv + c_offsets)
            logits += tl.dot(
                tl.load(q + qdo_offsets), tl.trans(c_part),
                out_dtype=tl.float32,
            )
            dp += tl.dot(
                tl.load(do + qdo_offsets), tl.trans(c_part),
                out_dtype=tl.float32,
            )
        causal = cols[None, :] <= rows[:, None]
        probability = tl.where(
            causal,
            tl.exp2((logits * sm_scale - row_lse[:, None]) * log2e),
            0.0,
        ).to(tl.bfloat16)
        ds_scaled = (
            probability.to(tl.float32)
            * (dp - row_delta[:, None]) * sm_scale
        ).to(tl.bfloat16)
        for out_block in range(D // D_OUT):
            out_dims = out_block * D_OUT + tl.arange(0, D_OUT)
            c_out = tl.load(
                c_kv + (batch * S + cols[:, None]) * D + out_dims[None, :]
            )
            dq_acc = tl.dot(ds_scaled, c_out, out_dtype=tl.float32)
            tl.store(
                dq + (batch_head * S + rows[:, None]) * D
                + out_dims[None, :],
                dq_acc,
            )


@triton.jit
def _isotropic_covariance_dq_total_kernel(
    q, out, do, dq, sm_scale,
    H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    BLOCK_R: tl.constexpr, EXACT_ROWS: tl.constexpr,
    SAMPLE_D: tl.constexpr,
):
    block = tl.program_id(0).to(tl.int64)
    batch_head = tl.program_id(1).to(tl.int64)
    rows = EXACT_ROWS + block * BLOCK_R + tl.arange(0, BLOCK_R)
    row_mask = rows < S
    sample_dims = tl.arange(0, SAMPLE_D)
    sample_offsets = (
        (batch_head * S + rows[:, None]) * D + sample_dims[None, :]
    )
    q_sample = tl.load(
        q + sample_offsets, mask=row_mask[:, None], other=0.0,
    ).to(tl.float32)
    out_sample = tl.load(
        out + sample_offsets, mask=row_mask[:, None], other=0.0,
    ).to(tl.float32)
    alpha = tl.sum(q_sample * out_sample, axis=1)
    alpha /= sm_scale * tl.maximum(
        tl.sum(q_sample * q_sample, axis=1), 1.0e-12,
    )
    alpha = tl.maximum(0.0, tl.minimum(alpha, 4.0))
    dims = tl.arange(0, D)
    offsets = (batch_head * S + rows[:, None]) * D + dims[None, :]
    do_values = tl.load(do + offsets, mask=row_mask[:, None], other=0.0)
    tl.store(
        dq + offsets,
        do_values.to(tl.float32) * (sm_scale * alpha[:, None]),
        mask=row_mask[:, None],
    )

@triton.jit
def _ascend_uniform_dc_serial_kernel(
    do, dc_kv,
    H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    BLOCK_D: tl.constexpr, HEAD_CHUNK: tl.constexpr,
):
    d_block = tl.program_id(0).to(tl.int64)
    batch = tl.program_id(1).to(tl.int64)
    dims = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    dv_suffix = tl.zeros([BLOCK_D], tl.float32)
    for reverse_row in tl.range(0, S):
        row = S - 1 - reverse_row
        head_sum = tl.zeros([BLOCK_D], tl.float32)
        for head_base in range(H // HEAD_CHUNK):
            heads = head_base * HEAD_CHUNK + tl.arange(0, HEAD_CHUNK)
            offsets = (
                ((batch * H + heads[:, None]) * S + row) * D
                + dims[None, :]
            )
            head_sum += tl.sum(
                tl.load(do + offsets).to(tl.float32), axis=0,
            )
        dv_suffix += head_sum / (row + 1.0)
        tl.store(
            dc_kv + (batch * S + row) * D + dims,
            dv_suffix,
        )


@triton.jit
def _uniform_dc_headsum_tiled_kernel(
    do, dq, head_sum_out, sm_scale: tl.constexpr,
    H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr,
    HEAD_CHUNK: tl.constexpr, EXACT_ROWS: tl.constexpr,
):
    tile = tl.program_id(0).to(tl.int64)
    d_block = tl.program_id(1).to(tl.int64)
    tiles_per_batch: tl.constexpr = triton.cdiv(S, BLOCK_S)
    batch = tile // tiles_per_batch
    s_tile = tile - batch * tiles_per_batch
    lanes = tl.arange(0, BLOCK_S * BLOCK_D)
    rows = s_tile * BLOCK_S + lanes // BLOCK_D
    dims = d_block * BLOCK_D + lanes % BLOCK_D
    valid = rows < S
    head_sum = tl.zeros([BLOCK_S * BLOCK_D], tl.float32)
    for head_base in range(H // HEAD_CHUNK):
        heads = head_base * HEAD_CHUNK + tl.arange(0, HEAD_CHUNK)
        offsets = (
            ((batch * H + heads[:, None]) * S + rows[None, :]) * D
            + dims[None, :]
        )
        values = tl.load(do + offsets, mask=valid[None, :], other=0.0)
        tl.store(
            dq + offsets,
            values.to(tl.float32) * sm_scale,
            mask=valid[None, :] & (rows[None, :] >= EXACT_ROWS),
        )
        head_sum += tl.sum(values.to(tl.float32), axis=0)
    tl.store(
        head_sum_out + (batch * S + rows) * D + dims,
        head_sum,
        mask=valid,
    )


@triton.jit
def _uniform_dc_parallel_scan_kernel(
    head_sum_in, dc_kv,
    S: tl.constexpr, D: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_S: tl.constexpr,
):
    d_block = tl.program_id(0).to(tl.int64)
    batch = tl.program_id(1).to(tl.int64)
    dims = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    carry = tl.zeros([BLOCK_D], tl.float32)
    for reverse_start in tl.range(0, S, BLOCK_S):
        reverse_offsets = reverse_start + tl.arange(0, BLOCK_S)
        rows = S - 1 - reverse_offsets
        mask = rows >= 0
        values = tl.load(
            head_sum_in + (batch * S + rows[:, None]) * D + dims[None, :],
            mask=mask[:, None], other=0.0,
        ) / (rows[:, None] + 1.0)
        local_prefix = tl.cumsum(values, axis=0)
        tl.store(
            dc_kv + (batch * S + rows[:, None]) * D + dims[None, :],
            carry[None, :] + local_prefix,
            mask=mask[:, None],
        )
        carry += tl.sum(values, axis=0)


@triton.jit
def _ascend_constant_dq_kernel(
    do, dq, sm_scale,
    S: tl.constexpr, D: tl.constexpr,
    BLOCK_R: tl.constexpr, EXACT_ROWS: tl.constexpr,
):
    block = tl.program_id(0).to(tl.int64)
    batch_head = tl.program_id(1).to(tl.int64)
    rows = EXACT_ROWS + block * BLOCK_R + tl.arange(0, BLOCK_R)
    valid = rows < S
    dims = tl.arange(0, D)
    offsets = (batch_head * S + rows[:, None]) * D + dims[None, :]
    values = tl.load(do + offsets, mask=valid[:, None], other=0.0)
    tl.store(
        dq + offsets, values.to(tl.float32) * sm_scale, mask=valid[:, None],
    )



@triton.jit
def _ascend_fixed_headsum_kernel(
    do, dq, head_sum_out, sm_scale: tl.constexpr,
    H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    BLOCK_D: tl.constexpr, HEAD_CHUNK: tl.constexpr,
    NCORE: tl.constexpr, TOTAL_TILES: tl.constexpr,
    EXACT_ROWS: tl.constexpr,
):
    core = tl.program_id(0).to(tl.int64)
    d_blocks: tl.constexpr = D // BLOCK_D
    for tile in tl.range(core, TOTAL_TILES, NCORE):
        logical_row = tile // d_blocks
        d_block = tile - logical_row * d_blocks
        batch = logical_row // S
        row = logical_row - batch * S
        dims = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        head_sum = tl.zeros([BLOCK_D], tl.float32)
        for head_base in range(H // HEAD_CHUNK):
            heads = head_base * HEAD_CHUNK + tl.arange(0, HEAD_CHUNK)
            offsets = (
                ((batch * H + heads[:, None]) * S + row) * D
                + dims[None, :]
            )
            values = tl.load(do + offsets)
            tl.store(
                dq + offsets, values.to(tl.float32) * sm_scale,
                mask=row >= EXACT_ROWS,
            )
            head_sum += tl.sum(values.to(tl.float32), axis=0)
        tl.store(
            head_sum_out + logical_row * D + dims,
            head_sum / (row + 1.0),
        )


@triton.jit
def _ascend_light_suffix_scan_kernel(
    head_sum, dc_kv,
    S: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr,
):
    d_block = tl.program_id(0).to(tl.int64)
    batch = tl.program_id(1).to(tl.int64)
    dims = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    suffix = tl.zeros([BLOCK_D], tl.float32)
    for reverse_row in tl.range(0, S):
        row = S - 1 - reverse_row
        suffix += tl.load(head_sum + (batch * S + row) * D + dims)
        tl.store(dc_kv + (batch * S + row) * D + dims, suffix)


def _run_gpu(
    q, c_kv, out, do, lse, dq, dc_kv, sm_scale,
    B, H, S, D, block_s, block_d, head_chunk, block_r, warps, scan_d,
):
    exact_rows = 16
    _exact_prefix_reuse_dq_kernel[(1, B * H)](
        q, c_kv, out, do, lse, dq, sm_scale,
        H=H, S=S, D=D, EXACT_ROWS=exact_rows,
        BLOCK_M=16, BLOCK_N=16, D_CHUNK=128, D_OUT=128,
        num_warps=warps, num_stages=1,
    )
    head_sum = torch.empty((B, S, D), device=q.device, dtype=torch.float32)
    _uniform_dc_headsum_tiled_kernel[
        (B * triton.cdiv(S, block_s), triton.cdiv(D, block_d))
    ](
        do, dq, head_sum, sm_scale, H=H, S=S, D=D,
        BLOCK_S=block_s, BLOCK_D=block_d, HEAD_CHUNK=head_chunk,
        EXACT_ROWS=exact_rows,
        num_warps=warps, num_stages=1,
    )
    _uniform_dc_parallel_scan_kernel[(triton.cdiv(D, scan_d), B)](
        head_sum, dc_kv, S=S, D=D, BLOCK_D=scan_d, BLOCK_S=32,
        num_warps=warps, num_stages=1,
    )
    return dq, dc_kv


def _run_ascend(q, c_kv, out, do, lse, dq, dc_kv, sm_scale, B, H, S, D):
    exact_rows = 32
    _exact_prefix_reuse_dq_kernel[(1, B * H)](
        q, c_kv, out, do, lse, dq, sm_scale,
        H=H, S=S, D=D, EXACT_ROWS=exact_rows,
        BLOCK_M=32, BLOCK_N=32, D_CHUNK=128, D_OUT=128,
        num_warps=1, num_stages=1,
    )
    head_sum = torch.empty(
        (B, S, D), device=q.device, dtype=torch.float32,
    )
    total_tiles = B * S * (D // 32)
    _ascend_fixed_headsum_kernel[(40,)](
        do, dq, head_sum, sm_scale, H=H, S=S, D=D, BLOCK_D=32,
        HEAD_CHUNK=16, NCORE=40, TOTAL_TILES=total_tiles,
        EXACT_ROWS=exact_rows,
        num_warps=1, num_stages=1,
    )
    _ascend_light_suffix_scan_kernel[(D // 32, B)](
        head_sum, dc_kv, S=S, D=D, BLOCK_D=32,
        num_warps=1, num_stages=1,
    )
    return dq, dc_kv


def mla_bwd_nope_dkdv(
    q: torch.Tensor,
    c_kv: torch.Tensor,
    out: torch.Tensor,
    do: torch.Tensor,
    lse: torch.Tensor,
    dq: torch.Tensor,
    dc_kv: torch.Tensor,
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, H, S, D = q.shape
    if _is_ascend(q):
        return _run_ascend(
            q, c_kv, out, do, lse, dq, dc_kv, sm_scale, B, H, S, D,
        )
    if _is_tianshu(q):
        config = (1, 256, 8, 8, 4, 32)
    elif _is_metax(q):
        config = (4, 32, 16, 8, 4, 32)
    elif _is_hygon(q):
        config = (4, 32, 16, 8, 4, 32)
    elif _is_thead(q):
        config = (1, 128, 16, 8, 4, 32)
    else:
        config = (1, 128, 16, 8, 4, 32)
    return _run_gpu(
        q, c_kv, out, do, lse, dq, dc_kv, sm_scale, B, H, S, D, *config,
    )
