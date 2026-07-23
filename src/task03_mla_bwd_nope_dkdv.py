import os
os.environ.setdefault("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS", "1")
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


@triton.jit
def _tle_bwd_qdo_producer(
    qdo_writer, q, do, lse, delta, batch, group,
    H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    HEAD_GROUP: tl.constexpr, BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr, DPH: tl.constexpr,
):
    key_start = tl.program_id(0) * BLOCK_N
    q_tiles = tl.cdiv(S - key_start, BLOCK_M)
    rows_lanes = tl.arange(0, BLOCK_M)
    dims = tl.arange(0, DPH)
    for head_offset in range(HEAD_GROUP):
        head = group * HEAD_GROUP + head_offset
        batch_head = batch * H + head
        for q_tile in tl.range(q_tiles):
            iteration = head_offset * q_tiles + q_tile
            rows = key_start + q_tile * BLOCK_M + rows_lanes
            row_mask = rows < S
            slot = qdo_writer.acquire(iteration)
            base = (batch_head * S + rows[:, None]) * D
            left_offsets = base + dims[None, :]
            right_offsets = base + DPH + dims[None, :]
            q_l = tl.load(q + left_offsets, mask=row_mask[:, None], other=0.0)
            q_r = tl.load(q + right_offsets, mask=row_mask[:, None], other=0.0)
            do_l = tl.load(do + left_offsets, mask=row_mask[:, None], other=0.0)
            do_r = tl.load(do + right_offsets, mask=row_mask[:, None], other=0.0)
            row_lse = tl.load(
                lse + batch_head * S + rows, mask=row_mask, other=0.0,
            ).to(tl.float32)
            row_delta = tl.load(
                delta + batch_head * S + rows, mask=row_mask, other=0.0,
            ).to(tl.float32)
            tl.store(tle.gpu.local_ptr(slot.q_l), q_l, mask=row_mask[:, None])
            tl.store(tle.gpu.local_ptr(slot.q_r), q_r, mask=row_mask[:, None])
            tl.store(tle.gpu.local_ptr(slot.do_l), do_l, mask=row_mask[:, None])
            tl.store(tle.gpu.local_ptr(slot.do_r), do_r, mask=row_mask[:, None])
            tl.store(tle.gpu.local_ptr(slot.lse), row_lse, mask=row_mask)
            tl.store(tle.gpu.local_ptr(slot.delta), row_delta, mask=row_mask)
            qdo_writer.commit(iteration)

@triton.jit
def _tle_bwd_dc_left_consumer(
    qdo_reader, score_writer, c_l_smem, c_r_smem, partial_dc,
    batch, group, sm_scale,
    H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    NUM_GROUPS: tl.constexpr, HEAD_GROUP: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, DPH: tl.constexpr,
):
    key_start = tl.program_id(0) * BLOCK_N
    q_tiles = tl.cdiv(S - key_start, BLOCK_M)
    row_lanes = tl.arange(0, BLOCK_M)
    col_lanes = tl.arange(0, BLOCK_N)
    dims = tl.arange(0, DPH)
    cols = key_start + col_lanes
    col_mask = cols < S
    c_l = tl.load(tle.gpu.local_ptr(c_l_smem))
    c_r = tl.load(tle.gpu.local_ptr(c_r_smem))
    dc_l = tl.zeros([BLOCK_N, DPH], dtype=tl.float32)
    log2e: tl.constexpr = 1.4426950408889634
    for head_offset in range(HEAD_GROUP):
        for q_tile in tl.range(q_tiles):
            iteration = head_offset * q_tiles + q_tile
            rows = key_start + q_tile * BLOCK_M + row_lanes
            row_mask = rows < S
            wait_result = qdo_reader.wait(iteration)
            slot = wait_result.slot
            q_l = tl.load(tle.gpu.local_ptr(slot.q_l))
            q_r = tl.load(tle.gpu.local_ptr(slot.q_r))
            do_l = tl.load(tle.gpu.local_ptr(slot.do_l))
            do_r = tl.load(tle.gpu.local_ptr(slot.do_r))
            row_lse = tl.load(tle.gpu.local_ptr(slot.lse)).to(tl.float32)
            row_delta = tl.load(tle.gpu.local_ptr(slot.delta)).to(tl.float32)
            logits = tl.dot(q_l, tl.trans(c_l), out_dtype=tl.float32)
            logits += tl.dot(q_r, tl.trans(c_r), out_dtype=tl.float32)
            dp = tl.dot(do_l, tl.trans(c_l), out_dtype=tl.float32)
            dp += tl.dot(do_r, tl.trans(c_r), out_dtype=tl.float32)
            valid = row_mask[:, None] & col_mask[None, :] & (
                cols[None, :] <= rows[:, None]
            )
            probability = tl.where(
                valid,
                tl.exp2((logits * sm_scale - row_lse[:, None]) * log2e),
                0.0,
            )
            ds = probability * (dp - row_delta[:, None]) * sm_scale
            p_mma = probability.to(tl.bfloat16)
            ds_mma = ds.to(tl.bfloat16)
            dc_l += tl.dot(tl.trans(ds_mma), q_l, out_dtype=tl.float32)
            dc_l += tl.dot(tl.trans(p_mma), do_l, out_dtype=tl.float32)
            score_slot = score_writer.acquire(iteration)
            tl.store(tle.gpu.local_ptr(score_slot.prob), p_mma)
            tl.store(tle.gpu.local_ptr(score_slot.ds), ds_mma)
            score_writer.commit(iteration)
            qdo_reader.release(iteration)
    out_dims = dims
    offsets = (
        ((batch * NUM_GROUPS + group) * S + cols[:, None]) * D
        + out_dims[None, :]
    )
    tl.store(
        partial_dc + offsets, dc_l,
        mask=col_mask[:, None],
    )

@triton.jit
def _tle_bwd_dc_right_consumer(
    qdo_reader, score_reader, partial_dc, batch, group,
    S: tl.constexpr, D: tl.constexpr, NUM_GROUPS: tl.constexpr,
    HEAD_GROUP: tl.constexpr, BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr, DPH: tl.constexpr,
):
    key_start = tl.program_id(0) * BLOCK_N
    q_tiles = tl.cdiv(S - key_start, BLOCK_M)
    col_lanes = tl.arange(0, BLOCK_N)
    dims = tl.arange(0, DPH)
    cols = key_start + col_lanes
    col_mask = cols < S
    dc_r = tl.zeros([BLOCK_N, DPH], dtype=tl.float32)
    for head_offset in range(HEAD_GROUP):
        for q_tile in tl.range(q_tiles):
            iteration = head_offset * q_tiles + q_tile
            q_wait = qdo_reader.wait(iteration)
            q_slot = q_wait.slot
            score_wait = score_reader.wait(iteration)
            score_slot = score_wait.slot
            q_r = tl.load(tle.gpu.local_ptr(q_slot.q_r))
            do_r = tl.load(tle.gpu.local_ptr(q_slot.do_r))
            probability = tl.load(tle.gpu.local_ptr(score_slot.prob))
            ds = tl.load(tle.gpu.local_ptr(score_slot.ds))
            dc_r += tl.dot(tl.trans(ds), q_r, out_dtype=tl.float32)
            dc_r += tl.dot(tl.trans(probability), do_r, out_dtype=tl.float32)
            score_reader.release(iteration)
            qdo_reader.release(iteration)
    out_dims = DPH + dims
    offsets = (
        ((batch * NUM_GROUPS + group) * S + cols[:, None]) * D
        + out_dims[None, :]
    )
    tl.store(
        partial_dc + offsets, dc_r,
        mask=col_mask[:, None],
    )

@triton.jit
def _tle_pipe_bwd_dc_kernel(
    q, c_kv, do, lse, delta, partial_dc, sm_scale,
    H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    NUM_GROUPS: tl.constexpr, HEAD_GROUP: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    PIPE_CAPACITY: tl.constexpr,
):
    DPH: tl.constexpr = D // 2
    key_block = tl.program_id(0)
    batch_group = tl.program_id(1)
    batch = batch_group // NUM_GROUPS
    group = batch_group - batch * NUM_GROUPS
    cols = key_block * BLOCK_N + tl.arange(0, BLOCK_N)
    dims = tl.arange(0, DPH)
    col_mask = cols < S
    c_base = (batch * S + cols[:, None]) * D
    c_l_smem = tle.gpu.alloc(
        [BLOCK_N, DPH], dtype=c_kv.dtype.element_ty,
        layout=None, scope=tle.gpu.smem,
    )
    c_r_smem = tle.gpu.alloc(
        [BLOCK_N, DPH], dtype=c_kv.dtype.element_ty,
        layout=None, scope=tle.gpu.smem,
    )
    c_l = tl.load(c_kv + c_base + dims[None, :], mask=col_mask[:, None], other=0.0)
    c_r = tl.load(
        c_kv + c_base + DPH + dims[None, :],
        mask=col_mask[:, None], other=0.0,
    )
    tl.store(tle.gpu.local_ptr(c_l_smem), c_l, mask=col_mask[:, None])
    tl.store(tle.gpu.local_ptr(c_r_smem), c_r, mask=col_mask[:, None])
    q_l_smem = tle.gpu.alloc(
        [PIPE_CAPACITY, BLOCK_M, DPH], dtype=q.dtype.element_ty,
        layout=None, scope=tle.gpu.smem,
    )
    q_r_smem = tle.gpu.alloc(
        [PIPE_CAPACITY, BLOCK_M, DPH], dtype=q.dtype.element_ty,
        layout=None, scope=tle.gpu.smem,
    )
    do_l_smem = tle.gpu.alloc(
        [PIPE_CAPACITY, BLOCK_M, DPH], dtype=q.dtype.element_ty,
        layout=None, scope=tle.gpu.smem,
    )
    do_r_smem = tle.gpu.alloc(
        [PIPE_CAPACITY, BLOCK_M, DPH], dtype=q.dtype.element_ty,
        layout=None, scope=tle.gpu.smem,
    )
    lse_smem = tle.gpu.alloc(
        [PIPE_CAPACITY, BLOCK_M], dtype=tl.float32,
        layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False,
    )
    delta_smem = tle.gpu.alloc(
        [PIPE_CAPACITY, BLOCK_M], dtype=tl.float32,
        layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False,
    )
    qdo_pipe = tle.pipe(
        capacity=PIPE_CAPACITY, scope="cta", name="mla_bwd_qdo",
        readers=("left", "right"), q_l=q_l_smem, q_r=q_r_smem,
        do_l=do_l_smem, do_r=do_r_smem, lse=lse_smem, delta=delta_smem,
    )
    prob_smem = tle.gpu.alloc(
        [PIPE_CAPACITY, BLOCK_M, BLOCK_N], dtype=q.dtype.element_ty,
        layout=None, scope=tle.gpu.smem,
    )
    ds_smem = tle.gpu.alloc(
        [PIPE_CAPACITY, BLOCK_M, BLOCK_N], dtype=q.dtype.element_ty,
        layout=None, scope=tle.gpu.smem,
    )
    score_pipe = tle.pipe(
        capacity=PIPE_CAPACITY, scope="cta", name="mla_bwd_score",
        prob=prob_smem, ds=ds_smem,
    )
    qdo_writer = qdo_pipe.writer()
    qdo_left_reader = qdo_pipe.reader("left")
    qdo_right_reader = qdo_pipe.reader("right", fields=("q_r", "do_r"))
    score_writer = score_pipe.writer()
    score_reader = score_pipe.reader()
    tle.gpu.warp_specialize(
        [
            (_tle_bwd_qdo_producer, (
                qdo_writer, q, do, lse, delta, batch, group,
                H, S, D, HEAD_GROUP, BLOCK_M, BLOCK_N, DPH,
            )),
            (_tle_bwd_dc_left_consumer, (
                qdo_left_reader, score_writer, c_l_smem, c_r_smem,
                partial_dc, batch, group, sm_scale, H, S, D,
                NUM_GROUPS, HEAD_GROUP, BLOCK_M, BLOCK_N, DPH,
            )),
            (_tle_bwd_dc_right_consumer, (
                qdo_right_reader, score_reader, partial_dc, batch, group,
                S, D, NUM_GROUPS, HEAD_GROUP, BLOCK_M, BLOCK_N, DPH,
            )),
        ],
        [4, 4],
        [240, 168],
    )


@triton.jit
def _ascend_fixed_delta_kernel(
    out,
    do,
    delta,
    S: tl.constexpr,
    D: tl.constexpr,
    TOTAL_ROWS: tl.constexpr,
    NCORE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    dims = tl.arange(0, D)
    for base in tl.range(0, TOTAL_ROWS, NCORE):
        logical = base + pid
        if logical < TOTAL_ROWS:
            offsets = logical * D + dims
            out_values = tl.load(out + offsets).to(tl.float32)
            do_values = tl.load(do + offsets).to(tl.float32)
            tl.store(delta + logical, tl.sum(out_values * do_values, axis=0))
@triton.jit
def _ascend_fixed_dq_worker_kernel(
    q,
    c_kv,
    do,
    lse,
    delta,
    dq,
    sm_scale,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    Q_BLOCKS: tl.constexpr,
    D_SLICES: tl.constexpr,
    TOTAL_TASKS: tl.constexpr,
    NCORE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D_CHUNK: tl.constexpr,
    D_SLICE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    for base in tl.range(0, TOTAL_TASKS, NCORE):
        logical = base + pid
        if logical < TOTAL_TASKS:
            d_slice = logical % D_SLICES
            owner = logical // D_SLICES
            batch_query = owner // H
            head = owner % H
            q_block = batch_query % Q_BLOCKS
            batch = batch_query // Q_BLOCKS
            batch_head = batch * H + head
            rows = q_block * BLOCK_M + tl.arange(0, BLOCK_M)
            out_dims = d_slice * D_SLICE + tl.arange(0, D_SLICE)
            row_mask = rows < S
            out_dim_mask = out_dims < D
            row_lse = tl.load(
                lse + batch_head * S + rows,
                mask=row_mask,
                other=0.0,
            ).to(tl.float32)
            row_delta = tl.load(
                delta + batch_head * S + rows,
                mask=row_mask,
                other=0.0,
            ).to(tl.float32)
            dq_acc = tl.zeros([BLOCK_M, D_SLICE], dtype=tl.float32)
            end_n = tl.minimum((q_block + 1) * BLOCK_M, S)
            for start_n in tl.range(0, end_n, BLOCK_N):
                cols = start_n + tl.arange(0, BLOCK_N)
                col_mask = cols < S
                logits = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
                dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
                for d_block in range(D // D_CHUNK):
                    dims = d_block * D_CHUNK + tl.arange(0, D_CHUNK)
                    qdo_offsets = (
                        (batch_head * S + rows[:, None]) * D
                        + dims[None, :]
                    )
                    kv_offsets = (
                        (batch * S + cols[:, None]) * D
                        + dims[None, :]
                    )
                    q_part = tl.load(
                        q + qdo_offsets,
                        mask=row_mask[:, None],
                        other=0.0,
                    )
                    do_part = tl.load(
                        do + qdo_offsets,
                        mask=row_mask[:, None],
                        other=0.0,
                    )
                    c_part = tl.load(
                        c_kv + kv_offsets,
                        mask=col_mask[:, None],
                        other=0.0,
                    )
                    logits += tl.dot(
                        q_part,
                        tl.trans(c_part),
                        out_dtype=tl.float32,
                    )
                    dp += tl.dot(
                        do_part,
                        tl.trans(c_part),
                        out_dtype=tl.float32,
                    )
                causal = (
                    row_mask[:, None]
                    & col_mask[None, :]
                    & (cols[None, :] <= rows[:, None])
                )
                probability = tl.where(
                    causal,
                    tl.exp(logits * sm_scale - row_lse[:, None]),
                    0.0,
                )
                ds_scaled = (
                    probability * (dp - row_delta[:, None]) * sm_scale
                )
                c_out_offsets = (
                    (batch * S + cols[:, None]) * D
                    + out_dims[None, :]
                )
                c_out = tl.load(
                    c_kv + c_out_offsets,
                    mask=col_mask[:, None] & out_dim_mask[None, :],
                    other=0.0,
                )
                dq_acc += tl.dot(
                    ds_scaled.to(tl.bfloat16),
                    c_out,
                    out_dtype=tl.float32,
                )
            dq_offsets = (
                (batch_head * S + rows[:, None]) * D
                + out_dims[None, :]
            )
            tl.store(
                dq + dq_offsets,
                dq_acc,
                mask=row_mask[:, None] & out_dim_mask[None, :],
            )
@triton.jit
def _ascend_fixed_dc_worker_kernel(
    q,
    c_kv,
    do,
    lse,
    delta,
    partial_dc,
    sm_scale,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    HEAD_GROUP: tl.constexpr,
    KEY_BLOCKS: tl.constexpr,
    D_SLICES: tl.constexpr,
    TOTAL_TASKS: tl.constexpr,
    NCORE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D_CHUNK: tl.constexpr,
    D_SLICE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    for base in tl.range(0, TOTAL_TASKS, NCORE):
        logical = base + pid
        if logical < TOTAL_TASKS:
            d_slice = logical % D_SLICES
            owner = logical // D_SLICES
            batch_key = owner // NUM_GROUPS
            group = owner % NUM_GROUPS
            key_block = batch_key % KEY_BLOCKS
            batch = batch_key // KEY_BLOCKS
            batch_group = batch * NUM_GROUPS + group
            cols = key_block * BLOCK_N + tl.arange(0, BLOCK_N)
            out_dims = d_slice * D_SLICE + tl.arange(0, D_SLICE)
            col_mask = cols < S
            out_dim_mask = out_dims < D
            dc_acc = tl.zeros([BLOCK_N, D_SLICE], dtype=tl.float32)
            for head_offset in range(HEAD_GROUP):
                head = group * HEAD_GROUP + head_offset
                batch_head = batch * H + head
                for start_m in tl.range(
                    key_block * BLOCK_N,
                    S,
                    BLOCK_M,
                ):
                    rows = start_m + tl.arange(0, BLOCK_M)
                    row_mask = rows < S
                    row_lse = tl.load(
                        lse + batch_head * S + rows,
                        mask=row_mask,
                        other=0.0,
                    ).to(tl.float32)
                    row_delta = tl.load(
                        delta + batch_head * S + rows,
                        mask=row_mask,
                        other=0.0,
                    ).to(tl.float32)
                    logits = tl.zeros(
                        [BLOCK_M, BLOCK_N],
                        dtype=tl.float32,
                    )
                    dp = tl.zeros(
                        [BLOCK_M, BLOCK_N],
                        dtype=tl.float32,
                    )
                    for d_block in range(D // D_CHUNK):
                        dims = (
                            d_block * D_CHUNK
                            + tl.arange(0, D_CHUNK)
                        )
                        qdo_offsets = (
                            (batch_head * S + rows[:, None]) * D
                            + dims[None, :]
                        )
                        kv_offsets = (
                            (batch * S + cols[:, None]) * D
                            + dims[None, :]
                        )
                        q_part = tl.load(
                            q + qdo_offsets,
                            mask=row_mask[:, None],
                            other=0.0,
                        )
                        do_part = tl.load(
                            do + qdo_offsets,
                            mask=row_mask[:, None],
                            other=0.0,
                        )
                        c_part = tl.load(
                            c_kv + kv_offsets,
                            mask=col_mask[:, None],
                            other=0.0,
                        )
                        logits += tl.dot(
                            q_part,
                            tl.trans(c_part),
                            out_dtype=tl.float32,
                        )
                        dp += tl.dot(
                            do_part,
                            tl.trans(c_part),
                            out_dtype=tl.float32,
                        )
                    causal = (
                        row_mask[:, None]
                        & col_mask[None, :]
                        & (cols[None, :] <= rows[:, None])
                    )
                    probability = tl.where(
                        causal,
                        tl.exp(
                            logits * sm_scale - row_lse[:, None]
                        ),
                        0.0,
                    )
                    ds_scaled = (
                        probability
                        * (dp - row_delta[:, None])
                        * sm_scale
                    )
                    qdo_out_offsets = (
                        (batch_head * S + rows[:, None]) * D
                        + out_dims[None, :]
                    )
                    output_mask = (
                        row_mask[:, None] & out_dim_mask[None, :]
                    )
                    q_out = tl.load(
                        q + qdo_out_offsets,
                        mask=output_mask,
                        other=0.0,
                    )
                    do_out = tl.load(
                        do + qdo_out_offsets,
                        mask=output_mask,
                        other=0.0,
                    )
                    dc_acc += tl.dot(
                        tl.trans(ds_scaled.to(tl.bfloat16)),
                        q_out,
                        out_dtype=tl.float32,
                    )
                    dc_acc += tl.dot(
                        tl.trans(probability.to(tl.bfloat16)),
                        do_out,
                        out_dtype=tl.float32,
                    )
            partial_offsets = (
                (batch_group * S + cols[:, None]) * D
                + out_dims[None, :]
            )
            tl.store(
                partial_dc + partial_offsets,
                dc_acc,
                mask=col_mask[:, None] & out_dim_mask[None, :],
            )
@triton.jit
def _materialize_pds_triangle_kernel(
    q,
    c_kv,
    do,
    lse,
    delta,
    pds,
    sm_scale,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    TRI_TILES: tl.constexpr,
    PDS_PLANE: tl.constexpr,
    HEAD_TILE: tl.constexpr,
    BLOCK: tl.constexpr,
    D_CHUNK: tl.constexpr,
    SEPARATE_PDS: tl.constexpr = False,
    ASCEND_DSA: tl.constexpr = False,
    L2_HEAD_CLUSTER: tl.constexpr = False,
    DS_ONLY: tl.constexpr = False,
):
    """Produce one causal P/dS tile without any D-output accumulator."""
    head_groups: tl.constexpr = H // HEAD_TILE
    tiles: tl.constexpr = (S + BLOCK - 1) // BLOCK
    if L2_HEAD_CLUSTER:
        owner = tl.program_id(0).to(tl.int64)
        batch_group = owner % head_groups
        triangle_owner = owner // head_groups
        triangle_id = triangle_owner % TRI_TILES
        batch = triangle_owner // TRI_TILES
        query_tile = (
            (tl.sqrt((8 * triangle_id + 1).to(tl.float32)) - 1.0) * 0.5
        ).to(tl.int64)
        key_tile = triangle_id - query_tile * (query_tile + 1) // 2
        batch_group += batch * head_groups
    else:
        query_tile = tl.program_id(0).to(tl.int64)
        key_tile = tl.program_id(1).to(tl.int64)
        batch_group = tl.program_id(2).to(tl.int64)
    if key_tile <= query_tile:
        batch = batch_group // head_groups
        head_base = (batch_group % head_groups) * HEAD_TILE
        flat_m: tl.constexpr = HEAD_TILE * BLOCK
        lanes = tl.arange(0, flat_m)
        head_offsets = lanes // BLOCK
        row_lanes = lanes - head_offsets * BLOCK
        batch_heads = batch * H + head_base + head_offsets
        col_lanes = tl.arange(0, BLOCK)
        rows = query_tile * BLOCK + row_lanes
        cols = key_tile * BLOCK + col_lanes
        row_mask = rows < S
        col_mask = cols < S
        if SEPARATE_PDS:
            row_lse = tl.load(lse + batch_heads * S + rows).to(tl.float32)
            row_delta = tl.load(delta + batch_heads * S + rows).to(tl.float32)
        else:
            row_lse = tl.load(
                lse + batch_heads * S + rows, mask=row_mask, other=0.0,
            ).to(tl.float32)
            row_delta = tl.load(
                delta + batch_heads * S + rows, mask=row_mask, other=0.0,
            ).to(tl.float32)
        if ASCEND_DSA or DS_ONLY:
            logits = tl.zeros([flat_m, BLOCK], dtype=tl.float32)
            dp = tl.zeros([flat_m, BLOCK], dtype=tl.float32)
        else:
            joined_scores = tl.zeros([2 * flat_m, BLOCK], dtype=tl.float32)
        for d_block in range(D // D_CHUNK):
            dims = d_block * D_CHUNK + tl.arange(0, D_CHUNK)
            qdo_offsets = (
                (batch_heads[:, None] * S + rows[:, None]) * D
                + dims[None, :]
            )
            kv_offsets = (
                (batch * S + cols[:, None]) * D + dims[None, :]
            )
            if DS_ONLY or SEPARATE_PDS:
                q_part = tl.load(q + qdo_offsets)
                do_part = tl.load(do + qdo_offsets)
                c_part = tl.load(c_kv + kv_offsets)
            else:
                q_part = tl.load(
                    q + qdo_offsets, mask=row_mask[:, None], other=0.0,
                ).to(tl.bfloat16)
                do_part = tl.load(
                    do + qdo_offsets, mask=row_mask[:, None], other=0.0,
                ).to(tl.bfloat16)
                c_part = tl.load(
                    c_kv + kv_offsets, mask=col_mask[:, None], other=0.0,
                ).to(tl.bfloat16)
            if ASCEND_DSA or DS_ONLY:
                logits += tl.dot(
                    q_part, tl.trans(c_part), out_dtype=tl.float32
                )
                dp += tl.dot(
                    do_part, tl.trans(c_part), out_dtype=tl.float32
                )
            else:
                qdo = tl.reshape(
                    tl.join(q_part, do_part).permute(0, 2, 1),
                    [2 * flat_m, D_CHUNK],
                )
                joined_scores += tl.dot(
                    qdo, tl.trans(c_part), out_dtype=tl.float32
                )
        if not ASCEND_DSA and not DS_ONLY:
            score_pairs = tl.reshape(
                joined_scores, [flat_m, 2, BLOCK]
            ).permute(0, 2, 1)
            logits, dp = tl.split(score_pairs)
        if SEPARATE_PDS:
            valid = cols[None, :] <= rows[:, None]
        else:
            valid = (
                row_mask[:, None] & col_mask[None, :]
                & (cols[None, :] <= rows[:, None])
            )
        log2e: tl.constexpr = 1.4426950408889634
        probability = tl.where(
            valid,
            tl.exp2((logits * sm_scale - row_lse[:, None]) * log2e),
            0.0,
        )
        ds_scaled = probability * (dp - row_delta[:, None]) * sm_scale
        triangle_id = query_tile * (query_tile + 1) // 2 + key_tile
        elements = row_lanes[:, None] * BLOCK + col_lanes[None, :]
        scratch_base = (
            (batch_heads[:, None] * TRI_TILES + triangle_id)
            * BLOCK * BLOCK
        )
        scratch_offsets = scratch_base + elements
        tile_mask = row_mask[:, None] & col_mask[None, :]
        if DS_ONLY:
            tl.store(
                pds + scratch_offsets,
                ds_scaled.to(tl.float16),
                mask=tile_mask,
            )
        elif SEPARATE_PDS:
            tl.store(
                pds + scratch_offsets,
                probability.to(tl.float16),
            )
            tl.store(
                pds + PDS_PLANE + scratch_offsets,
                ds_scaled.to(tl.float16),
            )
        else:
            tl.store(
                pds + 2 * scratch_offsets,
                probability.to(tl.bfloat16),
                mask=tile_mask,
            )
            tl.store(
                pds + 2 * scratch_offsets + 1,
                ds_scaled.to(tl.bfloat16),
                mask=tile_mask,
            )
@triton.jit
def _dq_from_materialized_pds_kernel(
    c_kv,
    pds,
    dq,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    TRI_TILES: tl.constexpr,
    PDS_PLANE: tl.constexpr,
    BLOCK: tl.constexpr,
    D_SLICE: tl.constexpr,
    SEPARATE_PDS: tl.constexpr = False,
    ASCEND_DSA: tl.constexpr = False,
    L2_HEAD_CLUSTER: tl.constexpr = False,
    DS_ONLY: tl.constexpr = False,
):
    tiles: tl.constexpr = (S + BLOCK - 1) // BLOCK
    if L2_HEAD_CLUSTER:
        owner = tl.program_id(0).to(tl.int64)
        d_slice = tl.program_id(1).to(tl.int64)
        batch_query = owner // H
        head = owner % H
        query_tile = batch_query % tiles
        batch = batch_query // tiles
        batch_head = batch * H + head
    else:
        query_tile = tl.program_id(0).to(tl.int64)
        batch_head = tl.program_id(1).to(tl.int64)
        d_slice = tl.program_id(2).to(tl.int64)
        batch = batch_head // H
    row_lanes = tl.arange(0, BLOCK)
    col_lanes = tl.arange(0, BLOCK)
    rows = query_tile * BLOCK + row_lanes
    out_dims = d_slice * D_SLICE + tl.arange(0, D_SLICE)
    row_mask = rows < S
    out_dim_mask = out_dims < D
    dq_acc = tl.zeros([BLOCK, D_SLICE], dtype=tl.float32)
    for key_tile in tl.range(0, query_tile + 1):
        cols = key_tile * BLOCK + col_lanes
        col_mask = cols < S
        triangle_id = query_tile * (query_tile + 1) // 2 + key_tile
        elements = row_lanes[:, None] * BLOCK + col_lanes[None, :]
        scratch_base = (
            (batch_head * TRI_TILES + triangle_id) * BLOCK * BLOCK
        )
        if DS_ONLY:
            ds_scaled = tl.load(
                pds + scratch_base + elements,
                mask=row_mask[:, None] & col_mask[None, :], other=0.0,
            )
        elif SEPARATE_PDS:
            ds_scaled = tl.load(
                pds + PDS_PLANE + scratch_base + elements,
            )
        else:
            ds_scaled = tl.load(
                pds + 2 * (scratch_base + elements) + 1,
                mask=row_mask[:, None] & col_mask[None, :], other=0.0,
            )
        c_offsets = (
            (batch * S + cols[:, None]) * D + out_dims[None, :]
        )
        if SEPARATE_PDS:
            c_part = tl.load(c_kv + c_offsets).to(tl.float16)
        else:
            c_part = tl.load(
                c_kv + c_offsets,
                mask=col_mask[:, None] & out_dim_mask[None, :], other=0.0,
            ).to(tl.bfloat16)
        dq_acc += tl.dot(ds_scaled, c_part, out_dtype=tl.float32)
    dq_offsets = (
        (batch_head * S + rows[:, None]) * D + out_dims[None, :]
    )
    if SEPARATE_PDS:
        tl.store(dq + dq_offsets, dq_acc)
    else:
        tl.store(
            dq + dq_offsets, dq_acc,
            mask=row_mask[:, None] & out_dim_mask[None, :],
        )
@triton.jit
def _dc_from_materialized_pds_kernel(
    q,
    c_kv,
    do,
    lse,
    pds,
    partial_dc,
    sm_scale,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    HEAD_GROUP: tl.constexpr,
    HEAD_TILE: tl.constexpr,
    TILES: tl.constexpr,
    Q_BLOCKS: tl.constexpr,
    TRI_TILES: tl.constexpr,
    PDS_PLANE: tl.constexpr,
    BLOCK: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    D_CHUNK: tl.constexpr,
    D_SLICE: tl.constexpr,
    SEPARATE_PDS: tl.constexpr = False,
    ASCEND_DSA: tl.constexpr = False,
    DS_ONLY: tl.constexpr = False,
):
    key_tile = tl.program_id(0).to(tl.int64)
    batch_group = tl.program_id(1).to(tl.int64)
    d_slice = tl.program_id(2).to(tl.int64)
    group = batch_group % NUM_GROUPS
    batch = batch_group // NUM_GROUPS
    head_base = group * HEAD_GROUP
    col_lanes = tl.arange(0, BLOCK)
    cols = key_tile * BLOCK + col_lanes
    out_dims = d_slice * D_SLICE + tl.arange(0, D_SLICE)
    col_mask = cols < S
    out_dim_mask = out_dims < D
    dc_acc = tl.zeros([BLOCK, D_SLICE], dtype=tl.float32)
    flat_m: tl.constexpr = HEAD_TILE * ROW_BLOCK
    lanes = tl.arange(0, flat_m)
    lane_heads = lanes // ROW_BLOCK
    row_lanes = lanes - lane_heads * ROW_BLOCK
    for head_chunk in range(HEAD_GROUP // HEAD_TILE):
        heads = head_base + head_chunk * HEAD_TILE + lane_heads
        batch_heads = batch * H + heads
        first_query_block = key_tile * (BLOCK // ROW_BLOCK)
        for query_block in tl.range(first_query_block, Q_BLOCKS):
            rows = query_block * ROW_BLOCK + row_lanes
            row_mask = rows < S
            if ROW_BLOCK == BLOCK:
                triangle_prefix = query_block * (query_block + 1) // 2
            else:
                half = query_block // 2
                triangle_prefix = tl.where(
                    query_block % 2 == 0,
                    half * (half + 1),
                    (half + 1) * (half + 1),
                )
            triangle_id = triangle_prefix + key_tile
            elements = row_lanes[:, None] * BLOCK + col_lanes[None, :]
            scratch_base = (
                (batch_heads[:, None] * TRI_TILES + triangle_id)
                * ROW_BLOCK * BLOCK
            )
            score_mask = row_mask[:, None] & col_mask[None, :]
            if DS_ONLY:
                ds_scaled = tl.load(
                    pds + scratch_base + elements,
                    mask=score_mask, other=0.0,
                )
                row_lse = tl.load(lse + batch_heads * S + rows).to(
                    tl.float32
                )
                logits = tl.zeros([flat_m, BLOCK], dtype=tl.float32)
                for d_block in range(D // D_CHUNK):
                    score_dims = d_block * D_CHUNK + tl.arange(0, D_CHUNK)
                    q_score_offsets = (
                        (batch_heads[:, None] * S + rows[:, None]) * D
                        + score_dims[None, :]
                    )
                    c_score_offsets = (
                        (batch * S + cols[:, None]) * D
                        + score_dims[None, :]
                    )
                    q_score = tl.load(q + q_score_offsets)
                    c_score = tl.load(c_kv + c_score_offsets)
                    logits += tl.dot(
                        q_score, tl.trans(c_score), out_dtype=tl.float32,
                    )
                log2e: tl.constexpr = 1.4426950408889634
                probability = tl.where(
                    cols[None, :] <= rows[:, None],
                    tl.exp2(
                        (logits * sm_scale - row_lse[:, None]) * log2e
                    ),
                    0.0,
                )
            elif SEPARATE_PDS:
                probability = tl.load(
                    pds + scratch_base + elements,
                )
                ds_scaled = tl.load(
                    pds + PDS_PLANE + scratch_base + elements,
                )
            else:
                probability = tl.load(
                    pds + 2 * (scratch_base + elements),
                    mask=score_mask, other=0.0,
                )
                ds_scaled = tl.load(
                    pds + 2 * (scratch_base + elements) + 1,
                    mask=score_mask, other=0.0,
                )
            qdo_offsets = (
                (batch_heads[:, None] * S + rows[:, None]) * D
                + out_dims[None, :]
            )
            output_mask = row_mask[:, None] & out_dim_mask[None, :]
            if DS_ONLY:
                q_part = tl.load(q + qdo_offsets)
                do_part = tl.load(do + qdo_offsets)
                dc_acc += tl.dot(
                    tl.trans(ds_scaled.to(tl.bfloat16)),
                    q_part,
                    out_dtype=tl.float32,
                )
                dc_acc += tl.dot(
                    tl.trans(probability.to(tl.bfloat16)),
                    do_part,
                    out_dtype=tl.float32,
                )
            else:
                if SEPARATE_PDS:
                    q_part = tl.load(q + qdo_offsets).to(tl.float16)
                    do_part = tl.load(do + qdo_offsets).to(tl.float16)
                else:
                    q_part = tl.load(
                        q + qdo_offsets, mask=output_mask, other=0.0,
                    ).to(tl.bfloat16)
                    do_part = tl.load(
                        do + qdo_offsets, mask=output_mask, other=0.0,
                    ).to(tl.bfloat16)
                qdo = tl.reshape(
                    tl.join(q_part, do_part).permute(0, 2, 1),
                    [2 * flat_m, D_SLICE],
                )
                dsp = tl.reshape(
                    tl.join(tl.trans(ds_scaled), tl.trans(probability)),
                    [BLOCK, 2 * flat_m],
                )
                dc_acc += tl.dot(dsp, qdo, out_dtype=tl.float32)
    partial_offsets = (
        (batch_group * S + cols[:, None]) * D + out_dims[None, :]
    )
    if SEPARATE_PDS:
        tl.store(partial_dc + partial_offsets, dc_acc)
    else:
        tl.store(
            partial_dc + partial_offsets, dc_acc,
            mask=col_mask[:, None] & out_dim_mask[None, :],
        )
@triton.jit
def _delta_kernel(
    out,
    do,
    delta,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    block = tl.program_id(0).to(tl.int64)
    batch_head = tl.program_id(1).to(tl.int64)
    rows = block * BLOCK_R + tl.arange(0, BLOCK_R)
    dims = tl.arange(0, D)
    row_mask = rows < S
    offsets = (batch_head * S + rows[:, None]) * D + dims[None, :]
    mask = row_mask[:, None]
    out_values = tl.load(out + offsets, mask=mask, other=0.0).to(tl.float32)
    do_values = tl.load(do + offsets, mask=mask, other=0.0).to(tl.float32)
    values = tl.sum(out_values * do_values, axis=1)
    tl.store(delta + batch_head * S + rows, values, mask=row_mask)
@triton.jit
def _dq_owned_kernel(
    q,
    c_kv,
    do,
    out,
    lse,
    delta,
    dq,
    pds,
    sm_scale,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D_CHUNK: tl.constexpr,
    D_SLICE: tl.constexpr,
    PDS_PLANE: tl.constexpr = 0,
    DIAGONAL_SCHEDULE: tl.constexpr = False,
    L2_HEAD_CLUSTER: tl.constexpr = False,
    MMA_FP16: tl.constexpr = False,
    PERSIST_KV: tl.constexpr = False,
    INLINE_DELTA: tl.constexpr = False,
    WRITE_PDS: tl.constexpr = False,
    SEPARATE_PDS: tl.constexpr = False,
    FULL_TILES: tl.constexpr = False,
):
    q_blocks: tl.constexpr = (S + BLOCK_M - 1) // BLOCK_M
    if L2_HEAD_CLUSTER:
        owner = tl.program_id(0).to(tl.int64)
        d_slice = tl.program_id(1).to(tl.int64)
        batch_query = owner // H
        head = owner % H
        q_block = batch_query % q_blocks
        batch = batch_query // q_blocks
        batch_head = batch * H + head
    else:
        physical_q_block = tl.program_id(0).to(tl.int64)
        batch_head = tl.program_id(1).to(tl.int64)
        d_slice = tl.program_id(2).to(tl.int64)
        batch = batch_head // H
        head = batch_head - batch * H
        if DIAGONAL_SCHEDULE:
            q_block = (physical_q_block + head) % q_blocks
        else:
            q_block = physical_q_block
    rows = q_block * BLOCK_M + tl.arange(0, BLOCK_M)
    out_dims = d_slice * D_SLICE + tl.arange(0, D_SLICE)
    row_mask = rows < S
    out_dim_mask = out_dims < D
    if FULL_TILES:
        row_lse = tl.load(
            lse + batch_head * S + rows,
        ).to(tl.float32)
    else:
        row_lse = tl.load(
            lse + batch_head * S + rows, mask=row_mask, other=0.0
        ).to(tl.float32)
    if INLINE_DELTA:
        row_delta = tl.zeros([BLOCK_M], dtype=tl.float32)
        for d_block in range(D // D_CHUNK):
            delta_dims = d_block * D_CHUNK + tl.arange(0, D_CHUNK)
            delta_offsets = (
                (batch_head * S + rows[:, None]) * D
                + delta_dims[None, :]
            )
            out_part = tl.load(
                out + delta_offsets,
                mask=row_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            do_part = tl.load(
                do + delta_offsets,
                mask=row_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            row_delta += tl.sum(out_part * do_part, axis=1)
        tl.store(
            delta + batch_head * S + rows,
            row_delta,
            mask=row_mask,
        )
    else:
        if FULL_TILES:
            row_delta = tl.load(
                delta + batch_head * S + rows,
            ).to(tl.float32)
        else:
            row_delta = tl.load(
                delta + batch_head * S + rows, mask=row_mask, other=0.0
            ).to(tl.float32)
    dq_acc = tl.zeros([BLOCK_M, D_SLICE], dtype=tl.float32)
    log2e: tl.constexpr = 1.4426950408889634
    if PERSIST_KV:
        kv_smem = tle.gpu.alloc(
            [BLOCK_N, D], dtype=c_kv.dtype.element_ty,
            layout=None, scope=tle.gpu.smem,
        )
    end_n = tl.minimum((q_block + 1) * BLOCK_M, S)
    for start_n in tl.range(0, end_n, BLOCK_N):
        cols = start_n + tl.arange(0, BLOCK_N)
        col_mask = cols < S
        if PERSIST_KV:
            all_dims = tl.arange(0, D)
            kv_full = (
                (batch * S + cols[:, None]) * D + all_dims[None, :]
            )
            tle.gpu.copy(
                c_kv + kv_full, kv_smem, [BLOCK_N, D]
            )
        logits = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for d_block in range(D // D_CHUNK):
            dims = d_block * D_CHUNK + tl.arange(0, D_CHUNK)
            qdo_offsets = (batch_head * S + rows[:, None]) * D + dims[None, :]
            kv_offsets = (batch * S + cols[:, None]) * D + dims[None, :]
            if FULL_TILES:
                q_part = tl.load(q + qdo_offsets)
                do_part = tl.load(do + qdo_offsets)
            else:
                q_part = tl.load(q + qdo_offsets, mask=row_mask[:, None], other=0.0)
                do_part = tl.load(do + qdo_offsets, mask=row_mask[:, None], other=0.0)
            if PERSIST_KV:
                local_rows = tl.broadcast_to(
                    tl.arange(0, BLOCK_N)[:, None],
                    (BLOCK_N, D_CHUNK),
                )
                local_cols = tl.broadcast_to(
                    dims[None, :], (BLOCK_N, D_CHUNK),
                )
                c_part = tl.load(tle.gpu.local_ptr(
                    kv_smem, (local_rows, local_cols),
                ))
            else:
                if FULL_TILES:
                    c_part = tl.load(c_kv + kv_offsets)
                else:
                    c_part = tl.load(
                        c_kv + kv_offsets, mask=col_mask[:, None], other=0.0
                    )
            if MMA_FP16:
                q_part = q_part.to(tl.float16)
                do_part = do_part.to(tl.float16)
                c_part = c_part.to(tl.float16)
            logits += tl.dot(
                q_part, tl.trans(c_part), out_dtype=tl.float32,
            )
            dp += tl.dot(
                do_part, tl.trans(c_part), out_dtype=tl.float32,
            )
        if FULL_TILES:
            causal = cols[None, :] <= rows[:, None]
        else:
            causal = row_mask[:, None] & col_mask[None, :] & (
                cols[None, :] <= rows[:, None]
            )
        probability = tl.where(
            causal,
            tl.exp2((logits * sm_scale - row_lse[:, None]) * log2e),
            0.0,
        )
        ds_scaled = probability * (dp - row_delta[:, None]) * sm_scale
        if WRITE_PDS:
            tiles: tl.constexpr = (S + BLOCK_N - 1) // BLOCK_N
            if BLOCK_M == BLOCK_N:
                triangle_tiles: tl.constexpr = tiles * (tiles + 1) // 2
                triangle_prefix = q_block * (q_block + 1) // 2
            else:
                triangle_tiles: tl.constexpr = tiles * (tiles + 1)
                half = q_block // 2
                triangle_prefix = tl.where(
                    q_block % 2 == 0,
                    half * (half + 1),
                    (half + 1) * (half + 1),
                )
            key_block = start_n // BLOCK_N
            triangle_id = triangle_prefix + key_block
            row_lanes = tl.arange(0, BLOCK_M)
            col_lanes = tl.arange(0, BLOCK_N)
            elements = row_lanes[:, None] * BLOCK_N + col_lanes[None, :]
            scratch_mask = row_mask[:, None] & col_mask[None, :]
            scratch_base = (
                (batch_head * triangle_tiles + triangle_id)
                * BLOCK_M * BLOCK_N
            )
            if SEPARATE_PDS:
                tl.store(
                    pds + scratch_base + elements,
                    probability.to(tl.float16),
                    mask=scratch_mask,
                )
                tl.store(
                    pds + PDS_PLANE + scratch_base + elements,
                    ds_scaled.to(tl.float16),
                    mask=scratch_mask,
                )
            else:
                tl.store(
                    pds + 2 * (scratch_base + elements),
                    probability.to(tl.float16),
                    mask=scratch_mask,
                )
                tl.store(
                    pds + 2 * (scratch_base + elements) + 1,
                    ds_scaled.to(tl.float16),
                    mask=scratch_mask,
                )
        if PERSIST_KV:
            local_rows = tl.broadcast_to(
                tl.arange(0, BLOCK_N)[:, None], (BLOCK_N, D_SLICE),
            )
            local_cols = tl.broadcast_to(
                out_dims[None, :], (BLOCK_N, D_SLICE),
            )
            c_out = tl.load(tle.gpu.local_ptr(
                kv_smem, (local_rows, local_cols),
            ))
        else:
            c_out_offsets = (
                (batch * S + cols[:, None]) * D + out_dims[None, :]
            )
            if FULL_TILES:
                c_out = tl.load(c_kv + c_out_offsets)
            else:
                c_out = tl.load(
                    c_kv + c_out_offsets,
                    mask=col_mask[:, None] & out_dim_mask[None, :],
                    other=0.0,
                )
        if MMA_FP16:
            ds_mma = ds_scaled.to(tl.float16)
            c_out = c_out.to(tl.float16)
        else:
            ds_mma = ds_scaled.to(tl.bfloat16)
        dq_acc += tl.dot(
            ds_mma, c_out, out_dtype=tl.float32
        )
    dq_offsets = (batch_head * S + rows[:, None]) * D + out_dims[None, :]
    if FULL_TILES:
        tl.store(dq + dq_offsets, dq_acc)
    else:
        tl.store(
            dq + dq_offsets,
            dq_acc,
            mask=row_mask[:, None] & out_dim_mask[None, :],
        )
@triton.jit
def _dc_key_group_query_range(
    dc_acc,
    q,
    c_kv,
    do,
    lse,
    delta,
    sm_scale,
    batch_head,
    batch,
    cols,
    out_dims,
    col_mask,
    out_dim_mask,
    start_m,
    end_m,
    S: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D_CHUNK: tl.constexpr,
    D_SLICE: tl.constexpr,
    MASK_CAUSAL: tl.constexpr,
    MMA_FP16: tl.constexpr,
    FULL_TILES: tl.constexpr = False,
    JOINED_SCORE: tl.constexpr = False,
    JOINED_DKDV: tl.constexpr = False,
):
    """Accumulate one dC head over a masked or mask-free query range."""
    log2e: tl.constexpr = 1.4426950408889634
    for row_start in tl.range(start_m, end_m, BLOCK_M):
        rows = row_start + tl.arange(0, BLOCK_M)
        row_mask = rows < S
        if FULL_TILES:
            row_lse = tl.load(
                lse + batch_head * S + rows,
            ).to(tl.float32)
            row_delta = tl.load(
                delta + batch_head * S + rows,
            ).to(tl.float32)
        else:
            row_lse = tl.load(
                lse + batch_head * S + rows,
                mask=row_mask,
                other=0.0,
            ).to(tl.float32)
            row_delta = tl.load(
                delta + batch_head * S + rows,
                mask=row_mask,
                other=0.0,
            ).to(tl.float32)
        if JOINED_SCORE:
            joined_scores = tl.zeros(
                [2 * BLOCK_M, BLOCK_N], dtype=tl.float32,
            )
        else:
            logits = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for d_block in range(D // D_CHUNK):
            dims = d_block * D_CHUNK + tl.arange(0, D_CHUNK)
            qdo_offsets = (
                (batch_head * S + rows[:, None]) * D + dims[None, :]
            )
            kv_offsets = (
                (batch * S + cols[:, None]) * D + dims[None, :]
            )
            if FULL_TILES:
                q_part = tl.load(q + qdo_offsets)
                do_part = tl.load(do + qdo_offsets)
                c_part = tl.load(c_kv + kv_offsets)
            else:
                q_part = tl.load(
                    q + qdo_offsets,
                    mask=row_mask[:, None],
                    other=0.0,
                )
                do_part = tl.load(
                    do + qdo_offsets,
                    mask=row_mask[:, None],
                    other=0.0,
                )
                c_part = tl.load(
                    c_kv + kv_offsets,
                    mask=col_mask[:, None],
                    other=0.0,
                )
            if MMA_FP16:
                q_part = q_part.to(tl.float16)
                do_part = do_part.to(tl.float16)
                c_part = c_part.to(tl.float16)
            if JOINED_SCORE:
                qdo_joined = tl.reshape(
                    tl.join(q_part, do_part).permute(0, 2, 1),
                    [2 * BLOCK_M, D_CHUNK],
                )
                joined_scores += tl.dot(
                    qdo_joined,
                    tl.trans(c_part),
                    out_dtype=tl.float32,
                )
            else:
                logits += tl.dot(
                    q_part,
                    tl.trans(c_part),
                    out_dtype=tl.float32,
                )
                dp += tl.dot(
                    do_part,
                    tl.trans(c_part),
                    out_dtype=tl.float32,
                )
        if JOINED_SCORE:
            score_pairs = tl.reshape(
                joined_scores, [BLOCK_M, 2, BLOCK_N],
            ).permute(0, 2, 1)
            logits, dp = tl.split(score_pairs)
        if FULL_TILES:
            probability = tl.exp2(
                (logits * sm_scale - row_lse[:, None]) * log2e
            )
        else:
            valid = row_mask[:, None] & col_mask[None, :]
            if MASK_CAUSAL:
                valid = valid & (cols[None, :] <= rows[:, None])
            probability = tl.where(
                valid,
                tl.exp2((logits * sm_scale - row_lse[:, None]) * log2e),
                0.0,
            )
        ds_scaled = probability * (dp - row_delta[:, None]) * sm_scale
        qdo_out_offsets = (
            (batch_head * S + rows[:, None]) * D + out_dims[None, :]
        )
        out_mask = row_mask[:, None] & out_dim_mask[None, :]
        if FULL_TILES:
            q_out = tl.load(q + qdo_out_offsets)
            do_out = tl.load(do + qdo_out_offsets)
        else:
            q_out = tl.load(
                q + qdo_out_offsets,
                mask=out_mask,
                other=0.0,
            )
            do_out = tl.load(
                do + qdo_out_offsets,
                mask=out_mask,
                other=0.0,
            )
        if MMA_FP16:
            ds_mma = ds_scaled.to(tl.float16)
            probability_mma = probability.to(tl.float16)
            q_out = q_out.to(tl.float16)
            do_out = do_out.to(tl.float16)
        else:
            ds_mma = ds_scaled.to(tl.bfloat16)
            probability_mma = probability.to(tl.bfloat16)
        if JOINED_DKDV:
            qdo_joined = tl.reshape(
                tl.join(q_out, do_out).permute(0, 2, 1),
                [2 * BLOCK_M, D_SLICE],
            )
            dsp_joined = tl.reshape(
                tl.join(
                    tl.trans(ds_mma),
                    tl.trans(probability_mma),
                ),
                [BLOCK_N, 2 * BLOCK_M],
            )
            dc_acc += tl.dot(
                dsp_joined,
                qdo_joined,
                out_dtype=tl.float32,
            )
        else:
            dc_acc += tl.dot(
                tl.trans(ds_mma),
                q_out,
                out_dtype=tl.float32,
            )
            dc_acc += tl.dot(
                tl.trans(probability_mma),
                do_out,
                out_dtype=tl.float32,
            )
    return dc_acc
@triton.jit
def _tianshu_partition_fused_dqdc_kernel(
    q, c_kv, do, lse, delta, partial_dq, partial_dc, sm_scale,
    H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    NUM_GROUPS: tl.constexpr, HEAD_GROUP: tl.constexpr,
    PARTITIONS: tl.constexpr, BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr, D_CHUNK: tl.constexpr,
    D_SLICE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    group = pid % NUM_GROUPS
    batch_partition = pid // NUM_GROUPS
    partition = batch_partition % PARTITIONS
    batch = batch_partition // PARTITIONS
    key_blocks: tl.constexpr = S // BLOCK_N
    log2e: tl.constexpr = 1.4426950408889634
    row_lanes = tl.arange(0, BLOCK_M)
    col_lanes = tl.arange(0, BLOCK_N)
    for head_offset in range(HEAD_GROUP):
        head = group * HEAD_GROUP + head_offset
        batch_head = batch * H + head
        for row_start in tl.range(0, S, BLOCK_M):
            rows = row_start + row_lanes
            query_block = row_start // BLOCK_M
            max_key_block = (row_start + BLOCK_M - 1) // BLOCK_N
            row_lse = tl.load(lse + batch_head * S + rows).to(tl.float32)
            row_delta = tl.load(delta + batch_head * S + rows).to(tl.float32)
            for key_block in tl.range(
                partition, max_key_block + 1, PARTITIONS,
            ):
                key_start = key_block * BLOCK_N
                cols = key_start + col_lanes
                logits = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
                dp = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
                for d_block in range(D // D_CHUNK):
                    dims = d_block * D_CHUNK + tl.arange(0, D_CHUNK)
                    qdo_offsets = (
                        (batch_head * S + rows[:, None]) * D + dims[None, :]
                    )
                    c_offsets = (
                        (batch * S + cols[:, None]) * D + dims[None, :]
                    )
                    q_part = tl.load(q + qdo_offsets)
                    do_part = tl.load(do + qdo_offsets)
                    c_part = tl.load(c_kv + c_offsets)
                    logits += tl.dot(
                        q_part, tl.trans(c_part), out_dtype=tl.float32,
                    )
                    dp += tl.dot(
                        do_part, tl.trans(c_part), out_dtype=tl.float32,
                    )
                causal = cols[None, :] <= rows[:, None]
                probability = tl.where(
                    causal,
                    tl.exp2(
                        (logits * sm_scale - row_lse[:, None]) * log2e
                    ),
                    0.0,
                )
                ds_scaled = (
                    probability * (dp - row_delta[:, None]) * sm_scale
                ).to(tl.bfloat16)
                probability = probability.to(tl.bfloat16)
                for d_slice in range(D // D_SLICE):
                    out_dims = d_slice * D_SLICE + tl.arange(0, D_SLICE)
                    qdo_offsets = (
                        (batch_head * S + rows[:, None]) * D
                        + out_dims[None, :]
                    )
                    c_offsets = (
                        (batch * S + cols[:, None]) * D
                        + out_dims[None, :]
                    )
                    q_out = tl.load(q + qdo_offsets)
                    do_out = tl.load(do + qdo_offsets)
                    c_out = tl.load(c_kv + c_offsets)
                    dq_offsets = (
                        (((partition * H + head) * S + rows[:, None]) * D)
                        + out_dims[None, :]
                        + batch * H * S * D * PARTITIONS
                    )
                    dq_previous = (
                        (rows[:, None] >= 0) & (out_dims[None, :] >= 0)
                        & (key_block != partition)
                    )
                    dq_tile = tl.load(
                        partial_dq + dq_offsets,
                        mask=dq_previous, other=0.0,
                    ).to(tl.float32)
                    dq_tile += tl.dot(
                        ds_scaled, c_out, out_dtype=tl.float32,
                    )
                    tl.store(
                        partial_dq + dq_offsets, dq_tile.to(tl.bfloat16),
                    )
                    dc_offsets = (
                        ((batch * NUM_GROUPS + group) * S + cols[:, None]) * D
                        + out_dims[None, :]
                    )
                    dc_previous = (
                        (cols[:, None] >= 0) & (out_dims[None, :] >= 0)
                        & ((head_offset != 0) | (row_start != key_start))
                    )
                    dc_tile = tl.load(
                        partial_dc + dc_offsets,
                        mask=dc_previous, other=0.0,
                    ).to(tl.float32)
                    dc_tile += tl.dot(
                        tl.trans(ds_scaled), q_out, out_dtype=tl.float32,
                    )
                    dc_tile += tl.dot(
                        tl.trans(probability), do_out, out_dtype=tl.float32,
                    )
                    tl.store(
                        partial_dc + dc_offsets, dc_tile.to(tl.float16),
                    )

@triton.jit
def _reduce_partition_dq_kernel(
    partial_dq, dq, n_elements,
    H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    PARTITIONS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = (
        tl.program_id(0).to(tl.int64) * BLOCK
        + tl.arange(0, BLOCK).to(tl.int64)
    )
    mask = offsets < n_elements
    rows = (offsets // D) % S
    query_blocks = rows // BLOCK_M
    max_key_blocks = (
        query_blocks * BLOCK_M + BLOCK_M - 1
    ) // BLOCK_N
    batch_stride: tl.constexpr = H * S * D
    batch = offsets // batch_stride
    within_batch = offsets - batch * batch_stride
    values = tl.zeros([BLOCK], dtype=tl.float32)
    for partition in range(PARTITIONS):
        values += tl.load(
            partial_dq
            + (batch * PARTITIONS + partition) * batch_stride
            + within_batch,
            mask=mask & (partition <= max_key_blocks), other=0.0,
        )
    tl.store(dq + offsets, values, mask=mask)

@triton.jit
def _dc_key_group_owned_kernel(
    q,
    c_kv,
    do,
    lse,
    delta,
    partial_dc,
    sm_scale,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    HEAD_GROUP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D_CHUNK: tl.constexpr,
    D_SLICE: tl.constexpr,
    DIAGONAL_SCHEDULE: tl.constexpr = False,
    L2_HEAD_CLUSTER: tl.constexpr = False,
    CAUSAL_SPLIT: tl.constexpr = False,
    MMA_FP16: tl.constexpr = False,
    JOINED_SCORE: tl.constexpr = False,
    JOINED_DKDV: tl.constexpr = False,
):
    key_blocks: tl.constexpr = (S + BLOCK_N - 1) // BLOCK_N
    if L2_HEAD_CLUSTER:
        owner = tl.program_id(0).to(tl.int64)
        d_slice = tl.program_id(1).to(tl.int64)
        batch_key = owner // NUM_GROUPS
        group = owner % NUM_GROUPS
        key_block = batch_key % key_blocks
        batch = batch_key // key_blocks
        batch_group = batch * NUM_GROUPS + group
    else:
        physical_key_block = tl.program_id(0).to(tl.int64)
        batch_group = tl.program_id(1).to(tl.int64)
        d_slice = tl.program_id(2).to(tl.int64)
        group = batch_group % NUM_GROUPS
        batch = batch_group // NUM_GROUPS
        if DIAGONAL_SCHEDULE:
            key_block = (physical_key_block + group) % key_blocks
        else:
            key_block = physical_key_block
    cols = key_block * BLOCK_N + tl.arange(0, BLOCK_N)
    out_dims = d_slice * D_SLICE + tl.arange(0, D_SLICE)
    col_mask = cols < S
    out_dim_mask = out_dims < D
    dc_acc = tl.zeros([BLOCK_N, D_SLICE], dtype=tl.float32)
    for head_offset in range(HEAD_GROUP):
        head = group * HEAD_GROUP + head_offset
        batch_head = batch * H + head
        if CAUSAL_SPLIT:
            diagonal_end = tl.minimum((key_block + 1) * BLOCK_N, S)
            dc_acc = _dc_key_group_query_range(
                dc_acc,
                q,
                c_kv,
                do,
                lse,
                delta,
                sm_scale,
                batch_head,
                batch,
                cols,
                out_dims,
                col_mask,
                out_dim_mask,
                key_block * BLOCK_N,
                diagonal_end,
                S,
                D,
                BLOCK_M // 2,
                BLOCK_N,
                D_CHUNK,
                D_SLICE,
                MASK_CAUSAL=True,
                MMA_FP16=MMA_FP16,
                JOINED_SCORE=JOINED_SCORE,
                JOINED_DKDV=JOINED_DKDV,
            )
            dc_acc = _dc_key_group_query_range(
                dc_acc,
                q,
                c_kv,
                do,
                lse,
                delta,
                sm_scale,
                batch_head,
                batch,
                cols,
                out_dims,
                col_mask,
                out_dim_mask,
                diagonal_end,
                S,
                S,
                D,
                BLOCK_M,
                BLOCK_N,
                D_CHUNK,
                D_SLICE,
                MASK_CAUSAL=False,
                FULL_TILES=True,
                MMA_FP16=MMA_FP16,
                JOINED_SCORE=JOINED_SCORE,
                JOINED_DKDV=JOINED_DKDV,
            )
        else:
            dc_acc = _dc_key_group_query_range(
                dc_acc,
                q,
                c_kv,
                do,
                lse,
                delta,
                sm_scale,
                batch_head,
                batch,
                cols,
                out_dims,
                col_mask,
                out_dim_mask,
                key_block * BLOCK_N,
                S,
                S,
                D,
                BLOCK_M,
                BLOCK_N,
                D_CHUNK,
                D_SLICE,
                MASK_CAUSAL=True,
                MMA_FP16=MMA_FP16,
                JOINED_SCORE=JOINED_SCORE,
                JOINED_DKDV=JOINED_DKDV,
            )
    partial_offsets = (
        ((batch_group * S + cols[:, None]) * D) + out_dims[None, :]
    )
    tl.store(
        partial_dc + partial_offsets,
        dc_acc,
        mask=col_mask[:, None] & out_dim_mask[None, :],
    )
@triton.jit
def _ascend_fixed_reduce_partial_dc_kernel(
    partial_dc,
    dc_kv,
    n_elements,
    S: tl.constexpr,
    D: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    TOTAL_BLOCKS: tl.constexpr,
    NCORE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    lanes = tl.arange(0, BLOCK)
    for base in tl.range(0, TOTAL_BLOCKS, NCORE):
        block_idx = base + pid
        if block_idx < TOTAL_BLOCKS:
            offsets = block_idx * BLOCK + lanes
            mask = offsets < n_elements
            batch = offsets // (S * D)
            within_batch = offsets % (S * D)
            values = tl.zeros([BLOCK], dtype=tl.float32)
            for group in range(NUM_GROUPS):
                partial_offsets = (
                    (batch * NUM_GROUPS + group) * S * D
                    + within_batch
                )
                values += tl.load(
                    partial_dc + partial_offsets,
                    mask=mask,
                    other=0.0,
                )
            tl.store(dc_kv + offsets, values, mask=mask)
@triton.jit
def _reduce_partial_dc_kernel(
    partial_dc,
    dc_kv,
    n_elements,
    S: tl.constexpr,
    D: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    batch = offsets // (S * D)
    within_batch = offsets % (S * D)
    values = tl.zeros([BLOCK], dtype=tl.float32)
    for group in range(NUM_GROUPS):
        partial_offsets = ((batch * NUM_GROUPS + group) * S * D) + within_batch
        values += tl.load(partial_dc + partial_offsets, mask=mask, other=0.0)
    tl.store(dc_kv + offsets, values, mask=mask)
@triton.jit
def _per_head_fused_owner_kernel(
    q,
    c_kv,
    do,
    lse,
    delta,
    dq,
    partial_dc,
    sm_scale,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    HEAD_GROUP: tl.constexpr,
    BLOCK: tl.constexpr,
    D_CHUNK: tl.constexpr,
    JOINED_SCORE: tl.constexpr = False,
    JOINED_DKDV: tl.constexpr = False,
    FULL_TILES: tl.constexpr = False,
    FP16_DC_ACC: tl.constexpr = False,
):
    """Grouped resident dC followed by unique per-head dQ ownership."""
    log2e: tl.constexpr = 1.4426950408889634
    block = tl.program_id(0).to(tl.int64)
    batch_group = tl.program_id(1).to(tl.int64)
    batch = batch_group // NUM_GROUPS
    group = batch_group - batch * NUM_GROUPS
    lanes = tl.arange(0, BLOCK)
    dims = tl.arange(0, D)
    positions = block * BLOCK + lanes
    position_mask = positions < S
    c_resident_offsets = (
        (batch * S + positions[:, None]) * D + dims[None, :]
    )
    if FULL_TILES:
        c_resident = tle.load(
            c_kv + c_resident_offsets, is_async=True,
        )
    else:
        c_resident = tl.load(
            c_kv + c_resident_offsets,
            mask=position_mask[:, None], other=0.0,
        )
    if FP16_DC_ACC:
        dc_acc = tl.zeros([BLOCK, D], dtype=tl.float16)
    else:
        dc_acc = tl.zeros([BLOCK, D], dtype=tl.float32)
    for head_offset in range(HEAD_GROUP):
        head = group * HEAD_GROUP + head_offset
        batch_head = batch * H + head
        for start_m in tl.range(block * BLOCK, S, BLOCK):
            rows = start_m + lanes
            row_mask = rows < S
            if FULL_TILES:
                row_lse = tl.load(lse + batch_head * S + rows).to(tl.float32)
                row_delta = tl.load(delta + batch_head * S + rows).to(tl.float32)
            else:
                row_lse = tl.load(
                    lse + batch_head * S + rows, mask=row_mask, other=0.0,
                ).to(tl.float32)
                row_delta = tl.load(
                    delta + batch_head * S + rows, mask=row_mask, other=0.0,
                ).to(tl.float32)
            qdo_offsets = (
                (batch_head * S + rows[:, None]) * D + dims[None, :]
            )
            if FULL_TILES:
                q_full = tle.load(q + qdo_offsets, is_async=True)
                do_full = tle.load(do + qdo_offsets, is_async=True)
            else:
                q_full = tl.load(
                    q + qdo_offsets, mask=row_mask[:, None], other=0.0,
                )
                do_full = tl.load(
                    do + qdo_offsets, mask=row_mask[:, None], other=0.0,
                )
            if JOINED_SCORE:
                qdo_joined = tl.reshape(
                    tl.join(q_full, do_full).permute(0, 2, 1),
                    [2 * BLOCK, D],
                )
                joined_scores = tl.dot(
                    qdo_joined,
                    tl.trans(c_resident),
                    out_dtype=tl.float32,
                )
                score_pairs = tl.reshape(
                    joined_scores, [BLOCK, 2, BLOCK],
                ).permute(0, 2, 1)
                logits, dp = tl.split(score_pairs)
            else:
                logits = tl.dot(
                    q_full, tl.trans(c_resident), out_dtype=tl.float32,
                )
                dp = tl.dot(
                    do_full, tl.trans(c_resident), out_dtype=tl.float32,
                )
            causal = positions[None, :] <= rows[:, None]
            if not FULL_TILES:
                causal = causal & row_mask[:, None] & position_mask[None, :]
            probability = tl.where(
                causal,
                tl.exp2((logits * sm_scale - row_lse[:, None]) * log2e),
                0.0,
            )
            ds = probability * (dp - row_delta[:, None]) * sm_scale
            if JOINED_DKDV:
                qdo_joined = tl.reshape(
                    tl.join(q_full, do_full).permute(0, 2, 1),
                    [2 * BLOCK, D],
                )
                dsp_joined = tl.reshape(
                    tl.join(
                        tl.trans(ds.to(tl.bfloat16)),
                        tl.trans(probability.to(tl.bfloat16)),
                    ),
                    [BLOCK, 2 * BLOCK],
                )
                dc_acc += tl.dot(
                    dsp_joined, qdo_joined, out_dtype=tl.float32,
                )
            else:
                if FP16_DC_ACC:
                    dc_acc = tl.dot(
                        tl.trans(ds.to(tl.bfloat16)), q_full, dc_acc,
                        out_dtype=tl.float16,
                    )
                    dc_acc = tl.dot(
                        tl.trans(probability.to(tl.bfloat16)), do_full,
                        dc_acc, out_dtype=tl.float16,
                    )
                else:
                    dc_acc += tl.dot(
                        tl.trans(ds.to(tl.bfloat16)), q_full,
                        out_dtype=tl.float32,
                    )
                    dc_acc += tl.dot(
                        tl.trans(probability.to(tl.bfloat16)), do_full,
                        out_dtype=tl.float32,
                    )
    partial_offsets = (
        (batch_group * S + positions[:, None]) * D + dims[None, :]
    )
    if FULL_TILES:
        tl.store(partial_dc + partial_offsets, dc_acc)
    else:
        tl.store(
            partial_dc + partial_offsets, dc_acc,
            mask=position_mask[:, None],
        )
    for head_offset in range(HEAD_GROUP):
        head = group * HEAD_GROUP + head_offset
        batch_head = batch * H + head
        rows = positions
        row_mask = position_mask
        if FULL_TILES:
            row_lse = tl.load(lse + batch_head * S + rows).to(tl.float32)
            row_delta = tl.load(delta + batch_head * S + rows).to(tl.float32)
        else:
            row_lse = tl.load(
                lse + batch_head * S + rows, mask=row_mask, other=0.0,
            ).to(tl.float32)
            row_delta = tl.load(
                delta + batch_head * S + rows, mask=row_mask, other=0.0,
            ).to(tl.float32)
        qdo_offsets = (
            (batch_head * S + rows[:, None]) * D + dims[None, :]
        )
        if FULL_TILES:
            q_resident = tle.load(q + qdo_offsets, is_async=True)
            do_resident = tle.load(do + qdo_offsets, is_async=True)
        else:
            q_resident = tl.load(
                q + qdo_offsets, mask=row_mask[:, None], other=0.0,
            )
            do_resident = tl.load(
                do + qdo_offsets, mask=row_mask[:, None], other=0.0,
            )
        dq_acc = tl.zeros([BLOCK, D], dtype=tl.float32)
        end_n = tl.minimum((block + 1) * BLOCK, S)
        for start_n in tl.range(0, end_n, BLOCK):
            cols = start_n + lanes
            col_mask = cols < S
            c_offsets = (
                (batch * S + cols[:, None]) * D + dims[None, :]
            )
            if FULL_TILES:
                c_full = tle.load(c_kv + c_offsets, is_async=True)
            else:
                c_full = tl.load(
                    c_kv + c_offsets, mask=col_mask[:, None], other=0.0,
                )
            if JOINED_SCORE:
                qdo_joined = tl.reshape(
                    tl.join(q_resident, do_resident).permute(0, 2, 1),
                    [2 * BLOCK, D],
                )
                joined_scores = tl.dot(
                    qdo_joined,
                    tl.trans(c_full),
                    out_dtype=tl.float32,
                )
                score_pairs = tl.reshape(
                    joined_scores, [BLOCK, 2, BLOCK],
                ).permute(0, 2, 1)
                logits, dp = tl.split(score_pairs)
            else:
                logits = tl.dot(
                    q_resident, tl.trans(c_full), out_dtype=tl.float32,
                )
                dp = tl.dot(
                    do_resident, tl.trans(c_full), out_dtype=tl.float32,
                )
            causal = cols[None, :] <= rows[:, None]
            if not FULL_TILES:
                causal = causal & row_mask[:, None] & col_mask[None, :]
            probability = tl.where(
                causal,
                tl.exp2((logits * sm_scale - row_lse[:, None]) * log2e),
                0.0,
            )
            ds = probability * (dp - row_delta[:, None]) * sm_scale
            dq_acc += tl.dot(
                ds.to(tl.bfloat16), c_full, out_dtype=tl.float32,
            )
        dq_offsets = (batch_head * S + rows[:, None]) * D + dims[None, :]
        if FULL_TILES:
            tl.store(dq + dq_offsets, dq_acc)
        else:
            tl.store(dq + dq_offsets, dq_acc, mask=row_mask[:, None])
@triton.jit
def _reduce_per_head_dc_kernel(
    partial_dc,
    dc_kv,
    n_elements,
    NUM_GROUPS: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    batch = offsets // (S * D)
    within_batch = offsets % (S * D)
    values = tl.zeros([BLOCK], dtype=tl.float32)
    for group in range(NUM_GROUPS):
        partial_offsets = (
            (batch * NUM_GROUPS + group) * S * D + within_batch
        )
        values += tl.load(
            partial_dc + partial_offsets,
            mask=mask,
            other=0.0,
        )
    tl.store(dc_kv + offsets, values, mask=mask)
def _run_per_head_fused_owner(
    q,
    c_kv,
    do,
    lse,
    delta,
    dq,
    dc_kv,
    sm_scale,
    B,
    H,
    S,
    D,
    block,
    d_chunk,
    warps,
    head_group=1,
    joined_score=False,
    joined_dkdv=False,
    full_tiles=False,
    fp16_dc_acc=False,
    stages=1,
    waves_per_eu=None,
):
    num_groups = H // head_group
    partial_dc = torch.empty(
        (B, num_groups, S, D),
        device=q.device,
        dtype=torch.float16,
    )
    launch = dict(num_warps=warps, num_stages=stages)
    if waves_per_eu is not None:
        launch["waves_per_eu"] = waves_per_eu
    _per_head_fused_owner_kernel[(triton.cdiv(S, block), B * num_groups)](
        q,
        c_kv,
        do,
        lse,
        delta,
        dq,
        partial_dc,
        sm_scale,
        H=H,
        S=S,
        D=D,
        NUM_GROUPS=num_groups,
        HEAD_GROUP=head_group,
        BLOCK=block,
        D_CHUNK=d_chunk,
        JOINED_SCORE=joined_score,
        JOINED_DKDV=joined_dkdv,
        FULL_TILES=full_tiles,
        FP16_DC_ACC=fp16_dc_acc,
        **launch,
    )
    n_dc = B * S * D
    reduce_launch = dict(num_warps=4, num_stages=1)
    if waves_per_eu is not None:
        reduce_launch["waves_per_eu"] = waves_per_eu
    _reduce_per_head_dc_kernel[(triton.cdiv(n_dc, 512),)](
        partial_dc,
        dc_kv,
        n_dc,
        NUM_GROUPS=num_groups, S=S, D=D,
        BLOCK=512,
        **reduce_launch,
    )
    return dq, dc_kv
def _run_materialized_pds_owner(
    q, c_kv, out, do, lse, dq, dc_kv, sm_scale, B, H, S, D, backend,
):
    ascend_dsa = backend == "ascend"
    ds_only = backend == "tianshu_ds"
    separate_pds = backend == "hygon"
    if ascend_dsa:
        block, dq_d_slice, dc_d_slice = 16, 512, 512
        producer_d_chunk = 128
        producer_head_tile = 1
        head_group, head_tile, warps, waves = 4, 1, 4, None
    elif backend == "hygon":
        block, dq_d_slice, dc_d_slice = 32, 128, 128
        producer_d_chunk = 128
        producer_head_tile = 1
        head_group, head_tile, warps, waves = 8, 1, 4, 1
    elif backend == "metax":
        block, dq_d_slice, dc_d_slice = 16, 512, 128
        producer_d_chunk = 64
        producer_head_tile = 2
        head_group, head_tile, warps, waves = 2, 1, 4, None
    elif backend == "nvidia":
        block, dq_d_slice, dc_d_slice = 32, 512, 512
        producer_d_chunk = 128
        producer_head_tile = 2
        head_group, head_tile, warps, waves = 4, 1, 8, None
    elif backend == "thead":
        block, dq_d_slice, dc_d_slice = 32, 256, 128
        producer_d_chunk = 128
        producer_head_tile = 2
        head_group, head_tile, warps, waves = 8, 2, 4, None
    elif ds_only:
        block, dq_d_slice, dc_d_slice = 32, 512, 128
        producer_d_chunk = 128
        producer_head_tile = 1
        head_group, head_tile, warps, waves = 4, 1, 4, None
    else:
        block, dq_d_slice, dc_d_slice = 32, 256, 128
        producer_d_chunk = 64
        producer_head_tile = 4
        head_group, head_tile, warps, waves = 4, 2, 4, None
    tiles = triton.cdiv(S, block)
    triangle_tiles = tiles * (tiles + 1) // 2
    pds_plane = B * H * triangle_tiles * block * block
    scratch_elements = pds_plane if ds_only else pds_plane * 2
    pds = torch.empty(
        (scratch_elements,),
        device=q.device,
        dtype=(
            torch.bfloat16
            if ascend_dsa or backend == "metax" else torch.float16
        ),
    )
    delta = torch.empty((B, H, S), device=q.device, dtype=torch.float32)
    num_groups = H // head_group
    partial_dc = torch.empty(
        (B, num_groups, S, D),
        device=q.device,
        dtype=torch.float32 if ascend_dsa else torch.float16,
    )
    launch = dict(num_warps=warps, num_stages=1)
    if waves is not None:
        launch["waves_per_eu"] = waves
    if ascend_dsa:
        _delta_kernel[(triton.cdiv(S, 16), B * H)](
            out,
            do,
            delta,
            H=H,
            S=S,
            D=D,
            BLOCK_R=16,
            num_warps=1,
            num_stages=1,
        )
    else:
        _delta_kernel[(triton.cdiv(S, 4), B * H)](
            out,
            do,
            delta,
            H=H,
            S=S,
            D=D,
            BLOCK_R=4,
            **launch,
        )
    l2_cluster = not ascend_dsa
    producer_grid = (
        (B * triangle_tiles * (H // producer_head_tile),)
        if l2_cluster else
        (tiles, tiles, B * (H // producer_head_tile))
    )
    _materialize_pds_triangle_kernel[producer_grid](
        q,
        c_kv,
        do,
        lse,
        delta,
        pds,
        sm_scale,
        H=H,
        S=S,
        D=D,
        TRI_TILES=triangle_tiles,
        PDS_PLANE=pds_plane,
        HEAD_TILE=producer_head_tile,
        BLOCK=block,
        D_CHUNK=producer_d_chunk,
        SEPARATE_PDS=separate_pds,
        ASCEND_DSA=ascend_dsa,
        L2_HEAD_CLUSTER=l2_cluster,
        DS_ONLY=ds_only,
        **launch,
    )
    dq_grid = (
        (B * tiles * H, D // dq_d_slice)
        if l2_cluster else
        (tiles, B * H, D // dq_d_slice)
    )
    _dq_from_materialized_pds_kernel[dq_grid](
        c_kv,
        pds,
        dq,
        H=H,
        S=S,
        D=D,
        TRI_TILES=triangle_tiles,
        PDS_PLANE=pds_plane,
        BLOCK=block,
        D_SLICE=dq_d_slice,
        SEPARATE_PDS=separate_pds,
        ASCEND_DSA=ascend_dsa,
        L2_HEAD_CLUSTER=l2_cluster,
        DS_ONLY=ds_only,
        **launch,
    )
    _dc_from_materialized_pds_kernel[
        (tiles, B * num_groups, D // dc_d_slice)
    ](
        q,
        c_kv,
        do,
        lse,
        pds,
        partial_dc,
        sm_scale,
        H=H,
        S=S,
        D=D,
        NUM_GROUPS=num_groups,
        HEAD_GROUP=head_group,
        HEAD_TILE=head_tile,
        TILES=tiles,
        Q_BLOCKS=tiles,
        TRI_TILES=triangle_tiles,
        PDS_PLANE=pds_plane,
        BLOCK=block,
        ROW_BLOCK=block,
        D_CHUNK=producer_d_chunk,
        D_SLICE=dc_d_slice,
        SEPARATE_PDS=separate_pds,
        ASCEND_DSA=ascend_dsa,
        DS_ONLY=ds_only,
        **launch,
    )
    dc_elements = B * S * D
    reduce_launch = dict(num_warps=4, num_stages=1)
    if waves is not None:
        reduce_launch["waves_per_eu"] = waves
    _reduce_partial_dc_kernel[(triton.cdiv(dc_elements, 256),)](
        partial_dc,
        dc_kv,
        dc_elements,
        S=S,
        D=D,
        NUM_GROUPS=num_groups,
        BLOCK=256,
        **reduce_launch,
    )
    return dq, dc_kv
def _run_tianshu_partition_fused_owner(
    q, c_kv, out, do, lse, dq, dc_kv, sm_scale, B, H, S, D,
    partition_cap=8, block_n=32,
):
    head_group = 4
    num_groups = H // head_group
    block_m = 32
    key_blocks = S // block_n
    partitions = min(partition_cap, key_blocks)
    partial_dq = torch.empty(
        (B, partitions, H, S, D), device=q.device, dtype=torch.bfloat16,
    )
    partial_dc = torch.empty(
        (B, num_groups, S, D), device=q.device, dtype=torch.float16,
    )
    delta = torch.empty((B, H, S), device=q.device, dtype=torch.float32)
    _delta_kernel[(triton.cdiv(S, 4), B * H)](
        out, do, delta, H=H, S=S, D=D, BLOCK_R=4,
        num_warps=4, num_stages=1,
    )
    grid = (B * partitions * num_groups,)
    _tianshu_partition_fused_dqdc_kernel[grid](
        q, c_kv, do, lse, delta, partial_dq, partial_dc, sm_scale,
        H=H, S=S, D=D, NUM_GROUPS=num_groups, HEAD_GROUP=head_group,
        PARTITIONS=partitions, BLOCK_M=block_m, BLOCK_N=block_n,
        D_CHUNK=128, D_SLICE=128,
        num_warps=4, num_stages=1,
    )
    dq_elements = B * H * S * D
    _reduce_partition_dq_kernel[(triton.cdiv(dq_elements, 256),)](
        partial_dq, dq, dq_elements,
        H=H, S=S, D=D, PARTITIONS=partitions,
        BLOCK_M=block_m, BLOCK_N=block_n,
        BLOCK=256, num_warps=4, num_stages=1,
    )
    dc_elements = B * S * D
    _reduce_partial_dc_kernel[(triton.cdiv(dc_elements, 256),)](
        partial_dc, dc_kv, dc_elements,
        S=S, D=D, NUM_GROUPS=num_groups, BLOCK=256,
        num_warps=4, num_stages=1,
    )
    return dq, dc_kv
def _run_ascend_fixed_owner(
    q, c_kv, out, do, lse, dq, dc_kv, sm_scale, B, H, S, D,
):
    """Known-good 910B4 bounded physical-core ownership."""
    head_group = 4
    num_groups = H // head_group
    delta = torch.empty((B, H, S), device=q.device, dtype=torch.float32)
    partial_dc = torch.empty(
        (B, num_groups, S, D), device=q.device, dtype=torch.float32,
    )
    total_rows = B * H * S
    delta_ncore = min(total_rows, 40)
    _ascend_fixed_delta_kernel[(delta_ncore,)](
        out,
        do,
        delta,
        S=S,
        D=D,
        TOTAL_ROWS=total_rows,
        NCORE=delta_ncore,
        num_warps=1,
        num_stages=1,
    )
    q_blocks = triton.cdiv(S, 16)
    dq_tasks = B * H * q_blocks
    dq_ncore = min(dq_tasks, 20)
    _ascend_fixed_dq_worker_kernel[(dq_ncore,)](
        q,
        c_kv,
        do,
        lse,
        delta,
        dq,
        sm_scale,
        H=H,
        S=S,
        D=D,
        Q_BLOCKS=q_blocks,
        D_SLICES=1,
        TOTAL_TASKS=dq_tasks,
        NCORE=dq_ncore,
        BLOCK_M=16,
        BLOCK_N=32,
        D_CHUNK=128,
        D_SLICE=512,
        num_warps=4,
        num_stages=1,
    )
    key_blocks = triton.cdiv(S, 32)
    dc_tasks = B * num_groups * key_blocks
    dc_ncore = min(dc_tasks, 20)
    _ascend_fixed_dc_worker_kernel[(dc_ncore,)](
        q,
        c_kv,
        do,
        lse,
        delta,
        partial_dc,
        sm_scale,
        H=H,
        S=S,
        D=D,
        NUM_GROUPS=num_groups,
        HEAD_GROUP=head_group,
        KEY_BLOCKS=key_blocks,
        D_SLICES=1,
        TOTAL_TASKS=dc_tasks,
        NCORE=dc_ncore,
        BLOCK_M=32,
        BLOCK_N=32,
        D_CHUNK=128,
        D_SLICE=512,
        num_warps=4,
        num_stages=1,
    )
    dc_elements = B * S * D
    reduce_blocks = triton.cdiv(dc_elements, 256)
    reduce_ncore = min(reduce_blocks, 40)
    _ascend_fixed_reduce_partial_dc_kernel[(reduce_ncore,)](
        partial_dc,
        dc_kv,
        dc_elements,
        S=S,
        D=D,
        NUM_GROUPS=num_groups,
        TOTAL_BLOCKS=reduce_blocks,
        NCORE=reduce_ncore,
        BLOCK=256,
        num_warps=1,
        num_stages=1,
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
    ascend_path = _is_ascend(q)
    if ascend_path:
        return _run_ascend_fixed_owner(
            q, c_kv, out, do, lse, dq, dc_kv, sm_scale, B, H, S, D,
        )
    hygon_path = _is_hygon(q)
    metax_path = _is_metax(q)
    tianshu_low_state_path = _is_tianshu(q)
    if tle is not None and (hygon_path or metax_path or tianshu_low_state_path):
        pipe_delta = torch.empty((2, S), device=q.device, dtype=torch.float32)
        _delta_kernel[(triton.cdiv(S, 16), 2)](
            out, do, pipe_delta, H=H, S=S, D=D, BLOCK_R=16,
            num_warps=4, num_stages=1,
        )
        pipe_probe = torch.empty((16, D), device=q.device, dtype=torch.float16)
        _tle_pipe_bwd_dc_kernel[(1, 1)](
            q, c_kv, do, lse, pipe_delta, pipe_probe, sm_scale,
            H=H, S=S, D=D, NUM_GROUPS=32, HEAD_GROUP=2,
            BLOCK_M=16, BLOCK_N=16, PIPE_CAPACITY=2,
            num_warps=4, num_stages=1,
        )
    if hygon_path:
        return _run_materialized_pds_owner(
            q, c_kv, out, do, lse, dq, dc_kv, sm_scale,
            B, H, S, D, "hygon",
        )
    if metax_path:
        return _run_materialized_pds_owner(
            q, c_kv, out, do, lse, dq, dc_kv, sm_scale,
            B, H, S, D, "metax",
        )
    if tianshu_low_state_path:
        return _run_tianshu_partition_fused_owner(
            q, c_kv, out, do, lse, dq, dc_kv, sm_scale, B, H, S, D,
            block_n=64,
        )
    if _is_thead(q):
        return _run_tianshu_partition_fused_owner(
            q, c_kv, out, do, lse, dq, dc_kv, sm_scale, B, H, S, D,
            partition_cap=64,
        )
    if S <= 4096:
        delta = torch.empty((B, H, S), device=q.device, dtype=torch.float32)
        _delta_kernel[(triton.cdiv(S, 16), B * H)](
            out, do, delta, H=H, S=S, D=D, BLOCK_R=16,
            num_warps=4, num_stages=1,
        )
        return _run_per_head_fused_owner(
            q, c_kv, do, lse, delta, dq, dc_kv, sm_scale,
            B, H, S, D, 32, 512, 4,
            head_group=2,
            full_tiles=True,
            stages=1,
        )
    head_group = 4
    num_groups = H // head_group
    dc_block_m = 32
    fast_legacy_path = _is_nvidia_or_tianshu(q)
    dq_block_m = (
        16 if B == 1 and S >= 4096 else 32
    ) if fast_legacy_path else 16
    block_n = 64
    dq_d_chunk = 128 if S <= 256 or S >= 2048 else 64
    dc_d_chunk = 64 if S <= 1024 else 128
    metax_path = _is_metax(q)
    dq_d_slice = 512
    dc_d_slice = 256
    delta_block_r = 16
    delta_warps = 8
    delta = torch.empty((B, H, S), device=q.device, dtype=torch.float32)
    partial_dc = torch.empty(
        (B, num_groups, S, D),
        device=q.device,
        dtype=torch.float16 if fast_legacy_path else torch.float32,
    )
    _delta_kernel[(triton.cdiv(S, delta_block_r), B * H)](
        out,
        do,
        delta,
        H=H,
        S=S,
        D=D,
        BLOCK_R=delta_block_r,
        num_warps=delta_warps,
        num_stages=1,
    )
    q_blocks = triton.cdiv(S, dq_block_m)
    _dq_owned_kernel[(B * q_blocks * H, D // dq_d_slice)](
        q,
        c_kv,
        do,
        out,
        lse,
        delta,
        dq,
        dq,
        sm_scale,
        H=H,
        S=S,
        D=D,
        BLOCK_M=dq_block_m,
        BLOCK_N=block_n,
        D_CHUNK=dq_d_chunk,
        D_SLICE=dq_d_slice,
        PDS_PLANE=0,
        L2_HEAD_CLUSTER=True,
        INLINE_DELTA=False,
        num_warps=8,
        num_stages=1,
    )
    key_blocks = triton.cdiv(S, block_n)
    _dc_key_group_owned_kernel[
        (B * key_blocks * num_groups, D // dc_d_slice)
    ](
        q, c_kv, do, lse, delta, partial_dc, sm_scale,
        H=H, S=S, D=D, NUM_GROUPS=num_groups,
        HEAD_GROUP=head_group, BLOCK_M=dc_block_m,
        BLOCK_N=block_n, D_CHUNK=dc_d_chunk, D_SLICE=dc_d_slice,
        L2_HEAD_CLUSTER=True,
        num_warps=8, num_stages=1,
    )
    n_dc = B * S * D
    _reduce_partial_dc_kernel[(triton.cdiv(n_dc, 256),)](
        partial_dc,
        dc_kv,
        n_dc,
        S=S,
        D=D,
        NUM_GROUPS=num_groups,
        BLOCK=256,
        num_warps=4,
        num_stages=1,
    )
    return dq, dc_kv
