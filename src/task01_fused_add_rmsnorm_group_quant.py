"""Cross-backend fused add, RMSNorm, and group-quantization kernel."""

import torch
import triton
import triton.language as tl


# Imported lazily only after the proven Ascend tag is selected. Other platform
# evaluators may load a Triton build that does not expose the TLE extension.
tle = None


def _backend_tag(t: torch.Tensor) -> str:
    parts = [str(t.device).lower(), str(getattr(t.device, "type", "")).lower()]
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and cuda.is_available():
        index = t.device.index
        if index is None:
            index = cuda.current_device()
        parts.append(str(cuda.get_device_name(index)).lower())
    version = getattr(torch, "version", None)
    hip_version = getattr(version, "hip", None)
    if hip_version is not None:
        parts.append("hip")
    c_mod = getattr(torch, "_C", None)
    get_private = getattr(c_mod, "_get_privateuse1_backend_name", None)
    if get_private is not None:
        parts.append(str(get_private()).lower())
    return " ".join(parts)


def _is_tianshu_tag(tag: str) -> bool:
    keys = ("tianshu", "tian", "iluvatar", "corex", "bi-v", "mr-v")
    return any(key in tag for key in keys)


def _is_ascend_tag(tag: str) -> bool:
    return any(key in tag for key in ("ascend", "npu", "910"))


def _is_hygon_tag(tag: str) -> bool:
    keys = ("hygon", "dcu", "hip", "rocm")
    excluded = ("metax", "maca")
    return any(key in tag for key in keys) and not any(
        key in tag for key in excluded
    )


def _is_metax_tag(tag: str) -> bool:
    return any(key in tag for key in ("metax", "maca"))


def _is_nvidia_tag(tag: str) -> bool:
    keys = (
        "nvidia", "geforce", "rtx", "tesla", "a100", "a800",
        "h100", "h800", "l20", "l40",
    )
    return any(key in tag for key in keys)


def _is_thead_tag(tag: str) -> bool:
    return any(key in tag for key in ("t-head", "thead", "zhenwu", "ppu"))


@triton.jit
def _preweighted_full_row_kernel(
    x,
    residual,
    gamma,
    residual_out,
    norm_out,
    x_q,
    x_scale,
    D: tl.constexpr,
    eps: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GROUPS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
    DIRECT_INT8: tl.constexpr,

):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    offsets = row * D + cols

    combined = tl.load(x + offsets, mask=mask, other=0.0).to(tl.float32)
    combined += tl.load(residual + offsets, mask=mask, other=0.0).to(tl.float32)
    sum_square = tl.sum(combined * combined, axis=0)
    tl.store(residual_out + offsets, combined, mask=mask)

    weight = tl.load(gamma + cols, mask=mask, other=0.0).to(tl.float32)
    weighted = combined * weight
    weighted_grouped = tl.reshape(
        tl.abs(weighted), (BLOCK_GROUPS, GROUP_SIZE)
    )
    weighted_max = tl.max(weighted_grouped, axis=1)

    inv_rms = tl.rsqrt(sum_square / D + eps)
    norm = weighted * inv_rms
    scales = tl.maximum(weighted_max * (inv_rms / 127.0), 1.0e-12)
    inverse_scales = 1.0 / scales
    inverse_expanded = tl.reshape(
        tl.broadcast_to(
            tl.reshape(inverse_scales, (BLOCK_GROUPS, 1)),
            (BLOCK_GROUPS, GROUP_SIZE),
        ),
        (BLOCK_D,),
    )
    if DIRECT_INT8:
        quantized = (norm * inverse_expanded).to(tl.int8)
    else:
        quantized = tl.clamp(
            norm * inverse_expanded, -127.0, 127.0
        ).to(tl.int32)

    tl.store(norm_out + offsets, norm, mask=mask)
    tl.store(x_q + offsets, quantized, mask=mask)
    group_ids = tl.arange(0, BLOCK_GROUPS)
    tl.store(x_scale + row * GROUPS + group_ids, scales, mask=group_ids < GROUPS)


@triton.jit
def _store_preweighted_segment(
    weighted,
    weighted_max,
    norm_out,
    x_q,
    x_scale,
    inv_rms,
    row,
    D: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GROUPS: tl.constexpr,
    SEGMENT: tl.constexpr,
    SEGMENT_GROUPS: tl.constexpr,
    SEGMENT_START: tl.constexpr,
    DIRECT_INT8: tl.constexpr = False,
):
    local_cols = tl.arange(0, SEGMENT)
    cols = SEGMENT_START + local_cols
    mask = cols < D
    offsets = row * D + cols

    norm = weighted * inv_rms
    scales = tl.maximum(weighted_max * (inv_rms / 127.0), 1.0e-12)
    inverse_scales = 1.0 / scales
    inverse_expanded = tl.reshape(
        tl.broadcast_to(
            tl.reshape(inverse_scales, (SEGMENT_GROUPS, 1)),
            (SEGMENT_GROUPS, GROUP_SIZE),
        ),
        (SEGMENT,),
    )
    if DIRECT_INT8:
        quantized = (norm * inverse_expanded).to(tl.int8)
    else:
        quantized = tl.clamp(
            norm * inverse_expanded, -127.0, 127.0
        ).to(tl.int32)

    tl.store(norm_out + offsets, norm, mask=mask)
    tl.store(x_q + offsets, quantized, mask=mask)
    local_groups = tl.arange(0, SEGMENT_GROUPS)
    group_ids = SEGMENT_START // GROUP_SIZE + local_groups
    tl.store(x_scale + row * GROUPS + group_ids, scales, mask=group_ids < GROUPS)


@triton.jit
def _preweighted_segmented_row_kernel(
    x,
    residual,
    gamma,
    residual_out,
    norm_out,
    x_q,
    x_scale,
    D: tl.constexpr,
    eps: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GROUPS: tl.constexpr,
    SEGMENT: tl.constexpr,
    SEGMENT_GROUPS: tl.constexpr,
    DIRECT_INT8: tl.constexpr,
):
    row = tl.program_id(0)
    local_cols = tl.arange(0, SEGMENT)

    cols0 = local_cols
    offsets0 = row * D + cols0
    combined0 = tl.load(x + offsets0).to(tl.float32)
    combined0 += tl.load(residual + offsets0).to(tl.float32)
    square_acc = combined0 * combined0
    tl.store(residual_out + offsets0, combined0)
    weight0 = tl.load(gamma + cols0).to(tl.float32)
    weighted0 = combined0 * weight0
    weighted_max0 = tl.max(
        tl.reshape(tl.abs(weighted0), (SEGMENT_GROUPS, GROUP_SIZE)), axis=1
    )

    cols1 = SEGMENT + local_cols
    mask1 = cols1 < D
    offsets1 = row * D + cols1
    combined1 = tl.load(x + offsets1, mask=mask1, other=0.0).to(tl.float32)
    combined1 += tl.load(residual + offsets1, mask=mask1, other=0.0).to(
        tl.float32
    )
    square_acc += combined1 * combined1
    tl.store(residual_out + offsets1, combined1, mask=mask1)
    weight1 = tl.load(gamma + cols1, mask=mask1, other=0.0).to(tl.float32)
    weighted1 = combined1 * weight1
    weighted_max1 = tl.max(
        tl.reshape(tl.abs(weighted1), (SEGMENT_GROUPS, GROUP_SIZE)), axis=1
    )

    if D > 2 * SEGMENT:
        cols2 = 2 * SEGMENT + local_cols
        mask2 = cols2 < D
        offsets2 = row * D + cols2
        combined2 = tl.load(x + offsets2, mask=mask2, other=0.0).to(tl.float32)
        combined2 += tl.load(
            residual + offsets2, mask=mask2, other=0.0
        ).to(tl.float32)
        square_acc += combined2 * combined2
        tl.store(residual_out + offsets2, combined2, mask=mask2)
        weight2 = tl.load(gamma + cols2, mask=mask2, other=0.0).to(tl.float32)
        weighted2 = combined2 * weight2
        weighted_max2 = tl.max(
            tl.reshape(tl.abs(weighted2), (SEGMENT_GROUPS, GROUP_SIZE)), axis=1
        )

    if D > 3 * SEGMENT:
        cols3 = 3 * SEGMENT + local_cols
        mask3 = cols3 < D
        offsets3 = row * D + cols3
        combined3 = tl.load(x + offsets3, mask=mask3, other=0.0).to(tl.float32)
        combined3 += tl.load(
            residual + offsets3, mask=mask3, other=0.0
        ).to(tl.float32)
        square_acc += combined3 * combined3
        tl.store(residual_out + offsets3, combined3, mask=mask3)
        weight3 = tl.load(gamma + cols3, mask=mask3, other=0.0).to(tl.float32)
        weighted3 = combined3 * weight3
        weighted_max3 = tl.max(
            tl.reshape(tl.abs(weighted3), (SEGMENT_GROUPS, GROUP_SIZE)), axis=1
        )

    square_group_sums = tl.sum(
        tl.reshape(square_acc, (SEGMENT_GROUPS, GROUP_SIZE)), axis=1
    )
    sum_square = tl.sum(square_group_sums, axis=0)
    inv_rms = tl.rsqrt(sum_square / D + eps)
    _store_preweighted_segment(
        weighted0,
        weighted_max0,
        norm_out,
        x_q,
        x_scale,
        inv_rms,
        row,
        D,
        GROUP_SIZE,
        GROUPS,
        SEGMENT,
        SEGMENT_GROUPS,
        0,
        DIRECT_INT8,
    )
    _store_preweighted_segment(
        weighted1,
        weighted_max1,
        norm_out,
        x_q,
        x_scale,
        inv_rms,
        row,
        D,
        GROUP_SIZE,
        GROUPS,
        SEGMENT,

        SEGMENT_GROUPS,
        SEGMENT,
        DIRECT_INT8,
    )
    if D > 2 * SEGMENT:
        _store_preweighted_segment(
            weighted2,
            weighted_max2,
            norm_out,
            x_q,
            x_scale,
            inv_rms,
            row,
            D,
            GROUP_SIZE,
            GROUPS,
            SEGMENT,
            SEGMENT_GROUPS,
            2 * SEGMENT,
            DIRECT_INT8,
        )
    if D > 3 * SEGMENT:
        _store_preweighted_segment(
            weighted3,
            weighted_max3,
            norm_out,
            x_q,
            x_scale,
            inv_rms,
            row,
            D,
            GROUP_SIZE,
            GROUPS,
            SEGMENT,
            SEGMENT_GROUPS,
            3 * SEGMENT,
            DIRECT_INT8,
        )


@triton.jit
def _ascend_subvec_r2_all_even_unitflag_only_kernel(
    x,
    residual,
    gamma,
    residual_out,
    norm_out,
    x_q,
    x_scale,
    D: tl.constexpr,
    eps: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GROUPS: tl.constexpr,
    SEGMENT: tl.constexpr,
    SEGMENT_GROUPS: tl.constexpr,
    DIRECT_INT8: tl.constexpr,
):
    """Accumulate square vectors first, then issue one RMS reduction.

    This follows the Ascend FlagGems reduction dependency graph while retaining
    the zero-reread preweighted output path used by this submission.
    """
    owner = tl.program_id(0)

    sid = tle.dsa.ascend.sub_vec_id()
    row = owner * 2 + sid
    local_cols = tl.arange(0, SEGMENT)

    cols0 = local_cols
    offsets0 = row * D + cols0
    combined0 = tl.load(x + offsets0).to(tl.float32)
    combined0 += tl.load(residual + offsets0).to(tl.float32)
    square_acc = combined0 * combined0
    tl.store(residual_out + offsets0, combined0)
    weight0 = tl.load(gamma + cols0).to(tl.float32)
    weighted0 = combined0 * weight0
    weighted_max0 = tl.max(
        tl.reshape(tl.abs(weighted0), (SEGMENT_GROUPS, GROUP_SIZE)), axis=1
    )
    weighted0 = weighted0.to(tl.bfloat16)

    cols1 = SEGMENT + local_cols
    mask1 = cols1 < D
    offsets1 = row * D + cols1
    combined1 = tl.load(x + offsets1, mask=mask1, other=0.0).to(tl.float32)
    combined1 += tl.load(residual + offsets1, mask=mask1, other=0.0).to(
        tl.float32
    )
    square_acc += combined1 * combined1
    tl.store(residual_out + offsets1, combined1, mask=mask1)
    weight1 = tl.load(gamma + cols1, mask=mask1, other=0.0).to(tl.float32)
    weighted1 = combined1 * weight1
    weighted_max1 = tl.max(
        tl.reshape(tl.abs(weighted1), (SEGMENT_GROUPS, GROUP_SIZE)), axis=1
    )
    weighted1 = weighted1.to(tl.bfloat16)

    if D > 2 * SEGMENT:
        cols2 = 2 * SEGMENT + local_cols
        mask2 = cols2 < D
        offsets2 = row * D + cols2
        combined2 = tl.load(x + offsets2, mask=mask2, other=0.0).to(tl.float32)
        combined2 += tl.load(
            residual + offsets2, mask=mask2, other=0.0
        ).to(tl.float32)
        square_acc += combined2 * combined2
        tl.store(residual_out + offsets2, combined2, mask=mask2)
        weight2 = tl.load(gamma + cols2, mask=mask2, other=0.0).to(tl.float32)
        weighted2 = combined2 * weight2
        weighted_max2 = tl.max(
            tl.reshape(tl.abs(weighted2), (SEGMENT_GROUPS, GROUP_SIZE)), axis=1
        )
        weighted2 = weighted2.to(tl.bfloat16)

    if D > 3 * SEGMENT:
        cols3 = 3 * SEGMENT + local_cols
        mask3 = cols3 < D
        offsets3 = row * D + cols3
        combined3 = tl.load(x + offsets3, mask=mask3, other=0.0).to(tl.float32)
        combined3 += tl.load(
            residual + offsets3, mask=mask3, other=0.0
        ).to(tl.float32)
        square_acc += combined3 * combined3

        tl.store(residual_out + offsets3, combined3, mask=mask3)
        weight3 = tl.load(gamma + cols3, mask=mask3, other=0.0).to(tl.float32)
        weighted3 = combined3 * weight3
        weighted_max3 = tl.max(
            tl.reshape(tl.abs(weighted3), (SEGMENT_GROUPS, GROUP_SIZE)), axis=1
        )
        weighted3 = weighted3.to(tl.bfloat16)

    square_group_sums = tl.sum(
        tl.reshape(square_acc, (SEGMENT_GROUPS, GROUP_SIZE)), axis=1
    )
    sum_square = tl.sum(square_group_sums, axis=0)
    inv_rms = tl.rsqrt(sum_square / D + eps)
    _store_preweighted_segment(
        weighted0,
        weighted_max0,
        norm_out,
        x_q,
        x_scale,
        inv_rms,
        row,
        D,
        GROUP_SIZE,
        GROUPS,
        SEGMENT,
        SEGMENT_GROUPS,
        0,
        DIRECT_INT8,
    )
    _store_preweighted_segment(
        weighted1,
        weighted_max1,
        norm_out,
        x_q,
        x_scale,
        inv_rms,
        row,
        D,
        GROUP_SIZE,
        GROUPS,
        SEGMENT,
        SEGMENT_GROUPS,
        SEGMENT,
        DIRECT_INT8,
    )
    if D > 2 * SEGMENT:
        _store_preweighted_segment(
            weighted2,
            weighted_max2,
            norm_out,
            x_q,
            x_scale,
            inv_rms,
            row,
            D,
            GROUP_SIZE,
            GROUPS,
            SEGMENT,
            SEGMENT_GROUPS,
            2 * SEGMENT,
            DIRECT_INT8,
        )
    if D > 3 * SEGMENT:
        _store_preweighted_segment(
            weighted3,
            weighted_max3,
            norm_out,
            x_q,
            x_scale,
            inv_rms,
            row,
            D,
            GROUP_SIZE,
            GROUPS,
            SEGMENT,
            SEGMENT_GROUPS,
            3 * SEGMENT,
            DIRECT_INT8,
        )


@triton.jit
def _ascend_vector_accum_segmented_row_kernel(
    x,
    residual,
    gamma,
    residual_out,
    norm_out,
    x_q,
    x_scale,
    D: tl.constexpr,
    eps: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GROUPS: tl.constexpr,
    SEGMENT: tl.constexpr,
    SEGMENT_GROUPS: tl.constexpr,
    DIRECT_INT8: tl.constexpr,
):
    """Accumulate square vectors first, then issue one RMS reduction.

    This follows the Ascend FlagGems reduction dependency graph while retaining
    the zero-reread preweighted output path used by this submission.
    """
    row = tl.program_id(0)
    local_cols = tl.arange(0, SEGMENT)

    cols0 = local_cols
    offsets0 = row * D + cols0
    combined0 = tl.load(x + offsets0).to(tl.float32)
    combined0 += tl.load(residual + offsets0).to(tl.float32)
    square_acc = combined0 * combined0
    tl.store(residual_out + offsets0, combined0)
    weight0 = tl.load(gamma + cols0).to(tl.float32)
    weighted0 = combined0 * weight0
    weighted_max0 = tl.max(
        tl.reshape(tl.abs(weighted0), (SEGMENT_GROUPS, GROUP_SIZE)), axis=1
    )
    weighted0 = weighted0.to(tl.bfloat16)

    cols1 = SEGMENT + local_cols
    mask1 = cols1 < D
    offsets1 = row * D + cols1
    combined1 = tl.load(x + offsets1, mask=mask1, other=0.0).to(tl.float32)
    combined1 += tl.load(residual + offsets1, mask=mask1, other=0.0).to(
        tl.float32
    )
    square_acc += combined1 * combined1
    tl.store(residual_out + offsets1, combined1, mask=mask1)
    weight1 = tl.load(gamma + cols1, mask=mask1, other=0.0).to(tl.float32)
    weighted1 = combined1 * weight1
    weighted_max1 = tl.max(
        tl.reshape(tl.abs(weighted1), (SEGMENT_GROUPS, GROUP_SIZE)), axis=1
    )
    weighted1 = weighted1.to(tl.bfloat16)

    if D > 2 * SEGMENT:
        cols2 = 2 * SEGMENT + local_cols
        mask2 = cols2 < D
        offsets2 = row * D + cols2
        combined2 = tl.load(x + offsets2, mask=mask2, other=0.0).to(tl.float32)
        combined2 += tl.load(
            residual + offsets2, mask=mask2, other=0.0
        ).to(tl.float32)
        square_acc += combined2 * combined2
        tl.store(residual_out + offsets2, combined2, mask=mask2)
        weight2 = tl.load(gamma + cols2, mask=mask2, other=0.0).to(tl.float32)
        weighted2 = combined2 * weight2
        weighted_max2 = tl.max(
            tl.reshape(tl.abs(weighted2), (SEGMENT_GROUPS, GROUP_SIZE)), axis=1
        )
        weighted2 = weighted2.to(tl.bfloat16)

    if D > 3 * SEGMENT:
        cols3 = 3 * SEGMENT + local_cols
        mask3 = cols3 < D
        offsets3 = row * D + cols3
        combined3 = tl.load(x + offsets3, mask=mask3, other=0.0).to(tl.float32)
        combined3 += tl.load(
            residual + offsets3, mask=mask3, other=0.0
        ).to(tl.float32)
        square_acc += combined3 * combined3

        tl.store(residual_out + offsets3, combined3, mask=mask3)
        weight3 = tl.load(gamma + cols3, mask=mask3, other=0.0).to(tl.float32)
        weighted3 = combined3 * weight3
        weighted_max3 = tl.max(
            tl.reshape(tl.abs(weighted3), (SEGMENT_GROUPS, GROUP_SIZE)), axis=1
        )
        weighted3 = weighted3.to(tl.bfloat16)

    square_group_sums = tl.sum(
        tl.reshape(square_acc, (SEGMENT_GROUPS, GROUP_SIZE)), axis=1
    )
    sum_square = tl.sum(square_group_sums, axis=0)
    inv_rms = tl.rsqrt(sum_square / D + eps)
    _store_preweighted_segment(
        weighted0,
        weighted_max0,
        norm_out,
        x_q,
        x_scale,
        inv_rms,
        row,
        D,
        GROUP_SIZE,
        GROUPS,
        SEGMENT,
        SEGMENT_GROUPS,
        0,
        DIRECT_INT8,
    )
    _store_preweighted_segment(
        weighted1,
        weighted_max1,
        norm_out,
        x_q,
        x_scale,
        inv_rms,
        row,
        D,
        GROUP_SIZE,
        GROUPS,
        SEGMENT,
        SEGMENT_GROUPS,
        SEGMENT,
        DIRECT_INT8,
    )
    if D > 2 * SEGMENT:
        _store_preweighted_segment(
            weighted2,
            weighted_max2,
            norm_out,
            x_q,
            x_scale,
            inv_rms,
            row,
            D,
            GROUP_SIZE,
            GROUPS,
            SEGMENT,
            SEGMENT_GROUPS,
            2 * SEGMENT,
            DIRECT_INT8,
        )
    if D > 3 * SEGMENT:
        _store_preweighted_segment(
            weighted3,
            weighted_max3,
            norm_out,
            x_q,
            x_scale,
            inv_rms,
            row,
            D,
            GROUP_SIZE,
            GROUPS,
            SEGMENT,
            SEGMENT_GROUPS,
            3 * SEGMENT,
            DIRECT_INT8,
        )


@triton.jit
def _tianshu_legacy_full_row_kernel(
    x,
    residual,
    gamma,
    residual_out,
    norm_out,
    x_q,
    x_scale,
    D: tl.constexpr,
    eps: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GROUPS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
    DIRECT_INT8: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    offsets = row * D + cols

    r = tl.load(x + offsets, mask=mask, other=0.0).to(tl.float32)
    r += tl.load(residual + offsets, mask=mask, other=0.0).to(tl.float32)
    mean_square = tl.sum(r * r, axis=0) / D
    inv_rms = tl.rsqrt(mean_square + eps)
    weight = tl.load(gamma + cols, mask=mask, other=0.0).to(tl.float32)
    y = r * inv_rms * weight

    y_grouped = tl.reshape(tl.abs(y), (BLOCK_GROUPS, GROUP_SIZE))
    scales = tl.maximum(tl.max(y_grouped, axis=1) / 127.0, 1.0e-12)
    scales_expanded = tl.reshape(
        tl.broadcast_to(
            tl.reshape(scales, (BLOCK_GROUPS, 1)),
            (BLOCK_GROUPS, GROUP_SIZE),
        ),
        (BLOCK_D,),
    )
    if DIRECT_INT8:
        q = (y / scales_expanded).to(tl.int8)
    else:
        q = tl.clamp(y / scales_expanded, -127.0, 127.0).to(tl.int32)


    tl.store(residual_out + offsets, r, mask=mask)
    tl.store(norm_out + offsets, y, mask=mask)
    tl.store(x_q + offsets, q, mask=mask)
    group_ids = tl.arange(0, BLOCK_GROUPS)
    tl.store(
        x_scale + row * GROUPS + group_ids,
        scales,
        mask=group_ids < GROUPS,
    )


@triton.jit
def _tianshu_wave64_hierarchical_max_kernel(
    x,
    residual,
    gamma,
    residual_out,
    norm_out,
    x_q,
    x_scale,
    D: tl.constexpr,
    eps: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GROUPS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
    WAVES_PER_GROUP: tl.constexpr,
):
    """Keep one-pass row ownership while exposing wave64-native reductions."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    offsets = row * D + cols

    r = tl.load(x + offsets, mask=mask, other=0.0).to(tl.float32)
    r += tl.load(residual + offsets, mask=mask, other=0.0).to(tl.float32)
    mean_square = tl.sum(r * r, axis=0) / D
    inv_rms = tl.rsqrt(mean_square + eps)
    weight = tl.load(gamma + cols, mask=mask, other=0.0).to(tl.float32)
    y = r * inv_rms * weight

    wave_values = tl.reshape(
        tl.abs(y),
        (BLOCK_GROUPS * WAVES_PER_GROUP, 64),
    )
    wave_maxima = tl.max(wave_values, axis=1)
    group_wave_maxima = tl.reshape(
        wave_maxima,
        (BLOCK_GROUPS, WAVES_PER_GROUP),
    )
    scales = tl.maximum(
        tl.max(group_wave_maxima, axis=1) / 127.0,
        1.0e-12,
    )
    scales_expanded = tl.reshape(
        tl.broadcast_to(
            tl.reshape(scales, (BLOCK_GROUPS, 1)),
            (BLOCK_GROUPS, GROUP_SIZE),
        ),
        (BLOCK_D,),
    )
    q = (y / scales_expanded).to(tl.int8)

    tl.store(residual_out + offsets, r, mask=mask)
    tl.store(norm_out + offsets, y, mask=mask)
    tl.store(x_q + offsets, q, mask=mask)
    group_ids = tl.arange(0, BLOCK_GROUPS)
    tl.store(
        x_scale + row * GROUPS + group_ids,
        scales,
        mask=group_ids < GROUPS,
    )


@triton.jit
def _tianshu_legacy_store_quant_segment(
    r, gamma, residual_out, norm_out, x_q, x_scale, inv_rms, row,
    D: tl.constexpr, GROUP_SIZE: tl.constexpr, GROUPS: tl.constexpr,
    SEGMENT: tl.constexpr, SEGMENT_GROUPS: tl.constexpr,
    SEGMENT_START: tl.constexpr,
    DIRECT_INT8: tl.constexpr = False,
):
    local_cols = tl.arange(0, SEGMENT)
    cols = SEGMENT_START + local_cols
    mask = cols < D
    offsets = row * D + cols
    weight = tl.load(gamma + cols, mask=mask, other=0.0).to(tl.float32)
    y = r * inv_rms * weight
    y_grouped = tl.reshape(tl.abs(y), (SEGMENT_GROUPS, GROUP_SIZE))
    scales = tl.maximum(tl.max(y_grouped, axis=1) / 127.0, 1.0e-12)
    scales_expanded = tl.reshape(
        tl.broadcast_to(
            tl.reshape(scales, (SEGMENT_GROUPS, 1)),
            (SEGMENT_GROUPS, GROUP_SIZE),
        ),
        (SEGMENT,),
    )
    if DIRECT_INT8:
        q = (y / scales_expanded).to(tl.int8)

    else:
        q = tl.clamp(y / scales_expanded, -127.0, 127.0).to(tl.int32)
    tl.store(residual_out + offsets, r, mask=mask)
    tl.store(norm_out + offsets, y, mask=mask)
    tl.store(x_q + offsets, q, mask=mask)
    local_groups = tl.arange(0, SEGMENT_GROUPS)
    group_ids = SEGMENT_START // GROUP_SIZE + local_groups
    tl.store(
        x_scale + row * GROUPS + group_ids,
        scales,
        mask=group_ids < GROUPS,
    )


@triton.jit
def _tianshu_legacy_segmented_kernel(
    x, residual, gamma, residual_out, norm_out, x_q, x_scale,
    D: tl.constexpr, eps: tl.constexpr, GROUP_SIZE: tl.constexpr,
    GROUPS: tl.constexpr, SEGMENT: tl.constexpr,
    SEGMENT_GROUPS: tl.constexpr,
    DIRECT_INT8: tl.constexpr,
):
    row = tl.program_id(0)
    local_cols = tl.arange(0, SEGMENT)
    offsets0 = row * D + local_cols
    r0 = tl.load(x + offsets0).to(tl.float32)
    r0 += tl.load(residual + offsets0).to(tl.float32)
    sum_square = tl.sum(r0 * r0, axis=0)

    cols1 = SEGMENT + local_cols
    mask1 = cols1 < D
    offsets1 = row * D + cols1
    r1 = tl.load(x + offsets1, mask=mask1, other=0.0).to(tl.float32)
    r1 += tl.load(residual + offsets1, mask=mask1, other=0.0).to(tl.float32)
    sum_square += tl.sum(r1 * r1, axis=0)

    if D > 2 * SEGMENT:
        cols2 = 2 * SEGMENT + local_cols
        mask2 = cols2 < D
        offsets2 = row * D + cols2
        r2 = tl.load(x + offsets2, mask=mask2, other=0.0).to(tl.float32)
        r2 += tl.load(residual + offsets2, mask=mask2, other=0.0).to(tl.float32)
        sum_square += tl.sum(r2 * r2, axis=0)

    if D > 3 * SEGMENT:
        cols3 = 3 * SEGMENT + local_cols
        mask3 = cols3 < D
        offsets3 = row * D + cols3
        r3 = tl.load(x + offsets3, mask=mask3, other=0.0).to(tl.float32)
        r3 += tl.load(residual + offsets3, mask=mask3, other=0.0).to(tl.float32)
        sum_square += tl.sum(r3 * r3, axis=0)

    inv_rms = tl.rsqrt(sum_square / D + eps)
    _tianshu_legacy_store_quant_segment(
        r0, gamma, residual_out, norm_out, x_q, x_scale, inv_rms, row,
        D, GROUP_SIZE, GROUPS, SEGMENT, SEGMENT_GROUPS, 0, DIRECT_INT8,
    )
    _tianshu_legacy_store_quant_segment(
        r1, gamma, residual_out, norm_out, x_q, x_scale, inv_rms, row,
        D, GROUP_SIZE, GROUPS, SEGMENT, SEGMENT_GROUPS, SEGMENT, DIRECT_INT8,
    )
    if D > 2 * SEGMENT:
        _tianshu_legacy_store_quant_segment(
            r2, gamma, residual_out, norm_out, x_q, x_scale, inv_rms, row,
            D, GROUP_SIZE, GROUPS, SEGMENT, SEGMENT_GROUPS, 2 * SEGMENT,
            DIRECT_INT8,
        )
    if D > 3 * SEGMENT:
        _tianshu_legacy_store_quant_segment(
            r3, gamma, residual_out, norm_out, x_q, x_scale, inv_rms, row,
            D, GROUP_SIZE, GROUPS, SEGMENT, SEGMENT_GROUPS, 3 * SEGMENT,
            DIRECT_INT8,
        )


@triton.jit
def _store_nvidia_row_group_segment(
    weighted,
    weighted_max,
    norm_out,
    x_q,
    x_scale,
    inv_rms,
    rows,
    row_mask,
    D: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GROUPS: tl.constexpr,
    ROW_TILE: tl.constexpr,
    SEGMENT: tl.constexpr,
    SEGMENT_GROUPS: tl.constexpr,
    SEGMENT_START: tl.constexpr,
):
    local_cols = tl.arange(0, SEGMENT)
    cols = SEGMENT_START + local_cols
    col_mask = cols < D
    mask = row_mask[:, None] & col_mask[None, :]
    offsets = rows[:, None] * D + cols[None, :]

    norm = weighted * inv_rms[:, None]
    scales = tl.maximum(
        weighted_max * (inv_rms[:, None] / 127.0), 1.0e-12
    )
    inverse_scales = 1.0 / scales
    inverse_expanded = tl.reshape(
        tl.broadcast_to(
            tl.reshape(inverse_scales, (ROW_TILE, SEGMENT_GROUPS, 1)),
            (ROW_TILE, SEGMENT_GROUPS, GROUP_SIZE),
        ),
        (ROW_TILE, SEGMENT),
    )
    quantized = tl.clamp(
        norm * inverse_expanded, -127.0, 127.0
    ).to(tl.int32)

    tl.store(norm_out + offsets, norm, mask=mask)
    tl.store(x_q + offsets, quantized, mask=mask)
    local_groups = tl.arange(0, SEGMENT_GROUPS)
    group_ids = SEGMENT_START // GROUP_SIZE + local_groups
    scale_offsets = rows[:, None] * GROUPS + group_ids[None, :]

    scale_mask = row_mask[:, None] & (group_ids[None, :] < GROUPS)
    tl.store(x_scale + scale_offsets, scales, mask=scale_mask)


@triton.jit
def _nvidia_parallel_row_group_kernel(
    x,
    residual,
    gamma,
    residual_out,
    norm_out,
    x_q,
    x_scale,
    M: tl.constexpr,
    D: tl.constexpr,
    eps: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GROUPS: tl.constexpr,
    ROW_TILE: tl.constexpr,
    SEGMENT: tl.constexpr,
    SEGMENT_GROUPS: tl.constexpr,
):
    # The row axis is explicit and disjoint: program p owns
    # [p * ROW_TILE, (p + 1) * ROW_TILE).  Gamma has no row axis, so each
    # segment is loaded once and broadcast across the complete row group.
    row_base = tl.program_id(0) * ROW_TILE
    rows = row_base + tl.arange(0, ROW_TILE)
    row_mask = rows < M
    local_cols = tl.arange(0, SEGMENT)

    cols0 = local_cols
    mask0 = row_mask[:, None]
    offsets0 = rows[:, None] * D + cols0[None, :]
    combined0 = tl.load(x + offsets0, mask=mask0, other=0.0).to(tl.float32)
    combined0 += tl.load(
        residual + offsets0, mask=mask0, other=0.0
    ).to(tl.float32)
    sum_square = tl.sum(combined0 * combined0, axis=1)
    tl.store(residual_out + offsets0, combined0, mask=mask0)
    weight0 = tl.load(gamma + cols0).to(tl.float32)
    weighted0 = combined0 * weight0[None, :]
    weighted_max0 = tl.max(
        tl.reshape(
            tl.abs(weighted0),
            (ROW_TILE, SEGMENT_GROUPS, GROUP_SIZE),
        ),
        axis=2,
    )

    cols1 = SEGMENT + local_cols
    col_mask1 = cols1 < D
    mask1 = row_mask[:, None] & col_mask1[None, :]
    offsets1 = rows[:, None] * D + cols1[None, :]
    combined1 = tl.load(x + offsets1, mask=mask1, other=0.0).to(tl.float32)
    combined1 += tl.load(
        residual + offsets1, mask=mask1, other=0.0
    ).to(tl.float32)
    sum_square += tl.sum(combined1 * combined1, axis=1)
    tl.store(residual_out + offsets1, combined1, mask=mask1)
    weight1 = tl.load(
        gamma + cols1, mask=col_mask1, other=0.0
    ).to(tl.float32)
    weighted1 = combined1 * weight1[None, :]
    weighted_max1 = tl.max(
        tl.reshape(
            tl.abs(weighted1),
            (ROW_TILE, SEGMENT_GROUPS, GROUP_SIZE),
        ),
        axis=2,
    )

    if D > 2 * SEGMENT:
        cols2 = 2 * SEGMENT + local_cols
        col_mask2 = cols2 < D
        mask2 = row_mask[:, None] & col_mask2[None, :]
        offsets2 = rows[:, None] * D + cols2[None, :]
        combined2 = tl.load(
            x + offsets2, mask=mask2, other=0.0
        ).to(tl.float32)
        combined2 += tl.load(
            residual + offsets2, mask=mask2, other=0.0
        ).to(tl.float32)
        sum_square += tl.sum(combined2 * combined2, axis=1)
        tl.store(residual_out + offsets2, combined2, mask=mask2)
        weight2 = tl.load(
            gamma + cols2, mask=col_mask2, other=0.0
        ).to(tl.float32)
        weighted2 = combined2 * weight2[None, :]
        weighted_max2 = tl.max(
            tl.reshape(
                tl.abs(weighted2),
                (ROW_TILE, SEGMENT_GROUPS, GROUP_SIZE),
            ),
            axis=2,
        )

    inv_rms = tl.rsqrt(sum_square / D + eps)
    _store_nvidia_row_group_segment(
        weighted0,
        weighted_max0,
        norm_out,
        x_q,
        x_scale,
        inv_rms,
        rows,
        row_mask,
        D,
        GROUP_SIZE,
        GROUPS,
        ROW_TILE,
        SEGMENT,
        SEGMENT_GROUPS,
        0,
    )
    _store_nvidia_row_group_segment(
        weighted1,
        weighted_max1,
        norm_out,
        x_q,
        x_scale,

        inv_rms,
        rows,
        row_mask,
        D,
        GROUP_SIZE,
        GROUPS,
        ROW_TILE,
        SEGMENT,
        SEGMENT_GROUPS,
        SEGMENT,
    )
    if D > 2 * SEGMENT:
        _store_nvidia_row_group_segment(
            weighted2,
            weighted_max2,
            norm_out,
            x_q,
            x_scale,
            inv_rms,
            rows,
            row_mask,
            D,
            GROUP_SIZE,
            GROUPS,
            ROW_TILE,
            SEGMENT,
            SEGMENT_GROUPS,
            2 * SEGMENT,
        )

@triton.jit
def _store_streaming_segment(
    weighted,
    weighted_max,
    norm_out,
    x_q,
    x_scale,
    inv_rms,
    row,
    D: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GROUPS: tl.constexpr,
    SEGMENT: tl.constexpr,
    SEGMENT_GROUPS: tl.constexpr,
    SEGMENT_START: tl.constexpr,
    DIRECT_INT8: tl.constexpr = False,
    STREAMING: tl.constexpr = False,
):
    local_cols = tl.arange(0, SEGMENT)
    cols = SEGMENT_START + local_cols
    mask = cols < D
    offsets = row * D + cols

    norm = weighted * inv_rms
    scales = tl.maximum(weighted_max * (inv_rms / 127.0), 1.0e-12)
    inverse_scales = 1.0 / scales
    inverse_expanded = tl.reshape(
        tl.broadcast_to(
            tl.reshape(inverse_scales, (SEGMENT_GROUPS, 1)),
            (SEGMENT_GROUPS, GROUP_SIZE),
        ),
        (SEGMENT,),
    )
    if DIRECT_INT8:
        quantized = (norm * inverse_expanded).to(tl.int8)
    else:
        quantized = tl.clamp(
            norm * inverse_expanded, -127.0, 127.0
        ).to(tl.int32)

    local_groups = tl.arange(0, SEGMENT_GROUPS)
    group_ids = SEGMENT_START // GROUP_SIZE + local_groups
    if STREAMING:
        tl.store(norm_out + offsets, norm, mask=mask, cache_modifier=".cs")
        tl.store(x_q + offsets, quantized, mask=mask, cache_modifier=".cs")
        tl.store(
            x_scale + row * GROUPS + group_ids,
            scales,
            mask=group_ids < GROUPS,
            cache_modifier=".cs",
        )
    else:
        tl.store(norm_out + offsets, norm, mask=mask)
        tl.store(x_q + offsets, quantized, mask=mask)
        tl.store(
            x_scale + row * GROUPS + group_ids,
            scales,
            mask=group_ids < GROUPS,
        )


@triton.jit
def _preweighted_segmented_streaming_kernel(
    x,
    residual,
    gamma,
    residual_out,
    norm_out,
    x_q,
    x_scale,
    D: tl.constexpr,
    eps: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GROUPS: tl.constexpr,
    SEGMENT: tl.constexpr,
    SEGMENT_GROUPS: tl.constexpr,
    DIRECT_INT8: tl.constexpr,
    STREAMING: tl.constexpr,
):
    row = tl.program_id(0)
    local_cols = tl.arange(0, SEGMENT)

    cols0 = local_cols
    offsets0 = row * D + cols0
    if STREAMING:
        combined0 = tl.load(x + offsets0, cache_modifier=".cg").to(tl.float32)
        combined0 += tl.load(
            residual + offsets0, cache_modifier=".cg"
        ).to(tl.float32)
    else:
        combined0 = tl.load(x + offsets0).to(tl.float32)
        combined0 += tl.load(residual + offsets0).to(tl.float32)
    square_acc = combined0 * combined0
    if STREAMING:
        tl.store(residual_out + offsets0, combined0, cache_modifier=".cs")
    else:
        tl.store(residual_out + offsets0, combined0)
    weight0 = tl.load(gamma + cols0).to(tl.float32)
    weighted0 = combined0 * weight0
    weighted_max0 = tl.max(
        tl.reshape(tl.abs(weighted0), (SEGMENT_GROUPS, GROUP_SIZE)), axis=1
    )

    cols1 = SEGMENT + local_cols
    mask1 = cols1 < D
    offsets1 = row * D + cols1
    if STREAMING:
        combined1 = tl.load(
            x + offsets1, mask=mask1, other=0.0, cache_modifier=".cg"
        ).to(tl.float32)
        combined1 += tl.load(
            residual + offsets1,
            mask=mask1,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
    else:
        combined1 = tl.load(x + offsets1, mask=mask1, other=0.0).to(tl.float32)
        combined1 += tl.load(residual + offsets1, mask=mask1, other=0.0).to(
            tl.float32
        )
    square_acc += combined1 * combined1
    if STREAMING:
        tl.store(
            residual_out + offsets1,
            combined1,
            mask=mask1,
            cache_modifier=".cs",
        )
    else:
        tl.store(residual_out + offsets1, combined1, mask=mask1)
    weight1 = tl.load(gamma + cols1, mask=mask1, other=0.0).to(tl.float32)
    weighted1 = combined1 * weight1
    weighted_max1 = tl.max(
        tl.reshape(tl.abs(weighted1), (SEGMENT_GROUPS, GROUP_SIZE)), axis=1
    )

    if D > 2 * SEGMENT:
        cols2 = 2 * SEGMENT + local_cols
        mask2 = cols2 < D
        offsets2 = row * D + cols2
        if STREAMING:
            combined2 = tl.load(
                x + offsets2, mask=mask2, other=0.0, cache_modifier=".cg"
            ).to(tl.float32)
            combined2 += tl.load(
                residual + offsets2,
                mask=mask2,
                other=0.0,
                cache_modifier=".cg",
            ).to(tl.float32)
        else:
            combined2 = tl.load(x + offsets2, mask=mask2, other=0.0).to(tl.float32)
            combined2 += tl.load(
                residual + offsets2, mask=mask2, other=0.0
            ).to(tl.float32)
        square_acc += combined2 * combined2
        if STREAMING:
            tl.store(
                residual_out + offsets2,
                combined2,
                mask=mask2,
                cache_modifier=".cs",
            )
        else:
            tl.store(residual_out + offsets2, combined2, mask=mask2)
        weight2 = tl.load(gamma + cols2, mask=mask2, other=0.0).to(tl.float32)
        weighted2 = combined2 * weight2
        weighted_max2 = tl.max(

            tl.reshape(tl.abs(weighted2), (SEGMENT_GROUPS, GROUP_SIZE)), axis=1
        )

    if D > 3 * SEGMENT:
        cols3 = 3 * SEGMENT + local_cols
        mask3 = cols3 < D
        offsets3 = row * D + cols3
        if STREAMING:
            combined3 = tl.load(
                x + offsets3, mask=mask3, other=0.0, cache_modifier=".cg"
            ).to(tl.float32)
            combined3 += tl.load(
                residual + offsets3,
                mask=mask3,
                other=0.0,
                cache_modifier=".cg",
            ).to(tl.float32)
        else:
            combined3 = tl.load(x + offsets3, mask=mask3, other=0.0).to(tl.float32)
            combined3 += tl.load(
                residual + offsets3, mask=mask3, other=0.0
            ).to(tl.float32)
        square_acc += combined3 * combined3
        if STREAMING:
            tl.store(
                residual_out + offsets3,
                combined3,
                mask=mask3,
                cache_modifier=".cs",
            )
        else:
            tl.store(residual_out + offsets3, combined3, mask=mask3)
        weight3 = tl.load(gamma + cols3, mask=mask3, other=0.0).to(tl.float32)
        weighted3 = combined3 * weight3
        weighted_max3 = tl.max(
            tl.reshape(tl.abs(weighted3), (SEGMENT_GROUPS, GROUP_SIZE)), axis=1
        )

    square_group_sums = tl.sum(
        tl.reshape(square_acc, (SEGMENT_GROUPS, GROUP_SIZE)), axis=1
    )
    sum_square = tl.sum(square_group_sums, axis=0)
    inv_rms = tl.rsqrt(sum_square / D + eps)
    _store_streaming_segment(
        weighted0,
        weighted_max0,
        norm_out,
        x_q,
        x_scale,
        inv_rms,
        row,
        D,
        GROUP_SIZE,
        GROUPS,
        SEGMENT,
        SEGMENT_GROUPS,
        0,
        DIRECT_INT8,
        STREAMING,
    )
    _store_streaming_segment(
        weighted1,
        weighted_max1,
        norm_out,
        x_q,
        x_scale,
        inv_rms,
        row,
        D,
        GROUP_SIZE,
        GROUPS,
        SEGMENT,
        SEGMENT_GROUPS,
        SEGMENT,
        DIRECT_INT8,
        STREAMING,
    )
    if D > 2 * SEGMENT:
        _store_streaming_segment(
            weighted2,
            weighted_max2,
            norm_out,
            x_q,
            x_scale,
            inv_rms,
            row,
            D,
            GROUP_SIZE,
            GROUPS,
            SEGMENT,
            SEGMENT_GROUPS,
            2 * SEGMENT,
            DIRECT_INT8,
            STREAMING,
        )
    if D > 3 * SEGMENT:
        _store_streaming_segment(
            weighted3,
            weighted_max3,
            norm_out,
            x_q,
            x_scale,
            inv_rms,
            row,
            D,
            GROUP_SIZE,
            GROUPS,
            SEGMENT,
            SEGMENT_GROUPS,
            3 * SEGMENT,
            DIRECT_INT8,
            STREAMING,
        )


@triton.jit
def _store_nvidia_row_group_streaming_segment(
    weighted,
    weighted_max,
    norm_out,
    x_q,
    x_scale,
    inv_rms,
    rows,
    row_mask,
    D: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GROUPS: tl.constexpr,
    ROW_TILE: tl.constexpr,
    SEGMENT: tl.constexpr,
    SEGMENT_GROUPS: tl.constexpr,
    SEGMENT_START: tl.constexpr,
):
    local_cols = tl.arange(0, SEGMENT)
    cols = SEGMENT_START + local_cols
    col_mask = cols < D
    mask = row_mask[:, None] & col_mask[None, :]
    offsets = rows[:, None] * D + cols[None, :]

    norm = weighted * inv_rms[:, None]
    scales = tl.maximum(
        weighted_max * (inv_rms[:, None] / 127.0), 1.0e-12
    )
    inverse_scales = 1.0 / scales
    inverse_expanded = tl.reshape(
        tl.broadcast_to(
            tl.reshape(inverse_scales, (ROW_TILE, SEGMENT_GROUPS, 1)),
            (ROW_TILE, SEGMENT_GROUPS, GROUP_SIZE),
        ),
        (ROW_TILE, SEGMENT),
    )
    quantized = tl.clamp(
        norm * inverse_expanded, -127.0, 127.0
    ).to(tl.int32)

    tl.store(norm_out + offsets, norm, mask=mask, cache_modifier=".cs")
    tl.store(x_q + offsets, quantized, mask=mask, cache_modifier=".cs")
    local_groups = tl.arange(0, SEGMENT_GROUPS)
    group_ids = SEGMENT_START // GROUP_SIZE + local_groups
    scale_offsets = rows[:, None] * GROUPS + group_ids[None, :]

    scale_mask = row_mask[:, None] & (group_ids[None, :] < GROUPS)
    tl.store(x_scale + scale_offsets, scales, mask=scale_mask)


@triton.jit
def _nvidia_parallel_row_group_streaming_kernel(
    x,
    residual,
    gamma,
    residual_out,
    norm_out,
    x_q,
    x_scale,
    M: tl.constexpr,
    D: tl.constexpr,
    eps: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GROUPS: tl.constexpr,
    ROW_TILE: tl.constexpr,
    SEGMENT: tl.constexpr,
    SEGMENT_GROUPS: tl.constexpr,
):
    # The row axis is explicit and disjoint: program p owns
    # [p * ROW_TILE, (p + 1) * ROW_TILE).  Gamma has no row axis, so each
    # segment is loaded once and broadcast across the complete row group.
    row_base = tl.program_id(0) * ROW_TILE
    rows = row_base + tl.arange(0, ROW_TILE)
    row_mask = rows < M
    local_cols = tl.arange(0, SEGMENT)

    cols0 = local_cols
    mask0 = row_mask[:, None]
    offsets0 = rows[:, None] * D + cols0[None, :]
    combined0 = tl.load(x + offsets0, mask=mask0, other=0.0, cache_modifier=".cg").to(tl.float32)
    combined0 += tl.load(
        residual + offsets0, mask=mask0, other=0.0, cache_modifier=".cg"
    ).to(tl.float32)
    sum_square = tl.sum(combined0 * combined0, axis=1)
    tl.store(residual_out + offsets0, combined0, mask=mask0, cache_modifier=".cs")
    weight0 = tl.load(gamma + cols0).to(tl.float32)
    weighted0 = combined0 * weight0[None, :]
    weighted_max0 = tl.max(
        tl.reshape(
            tl.abs(weighted0),
            (ROW_TILE, SEGMENT_GROUPS, GROUP_SIZE),
        ),
        axis=2,
    )

    cols1 = SEGMENT + local_cols
    col_mask1 = cols1 < D
    mask1 = row_mask[:, None] & col_mask1[None, :]
    offsets1 = rows[:, None] * D + cols1[None, :]
    combined1 = tl.load(x + offsets1, mask=mask1, other=0.0, cache_modifier=".cg").to(tl.float32)
    combined1 += tl.load(
        residual + offsets1, mask=mask1, other=0.0, cache_modifier=".cg"
    ).to(tl.float32)
    sum_square += tl.sum(combined1 * combined1, axis=1)
    tl.store(residual_out + offsets1, combined1, mask=mask1, cache_modifier=".cs")
    weight1 = tl.load(
        gamma + cols1, mask=col_mask1, other=0.0
    ).to(tl.float32)
    weighted1 = combined1 * weight1[None, :]
    weighted_max1 = tl.max(
        tl.reshape(
            tl.abs(weighted1),
            (ROW_TILE, SEGMENT_GROUPS, GROUP_SIZE),
        ),
        axis=2,
    )

    if D > 2 * SEGMENT:
        cols2 = 2 * SEGMENT + local_cols
        col_mask2 = cols2 < D
        mask2 = row_mask[:, None] & col_mask2[None, :]
        offsets2 = rows[:, None] * D + cols2[None, :]
        combined2 = tl.load(
            x + offsets2, mask=mask2, other=0.0, cache_modifier=".cg"
        ).to(tl.float32)
        combined2 += tl.load(
            residual + offsets2, mask=mask2, other=0.0, cache_modifier=".cg"
        ).to(tl.float32)
        sum_square += tl.sum(combined2 * combined2, axis=1)
        tl.store(residual_out + offsets2, combined2, mask=mask2, cache_modifier=".cs")
        weight2 = tl.load(
            gamma + cols2, mask=col_mask2, other=0.0
        ).to(tl.float32)
        weighted2 = combined2 * weight2[None, :]
        weighted_max2 = tl.max(
            tl.reshape(
                tl.abs(weighted2),
                (ROW_TILE, SEGMENT_GROUPS, GROUP_SIZE),
            ),
            axis=2,
        )

    inv_rms = tl.rsqrt(sum_square / D + eps)
    _store_nvidia_row_group_streaming_segment(
        weighted0,
        weighted_max0,
        norm_out,
        x_q,
        x_scale,
        inv_rms,
        rows,
        row_mask,
        D,
        GROUP_SIZE,
        GROUPS,
        ROW_TILE,
        SEGMENT,
        SEGMENT_GROUPS,
        0,
    )
    _store_nvidia_row_group_streaming_segment(
        weighted1,
        weighted_max1,
        norm_out,
        x_q,
        x_scale,

        inv_rms,
        rows,
        row_mask,
        D,
        GROUP_SIZE,
        GROUPS,
        ROW_TILE,
        SEGMENT,
        SEGMENT_GROUPS,
        SEGMENT,
    )
    if D > 2 * SEGMENT:
        _store_nvidia_row_group_streaming_segment(
            weighted2,
            weighted_max2,
            norm_out,
            x_q,
            x_scale,
            inv_rms,
            rows,
            row_mask,
            D,
            GROUP_SIZE,
            GROUPS,
            ROW_TILE,
            SEGMENT,
            SEGMENT_GROUPS,
            2 * SEGMENT,
        )

def _launch_task01_stable(
    x: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    residual_out: torch.Tensor,
    norm_out: torch.Tensor,
    x_q: torch.Tensor,
    x_scale: torch.Tensor,
    eps: float,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    M, D = x.shape
    groups = D // group_size
    tag = _backend_tag(x)
    is_ascend = _is_ascend_tag(tag)
    is_hygon = _is_hygon_tag(tag)
    is_metax = _is_metax_tag(tag)
    is_nvidia = _is_nvidia_tag(tag)
    is_thead = _is_thead_tag(tag)
    is_tianshu = _is_tianshu_tag(tag)

    if is_tianshu and M >= 128:
        block_d = triton.next_power_of_2(D)
        if D >= 4096:
            _tianshu_legacy_full_row_kernel[(M,)](
                x,
                residual,
                gamma,
                residual_out,
                norm_out,
                x_q,
                x_scale,
                D=D,
                eps=eps,
                GROUP_SIZE=group_size,
                GROUPS=groups,
                BLOCK_D=block_d,
                BLOCK_GROUPS=block_d // group_size,
                DIRECT_INT8=True,
                num_warps=8,
                num_stages=1 if D == 4096 or D == 7168 else 2,
            )
        else:
            segment = 4096
            _tianshu_legacy_segmented_kernel[(M,)](
                x,
                residual,
                gamma,
                residual_out,
                norm_out,
                x_q,
                x_scale,
                D=D,
                eps=eps,
                GROUP_SIZE=group_size,
                GROUPS=groups,
                SEGMENT=segment,
                SEGMENT_GROUPS=segment // group_size,
                DIRECT_INT8=True,
                num_warps=8,
                num_stages=1 if D == 7168 else 2,
            )
    elif is_ascend and M >= 128:
        # One final RMS reduction with 4096-wide segments limits helper count.
        segment = 4096
        _ascend_vector_accum_segmented_row_kernel[(M,)](
            x,
            residual,
            gamma,
            residual_out,
            norm_out,
            x_q,
            x_scale,
            D=D,
            eps=eps,
            GROUP_SIZE=group_size,
            GROUPS=groups,
            SEGMENT=segment,
            SEGMENT_GROUPS=segment // group_size,
            DIRECT_INT8=True,
            num_warps=8,
            num_stages=1,
        )
    elif is_nvidia and M >= 512 and D <= 6144:
        row_tile = 2
        segment = 2048
        _nvidia_parallel_row_group_kernel[
            (triton.cdiv(M, row_tile),)

        ](
            x,
            residual,
            gamma,
            residual_out,
            norm_out,
            x_q,
            x_scale,
            M=M,
            D=D,
            eps=eps,
            GROUP_SIZE=group_size,
            GROUPS=groups,
            ROW_TILE=row_tile,
            SEGMENT=segment,
            SEGMENT_GROUPS=segment // group_size,
            num_warps=8,
            num_stages=1,
        )
    elif M == 128 and D == 4096 and (
        is_metax or is_nvidia or is_thead
    ):
        _preweighted_full_row_kernel[(M,)](
            x,
            residual,
            gamma,
            residual_out,
            norm_out,
            x_q,
            x_scale,
            D=D,
            eps=eps,
            GROUP_SIZE=group_size,
            GROUPS=groups,
            BLOCK_D=4096,
            BLOCK_GROUPS=4096 // group_size,
            DIRECT_INT8=is_thead,
            num_warps=4 if is_metax else 8,
            num_stages=1,
        )
    elif M >= 128:
        segment = 4096 if (
            is_metax and (D == 7168 or D == 8192)
        ) else 2048
        target_warps = 4 if is_hygon or is_metax else 8
        target_stages = 1
        _preweighted_segmented_row_kernel[(M,)](
            x,
            residual,
            gamma,
            residual_out,
            norm_out,
            x_q,
            x_scale,
            D=D,
            eps=eps,
            GROUP_SIZE=group_size,
            GROUPS=groups,
            SEGMENT=segment,
            SEGMENT_GROUPS=segment // group_size,
            DIRECT_INT8=is_thead,
            num_warps=target_warps,
            num_stages=target_stages,
        )
    else:
        block_d = triton.next_power_of_2(D)
        target_warps = 4 if M == 1 or is_hygon or is_metax else 8
        target_stages = 1
        kernel = (
            _tianshu_legacy_full_row_kernel
            if is_tianshu
            else _preweighted_full_row_kernel
        )
        kernel[(M,)](
            x,
            residual,
            gamma,
            residual_out,
            norm_out,
            x_q,
            x_scale,
            D=D,
            eps=eps,
            GROUP_SIZE=group_size,
            GROUPS=groups,
            BLOCK_D=block_d,
            BLOCK_GROUPS=block_d // group_size,
            DIRECT_INT8=is_ascend or is_thead or is_tianshu,
            num_warps=target_warps,
            num_stages=target_stages,
        )
    return residual_out, norm_out, x_q, x_scale


def fused_add_rmsnorm_group_quant(
    x: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    residual_out: torch.Tensor,
    norm_out: torch.Tensor,
    x_q: torch.Tensor,
    x_scale: torch.Tensor,
    eps: float,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    M, D = x.shape
    groups = D // group_size
    tag = _backend_tag(x)
    global tle
    if _is_ascend_tag(tag) and M >= 8:
        if tle is None:
            import triton.experimental.tle as tle_module
            tle = tle_module
        _ascend_subvec_r2_all_even_unitflag_only_kernel[(triton.cdiv(M, 2),)](
            x, residual, gamma, residual_out, norm_out, x_q, x_scale,
            D=D,
            eps=eps,
            GROUP_SIZE=group_size,
            GROUPS=groups,
            SEGMENT=4096,
            SEGMENT_GROUPS=4096 // group_size,
            DIRECT_INT8=True,
            num_warps=8,
            num_stages=1,
            multibuffer=False,
            unit_flag=True,
        )
        return residual_out, norm_out, x_q, x_scale
    is_ascend = _is_ascend_tag(tag)
    is_hygon = _is_hygon_tag(tag)
    is_nvidia = _is_nvidia_tag(tag)
    is_thead = _is_thead_tag(tag)
    is_tianshu = _is_tianshu_tag(tag)

    # Ascend falls through to the exact, reproducible implementation below.
    if not is_ascend and is_nvidia and M >= 512 and D <= 6144:
        row_tile = 2
        segment = 2048
        _nvidia_parallel_row_group_streaming_kernel[
            (triton.cdiv(M, row_tile),)
        ](
            x, residual, gamma, residual_out, norm_out, x_q, x_scale,
            M=M,
            D=D,
            eps=eps,
            GROUP_SIZE=group_size,
            GROUPS=groups,
            ROW_TILE=row_tile,
            SEGMENT=segment,
            SEGMENT_GROUPS=segment // group_size,
            num_warps=8,
            num_stages=1,
        )
        return residual_out, norm_out, x_q, x_scale

    use_stable_full_row = (
        M == 128 and D == 4096 and (is_nvidia or is_thead)
    )
    if (
        not is_ascend
        and not is_tianshu
        and (is_hygon or is_nvidia or is_thead)
        and M >= 128
        and not use_stable_full_row
    ):
        segment = 2048
        target_warps = 4 if is_hygon else 8
        _preweighted_segmented_streaming_kernel[(M,)](
            x, residual, gamma, residual_out, norm_out, x_q, x_scale,
            D=D,
            eps=eps,
            GROUP_SIZE=group_size,
            GROUPS=groups,
            SEGMENT=segment,
            SEGMENT_GROUPS=segment // group_size,
            DIRECT_INT8=is_thead,
            STREAMING=True,
            num_warps=target_warps,
            num_stages=1,
        )
        return residual_out, norm_out, x_q, x_scale

    if _is_tianshu_tag(tag) and M >= 512:
        block_d = triton.next_power_of_2(D)
        _tianshu_wave64_hierarchical_max_kernel[(M,)](
            x,
            residual,
            gamma,
            residual_out,
            norm_out,
            x_q,
            x_scale,
            D=D,
            eps=eps,
            GROUP_SIZE=group_size,
            GROUPS=groups,
            BLOCK_D=block_d,
            BLOCK_GROUPS=block_d // group_size,
            WAVES_PER_GROUP=group_size // 64,
            num_warps=8,

            num_stages=1 if D == 4096 or D == 7168 else 2,
        )
        return residual_out, norm_out, x_q, x_scale
    return _launch_task01_stable(
        x,
        residual,
        gamma,
        residual_out,
        norm_out,
        x_q,
        x_scale,
        eps,
        group_size,
    )
