"""PCM audio operations with explicit clipping, layout, and length rules.

Channel handling is loudspeaker-based, not index-based.  Two sources that
declare the same channel count but different speaker masks are aligned by
position; content that has no exact destination is folded by category instead
of being dropped or assigned to an unrelated loudspeaker.
"""

from __future__ import annotations

import math
from typing import Any

from .errors import MediaOperationFailure
from .layouts import (
    NAMED_LAYOUTS,
    contributions,
    mask_for_positions,
    positions_for_mask,
)
from .wav import (
    AudioClip,
    decode_wav,
    encode_wav,
    rechannel,
    replace_channels,
)


PCM_MIN = -32768
PCM_MAX = 32767
MAX_DELIVERY_CHANNELS = 18


def _clip_sample(value: float) -> int:
    if value >= PCM_MAX:
        return PCM_MAX
    if value <= PCM_MIN:
        return PCM_MIN
    return int(round(value))


def _gain_factor(db: float) -> float:
    return math.pow(10.0, float(db) / 20.0)


def trim_audio(clip: AudioClip, operation: dict[str, Any]) -> AudioClip:
    start = int(round(float(operation["start_seconds"]) * clip.sample_rate))
    duration = int(round(float(operation["duration_seconds"]) * clip.sample_rate))
    if start < 0 or start >= clip.frame_count:
        raise MediaOperationFailure(
            "AUDIO_TRIM_OUT_OF_RANGE",
            "The requested audio trim starts outside the source.",
        )
    end = min(clip.frame_count, start + max(1, duration))
    if end <= start:
        raise MediaOperationFailure(
            "AUDIO_TRIM_EMPTY",
            "The requested audio trim produces no sample frames.",
        )
    return replace_channels(clip, (values[start:end] for values in clip.channels))


def gain_audio(clip: AudioClip, operation: dict[str, Any]) -> AudioClip:
    factor = _gain_factor(float(operation["db"]))
    return replace_channels(
        clip,
        (
            [_clip_sample(value * factor) for value in values]
            for values in clip.channels
        ),
    )


def normalize_audio(clip: AudioClip, operation: dict[str, Any]) -> AudioClip:
    peak = max(
        (abs(value) for channel in clip.channels for value in channel),
        default=0,
    )
    if peak == 0:
        return clip
    target = float(operation["target_peak"])
    desired = max(1.0, target * PCM_MAX)
    factor = desired / peak
    return replace_channels(
        clip,
        (
            [_clip_sample(value * factor) for value in values]
            for values in clip.channels
        ),
    )


def _delivery_positions(operation: dict[str, Any]) -> tuple[str, ...]:
    """Resolve a delivery operation to canonical (bit-ascending) positions."""
    target_layout = operation.get("target_layout")
    speaker_positions = operation.get("speaker_positions")
    if target_layout is not None:
        if speaker_positions is not None:
            raise MediaOperationFailure(
                "AUDIO_LAYOUT_AMBIGUOUS",
                "audio_layout accepts either target_layout or speaker_positions.",
            )
        if not isinstance(target_layout, str):
            raise MediaOperationFailure(
                "AUDIO_LAYOUT_INVALID",
                "target_layout must be a layout name.",
            )
        normalized = target_layout.strip().lower()
        if normalized not in NAMED_LAYOUTS:
            raise MediaOperationFailure(
                "AUDIO_LAYOUT_UNKNOWN",
                "target_layout is not a supported speaker layout.",
            )
        return positions_for_mask(NAMED_LAYOUTS[normalized])
    if speaker_positions is None:
        raise MediaOperationFailure(
            "AUDIO_LAYOUT_MISSING",
            "audio_layout requires target_layout or speaker_positions.",
        )
    if not isinstance(speaker_positions, (list, tuple)) or not speaker_positions:
        raise MediaOperationFailure(
            "AUDIO_LAYOUT_INVALID",
            "speaker_positions must be a non-empty array.",
        )
    requested = []
    for position in speaker_positions:
        if not isinstance(position, str):
            raise MediaOperationFailure(
                "AUDIO_LAYOUT_INVALID",
                "speaker_positions entries must be strings.",
            )
        requested.append(position.strip().upper())
    if len(set(requested)) != len(requested):
        raise MediaOperationFailure(
            "AUDIO_LAYOUT_INVALID",
            "speaker_positions must not repeat a position.",
        )
    try:
        mask = mask_for_positions(tuple(requested))
    except ValueError as exc:
        raise MediaOperationFailure(
            "AUDIO_LAYOUT_UNKNOWN",
            "speaker_positions contains an unsupported speaker.",
        ) from exc
    if len(positions_for_mask(mask)) != len(requested):
        raise MediaOperationFailure(
            "AUDIO_LAYOUT_INVALID",
            "speaker_positions is not a usable speaker set.",
        )
    return positions_for_mask(mask)


def _rendered_layout_mask(
    target_positions: tuple[str, ...],
) -> int | None:
    # Plain mono and stereo stay plain PCM: they are the canonical deliveries
    # whose speaker positions WAVE readers already imply, and keeping them
    # plain preserves the byte layout of legacy two-channel results.
    if target_positions == ("FC",) or target_positions == ("FL", "FR"):
        return None
    return mask_for_positions(target_positions)


def layout_audio(clip: AudioClip, operation: dict[str, Any]) -> AudioClip:
    target_positions = _delivery_positions(operation)
    if len(target_positions) > MAX_DELIVERY_CHANNELS:
        raise MediaOperationFailure(
            "AUDIO_LAYOUT_TOO_MANY_CHANNELS",
            "The requested speaker layout exceeds the channel budget.",
        )
    source_positions = clip.positions
    if len(source_positions) != clip.channel_count:
        raise MediaOperationFailure(
            "AUDIO_LAYOUT_SOURCE_INVALID",
            "The source channel declaration cannot be resolved.",
        )

    rendered: list[list[float]] = [
        [0.0] * clip.frame_count for _ in target_positions
    ]
    target_index = {
        position: index for index, position in enumerate(target_positions)
    }
    for source_index, source_position in enumerate(source_positions):
        routes = contributions(source_position, target_positions)
        values = clip.channels[source_index]
        for target_position, gain in routes:
            destination = rendered[target_index[target_position]]
            if gain == 1.0:
                for frame_index, value in enumerate(values):
                    destination[frame_index] += value
            else:
                for frame_index, value in enumerate(values):
                    destination[frame_index] += value * gain
    return rechannel(
        clip,
        ([_clip_sample(value) for value in channel] for channel in rendered),
        _rendered_layout_mask(target_positions),
    )


def _mix_routes(
    source_positions: tuple[str, ...],
    target_positions: tuple[str, ...],
) -> list[tuple[int, int, float]]:
    """Return (source index, target index, gain) triples for a mix."""
    target_index = {
        position: index for index, position in enumerate(target_positions)
    }
    routes: list[tuple[int, int, float]] = []
    for source_index, source_position in enumerate(source_positions):
        # A secondary speaker with no sensible destination simply cannot be
        # represented in the primary layout; it is not a plan failure.
        for target_position, gain in contributions(
            source_position,
            target_positions,
        ):
            destination = target_index.get(target_position)
            if destination is not None:
                routes.append((source_index, destination, gain))
    return routes


def mix_audio(
    primary: AudioClip,
    secondary: AudioClip,
    operation: dict[str, Any],
) -> AudioClip:
    if primary.sample_rate != secondary.sample_rate:
        raise MediaOperationFailure(
            "AUDIO_MIX_RATE_MISMATCH",
            "Mixed audio sources must share a sample rate.",
        )
    primary_positions = primary.positions
    secondary_positions = secondary.positions
    if len(primary_positions) != primary.channel_count:
        raise MediaOperationFailure(
            "AUDIO_MIX_LAYOUT_INVALID",
            "The primary channel declaration cannot be resolved.",
        )
    if len(secondary_positions) != secondary.channel_count:
        raise MediaOperationFailure(
            "AUDIO_MIX_LAYOUT_INVALID",
            "The secondary channel declaration cannot be resolved.",
        )

    frame_count = max(primary.frame_count, secondary.frame_count)
    primary_weight = float(operation.get("primary_weight", 1.0))
    secondary_weight = float(operation.get("secondary_weight", 1.0))
    secondary_factor = _gain_factor(float(operation["secondary_gain"])) * secondary_weight

    mixed: list[list[float]] = [
        [0.0] * frame_count for _ in range(primary.channel_count)
    ]
    for channel_index, values in enumerate(primary.channels):
        target = mixed[channel_index]
        weight = primary_weight
        for frame_index, value in enumerate(values):
            target[frame_index] += value * weight
    for source_index, target_index, gain in _mix_routes(
        secondary_positions,
        primary_positions,
    ):
        values = secondary.channels[source_index]
        target = mixed[target_index]
        factor = secondary_factor * gain
        for frame_index, value in enumerate(values):
            target[frame_index] += value * factor

    return replace_channels(
        primary,
        (
            [_clip_sample(value) for value in values]
            for values in mixed
        ),
    )


def render_audio(
    primary_bytes: bytes,
    operations: list[dict[str, Any]],
    secondary_bytes: bytes | None,
) -> bytes:
    clip = decode_wav(primary_bytes)
    secondary = decode_wav(secondary_bytes) if secondary_bytes is not None else None
    mixed = False

    try:
        for operation in operations:
            operation_type = operation["type"]
            if operation_type == "media_trim":
                clip = trim_audio(clip, operation)
            elif operation_type == "audio_gain":
                clip = gain_audio(clip, operation)
            elif operation_type == "audio_normalize":
                clip = normalize_audio(clip, operation)
            elif operation_type == "audio_layout":
                clip = layout_audio(clip, operation)
            elif operation_type == "audio_mix":
                if secondary is None:
                    raise MediaOperationFailure(
                        "AUDIO_MIX_SOURCE_MISSING",
                        "audio_mix requires a secondary media asset.",
                    )
                if mixed:
                    raise MediaOperationFailure(
                        "AUDIO_MIX_DUPLICATE",
                        "An audio plan may mix the secondary source only once.",
                    )
                clip = mix_audio(clip, secondary, operation)
                mixed = True
            else:
                raise MediaOperationFailure(
                    "UNSUPPORTED_AUDIO_OPERATION",
                    f"Unsupported audio operation: {operation_type}",
                )
    except (KeyError, TypeError, ValueError) as exc:
        raise MediaOperationFailure(
            "AUDIO_OPERATION_INVALID",
            "The audio operation payload is malformed.",
        ) from exc

    return encode_wav(clip)
