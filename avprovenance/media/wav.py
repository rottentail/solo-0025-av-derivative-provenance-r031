"""PCM WAV inspection, decoding, and deterministic encoding.

Plain ``WAVE_FORMAT_PCM`` files with one or two channels keep the legacy
left/right interpretation and byte layout.  Multichannel material uses
``WAVE_FORMAT_EXTENSIBLE`` with a channel mask; the mask bit order is also the
interleaved sample order.  Encoding preserves whichever declaration the clip
carries, so an independent reader sees the same loudspeaker assignment that
the media descriptor reports.
"""

from __future__ import annotations

import array
import io
import struct
import sys
import wave
from dataclasses import dataclass
from typing import Iterable, Sequence

from ..errors import UnsupportedMedia
from ..models import MediaDescriptor
from .layouts import (
    implicit_positions,
    layout_name_for_mask,
    mask_is_valid,
    positions_for_mask,
)

WAVE_FORMAT_PCM = 0x0001
WAVE_FORMAT_EXTENSIBLE = 0xFFFE
MAX_WAV_CHANNELS = 18
# KSDATAFORMAT_SUBTYPE_PCM from the WAVE_FORMAT_EXTENSIBLE specification.
PCM_SUBFORMAT_GUID = (
    b"\x01\x00\x00\x00\x00\x00\x10\x00"
    b"\x80\x00\x00\xaa\x00\x38\x9b\x71"
)


@dataclass(frozen=True)
class AudioClip:
    sample_rate: int
    channels: tuple[list[int], ...]
    # ``None`` means a plain PCM header, whose positions are implicit.  A mask
    # value means the bytes carry a WAVE_FORMAT_EXTENSIBLE declaration.
    channel_mask: int | None = None

    @property
    def frame_count(self) -> int:
        return len(self.channels[0]) if self.channels else 0

    @property
    def channel_count(self) -> int:
        return len(self.channels)

    @property
    def positions(self) -> tuple[str, ...]:
        if self.channel_mask is None:
            return implicit_positions(self.channel_count)
        return positions_for_mask(self.channel_mask)


@dataclass(frozen=True)
class _ParsedWav:
    sample_rate: int
    channels: int
    channel_mask: int | None
    frame_count: int
    payload: bytes


def _scan_chunks(data: bytes) -> dict[bytes, bytes]:
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise UnsupportedMedia(
            "The audio payload is not a readable RIFF/WAVE file."
        )
    chunks: dict[bytes, bytes] = {}
    position = 12
    while position + 8 <= len(data):
        chunk_id = data[position:position + 4]
        size = int.from_bytes(data[position + 4:position + 8], "little")
        start = position + 8
        end = start + size
        if end > len(data):
            raise UnsupportedMedia(
                "A WAV chunk extends past the payload.",
                details={"chunk": chunk_id.decode("latin-1", "replace")},
            )
        chunks.setdefault(chunk_id, data[start:end])
        position = end + (size & 1)
    if b"fmt " not in chunks:
        raise UnsupportedMedia("The WAV payload has no format chunk.")
    if b"data" not in chunks:
        raise UnsupportedMedia("The WAV payload has no sample data chunk.")
    return chunks


def _validate_rate(sample_rate: int) -> None:
    if sample_rate < 8000 or sample_rate > 192000:
        raise UnsupportedMedia(
            "WAV sample rate is outside the supported range.",
            details={"sample_rate": sample_rate},
        )


def _validate_block(channels: int, bits: int, block_align: int, *, strict: bool) -> None:
    if bits != 16:
        raise UnsupportedMedia(
            "Only signed 16-bit PCM WAV is supported.",
            details={"bits_per_sample": bits},
        )
    if strict and block_align != channels * 2:
        raise UnsupportedMedia(
            "The WAV block alignment does not match its channels.",
            details={"block_align": block_align, "channels": channels},
        )


def _parse_plain_format(fmt: bytes) -> tuple[int, int | None]:
    if len(fmt) < 16:
        raise UnsupportedMedia("The WAV format chunk is truncated.")
    (_tag, channels, _rate, _byte_rate, block_align, bits) = struct.unpack_from(
        "<HHIIHH", fmt, 0
    )
    if channels not in (1, 2):
        raise UnsupportedMedia(
            "WAV input must contain one or two channels.",
            details={"channels": channels},
        )
    _validate_block(channels, bits, block_align, strict=False)
    return channels, None


def _parse_extensible_format(fmt: bytes) -> tuple[int, int]:
    if len(fmt) < 40:
        raise UnsupportedMedia("The extensible WAV format chunk is truncated.")
    (
        _tag,
        channels,
        _rate,
        _byte_rate,
        block_align,
        bits,
        size_extra,
    ) = struct.unpack_from("<HHIIHHH", fmt, 0)
    if size_extra < 22:
        raise UnsupportedMedia(
            "The extensible WAV format extension is too short.",
            details={"cb_size": size_extra},
        )
    valid_bits, channel_mask = struct.unpack_from("<HI", fmt, 18)
    subformat = fmt[24:40]
    if subformat != PCM_SUBFORMAT_GUID:
        raise UnsupportedMedia(
            "The extensible WAV subformat is not signed PCM.",
            details={"subformat_hex": subformat.hex()},
        )
    _validate_block(channels, bits, block_align, strict=True)
    if valid_bits not in (0, 16):
        raise UnsupportedMedia(
            "Only signed 16-bit PCM WAV is supported.",
            details={"valid_bits_per_sample": valid_bits},
        )
    if channels < 1 or channels > MAX_WAV_CHANNELS:
        raise UnsupportedMedia(
            "WAV channel count is outside the supported range.",
            details={"channels": channels, "maximum": MAX_WAV_CHANNELS},
        )
    if channel_mask == 0:
        # Some encoders emit an extensible header without a mask for mono or
        # stereo.  The WAVE convention still implies front centre and front
        # left/right, so those files remain usable.
        if channels not in (1, 2):
            raise UnsupportedMedia(
                "A multichannel WAV must declare a channel mask.",
                details={"channels": channels},
            )
        return channels, None
    if not mask_is_valid(channel_mask):
        raise UnsupportedMedia(
            "The WAV channel mask does not name supported speakers.",
            details={"channel_mask": hex(channel_mask)},
        )
    if bin(channel_mask).count("1") != channels:
        raise UnsupportedMedia(
            "The WAV channel mask disagrees with its channel count.",
            details={"channels": channels, "channel_mask": hex(channel_mask)},
        )
    return channels, channel_mask


def _parse_wav(data: bytes) -> _ParsedWav:
    chunks = _scan_chunks(data)
    fmt = chunks[b"fmt "]
    tag = struct.unpack_from("<H", fmt, 0)[0] if len(fmt) >= 2 else 0
    if tag == WAVE_FORMAT_PCM:
        channels, channel_mask = _parse_plain_format(fmt)
    elif tag == WAVE_FORMAT_EXTENSIBLE:
        channels, channel_mask = _parse_extensible_format(fmt)
    else:
        raise UnsupportedMedia(
            "Compressed WAV payloads are not supported.",
            details={"format_tag": hex(tag)},
        )
    sample_rate = struct.unpack_from("<I", fmt, 4)[0]
    _validate_rate(sample_rate)

    frame_size = channels * 2
    payload = chunks[b"data"]
    if len(payload) % frame_size:
        raise UnsupportedMedia("WAV sample data is not frame-aligned.")
    frame_count = len(payload) // frame_size
    if frame_count <= 0:
        raise UnsupportedMedia("WAV input has no sample frames.")
    return _ParsedWav(
        sample_rate=sample_rate,
        channels=channels,
        channel_mask=channel_mask,
        frame_count=frame_count,
        payload=payload,
    )


def _descriptor_properties(parsed: _ParsedWav) -> dict[str, object]:
    if parsed.channel_mask is None:
        positions = implicit_positions(parsed.channels)
        layout_name = "mono" if parsed.channels == 1 else "stereo"
    else:
        positions = positions_for_mask(parsed.channel_mask)
        layout_name = layout_name_for_mask(parsed.channel_mask)
    properties: dict[str, object] = {
        "codec": "pcm_s16le",
        "sample_rate": parsed.sample_rate,
        "channels": parsed.channels,
        "frame_count": parsed.frame_count,
        "duration_seconds": round(
            parsed.frame_count / parsed.sample_rate,
            9,
        ),
        "channel_layout": layout_name,
        "speaker_positions": list(positions),
    }
    if parsed.channel_mask is not None:
        properties["channel_mask"] = parsed.channel_mask
    return properties


def inspect_wav(data: bytes) -> MediaDescriptor:
    parsed = _parse_wav(data)
    return MediaDescriptor(
        kind="audio",
        mime_type="audio/wav",
        container="wav",
        properties=_descriptor_properties(parsed),
    )


def decode_wav(data: bytes) -> AudioClip:
    parsed = _parse_wav(data)
    interleaved = array.array("h")
    interleaved.frombytes(parsed.payload)
    if sys.byteorder != "little":
        interleaved.byteswap()
    channel_values = tuple(
        interleaved[index::parsed.channels].tolist()
        for index in range(parsed.channels)
    )
    return AudioClip(
        sample_rate=parsed.sample_rate,
        channels=channel_values,
        channel_mask=parsed.channel_mask,
    )


def _interleave(channels: tuple[list[int], ...], frame_count: int) -> list[int]:
    channel_count = len(channels)
    result = [0] * (frame_count * channel_count)
    for channel_index, values in enumerate(channels):
        result[channel_index::channel_count] = values[:frame_count]
    return result


def _plain_wav_bytes(clip: AudioClip, values: array.array) -> bytes:
    payload = io.BytesIO()
    with wave.open(payload, "wb") as writer:
        writer.setnchannels(len(clip.channels))
        writer.setsampwidth(2)
        writer.setframerate(clip.sample_rate)
        writer.writeframes(values.tobytes())
    return payload.getvalue()


def _extensible_wav_bytes(clip: AudioClip, values: array.array) -> bytes:
    channel_mask = clip.channel_mask
    if channel_mask is None:
        raise UnsupportedMedia(
            "An extensible WAV clip requires a channel mask."
        )
    channels = len(clip.channels)
    byte_rate = clip.sample_rate * channels * 2
    block_align = channels * 2
    format_payload = struct.pack(
        "<HHIIHHHHI16s",
        WAVE_FORMAT_EXTENSIBLE,
        channels,
        clip.sample_rate,
        byte_rate,
        block_align,
        16,
        22,
        16,
        channel_mask,
        PCM_SUBFORMAT_GUID,
    )
    sample_bytes = values.tobytes()
    data_padding = b"\x00" if len(sample_bytes) % 2 else b""
    riff_payload = b"".join(
        (
            b"fmt ",
            struct.pack("<I", len(format_payload)),
            format_payload,
            b"data",
            struct.pack("<I", len(sample_bytes)),
            sample_bytes,
            data_padding,
        )
    )
    return b"".join(
        (
            b"RIFF",
            struct.pack("<I", len(riff_payload) + 4),
            b"WAVE",
            riff_payload,
        )
    )


def encode_wav(clip: AudioClip) -> bytes:
    if not clip.channels:
        raise UnsupportedMedia("Cannot encode an audio clip without channels.")
    if any(len(channel) != clip.frame_count for channel in clip.channels):
        raise UnsupportedMedia("Audio channels do not share one frame count.")
    if clip.channel_mask is None:
        if clip.channel_count not in (1, 2):
            raise UnsupportedMedia(
                "A plain PCM clip supports only one or two channels.",
                details={"channels": clip.channel_count},
            )
    else:
        positions = positions_for_mask(clip.channel_mask)
        if len(positions) != clip.channel_count:
            raise UnsupportedMedia(
                "The clip channel mask disagrees with its channel count."
            )
    values = array.array("h", _interleave(clip.channels, clip.frame_count))
    if sys.byteorder != "little":
        values.byteswap()
    if clip.channel_mask is None:
        return _plain_wav_bytes(clip, values)
    return _extensible_wav_bytes(clip, values)


def replace_channels(
    clip: AudioClip,
    channels: Iterable[Sequence[int]],
) -> AudioClip:
    """Copy channels while keeping the clip's channel declaration."""
    values = tuple(list(channel) for channel in channels)
    if not values:
        raise UnsupportedMedia("Audio operations produced no channels.")
    expected = len(values[0])
    if any(len(channel) != expected for channel in values):
        raise UnsupportedMedia("Audio operation produced misaligned channels.")
    return AudioClip(
        sample_rate=clip.sample_rate,
        channels=values,
        channel_mask=clip.channel_mask,
    )


def rechannel(
    clip: AudioClip,
    channels: Iterable[Sequence[int]],
    channel_mask: int | None,
) -> AudioClip:
    """Replace channels together with their loudspeaker declaration."""
    values = tuple(list(channel) for channel in channels)
    if not values:
        raise UnsupportedMedia("Audio operations produced no channels.")
    expected = len(values[0])
    if any(len(channel) != expected for channel in values):
        raise UnsupportedMedia("Audio operation produced misaligned channels.")
    if channel_mask is None:
        if len(values) not in (1, 2):
            raise UnsupportedMedia(
                "A plain PCM clip supports only one or two channels."
            )
    elif bin(channel_mask).count("1") != len(values):
        raise UnsupportedMedia(
            "The channel mask disagrees with the produced channel count."
        )
    return AudioClip(
        sample_rate=clip.sample_rate,
        channels=values,
        channel_mask=channel_mask,
    )
