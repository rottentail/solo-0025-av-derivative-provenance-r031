"""Speaker positions and channel layouts for WAV audio.

The WAVE format declares channel order for ``WAVE_FORMAT_EXTENSIBLE`` headers
with a 32-bit channel mask.  Bits are ordered from the least significant bit,
which is also the interleaved sample order inside a frame.  This module keeps
that ordering in one place so inspection, mixing, and layout conversion all
agree about which sample index is which loudspeaker.
"""

from __future__ import annotations

# WAVE speaker positions in channel order (ascending channel-mask bit).
SPEAKER_BITS: tuple[tuple[str, int], ...] = (
    ("FL", 0x00001),  # front left
    ("FR", 0x00002),  # front right
    ("FC", 0x00004),  # front centre
    ("LFE", 0x00008),  # low frequency effects
    ("BL", 0x00010),  # back left
    ("BR", 0x00020),  # back right
    ("FLC", 0x00040),  # front left of centre
    ("FRC", 0x00080),  # front right of centre
    ("BC", 0x00100),  # back centre
    ("SL", 0x00200),  # side left
    ("SR", 0x00400),  # side right
    ("TC", 0x00800),  # top centre
    ("TFL", 0x01000),  # top front left
    ("TFC", 0x02000),  # top front centre
    ("TFR", 0x04000),  # top front right
    ("TBL", 0x08000),  # top back left
    ("TBC", 0x10000),  # top back centre
    ("TBR", 0x20000),  # top back right
)

POSITION_BITS: dict[str, int] = {name: bit for name, bit in SPEAKER_BITS}
VALID_CHANNEL_MASK = sum(bit for _name, bit in SPEAKER_BITS)


# Canonical layout names.  Delivered stereo and mono use plain PCM headers,
# matching the legacy encoder byte-for-byte; every other target layout uses a
# WAVE_FORMAT_EXTENSIBLE header so independent readers see the declaration.
_LAYOUT_NAMES: tuple[tuple[str, int], ...] = (
    ("mono", 0x00004),
    ("stereo", 0x00003),
    ("2.1", 0x0000B),
    ("3.0", 0x00007),
    ("4.0", 0x00033),
    ("5.0", 0x00037),
    ("5.1", 0x0003F),
    ("5.0(side)", 0x00607),
    ("5.1(side)", 0x0060F),
    ("6.1", 0x00707),
    ("7.0", 0x00637),
    ("7.1", 0x0063F),
)

NAMED_LAYOUTS: dict[str, int] = dict(_LAYOUT_NAMES)

_LAYOUT_BY_MASK: dict[int, str] = {
    mask: name for name, mask in _LAYOUT_NAMES
}

# A quiet fold is used whenever a channel has to change family (for example a
# back-left channel delivered to a front-left speaker), and when a centre-like
# channel is spread over a left/right pair.  Exact speaker matches keep unity
# gain and single-channel layouts use half gain to avoid doubling amplitude.
_FOLD_GAIN = 0.7071067811865476
_HALF_GAIN = 0.5


def positions_for_mask(channel_mask: int) -> tuple[str, ...]:
    """Return loudspeaker labels in WAVE channel order for ``channel_mask``."""
    return tuple(
        name for name, bit in SPEAKER_BITS if channel_mask & bit
    )


def mask_for_positions(positions: tuple[str, ...]) -> int:
    mask = 0
    for position in positions:
        try:
            mask |= POSITION_BITS[position]
        except KeyError as exc:
            raise ValueError(f"Unknown speaker position: {position}") from exc
    return mask


def mask_is_valid(channel_mask: int) -> bool:
    return (
        channel_mask > 0
        and channel_mask & ~VALID_CHANNEL_MASK == 0
    )


def implicit_positions(channel_count: int) -> tuple[str, ...]:
    """Speaker positions implied by a plain PCM WAV header.

    Plain PCM has no channel mask.  The WAVE convention is mono at the front
    centre and stereo as front left/right, which also preserves the original
    left/right relationship of legacy two-channel material.
    """
    if channel_count == 1:
        return ("FC",)
    if channel_count == 2:
        return ("FL", "FR")
    raise ValueError("Plain PCM WAV layout is only defined for one or two channels.")


def layout_name_for_mask(channel_mask: int) -> str:
    """Return the canonical layout name for a declared channel mask."""
    return _LAYOUT_BY_MASK.get(channel_mask, "custom")


def _first_available(
    targets: set[str],
    candidates: tuple[str, ...],
    gain: float,
) -> tuple[str, float] | None:
    for candidate in candidates:
        if candidate in targets:
            return (candidate, gain)
    return None


def _left_destination(
    position: str,
    targets: set[str],
) -> tuple[tuple[str, float], ...]:
    candidates = {
        "FL": ("FL", "FLC", "SL", "BL", "TFL", "TBL"),
        "FLC": ("FLC", "FL", "SL", "BL", "TFL", "TBL"),
        "SL": ("SL", "BL", "TBL", "FL", "FLC", "TFL"),
        "BL": ("BL", "SL", "TBL", "FL", "FLC", "TFL"),
        "TFL": ("TFL", "FL", "FLC", "SL", "BL", "TBL"),
        "TBL": ("TBL", "BL", "SL", "TFL", "FL", "FLC"),
    }[position]
    match = _first_available(targets, candidates, _FOLD_GAIN)
    if match is not None:
        return (match,)
    if "FC" in targets:
        # A mono delivery receives each side of the stereo image once, at
        # half gain, so a full-scale left plus right pair does not clip.
        return (("FC", _HALF_GAIN),)
    if "TC" in targets:
        return (("TC", _FOLD_GAIN),)
    if "FR" in targets:
        return (("FR", _FOLD_GAIN),)
    return ()


def _right_destination(
    position: str,
    targets: set[str],
) -> tuple[tuple[str, float], ...]:
    candidates = {
        "FR": ("FR", "FRC", "SR", "BR", "TFR", "TBR"),
        "FRC": ("FRC", "FR", "SR", "BR", "TFR", "TBR"),
        "SR": ("SR", "BR", "TBR", "FR", "FRC", "TFR"),
        "BR": ("BR", "SR", "TBR", "FR", "FRC", "TFR"),
        "TFR": ("TFR", "FR", "FRC", "SR", "BR", "TBR"),
        "TBR": ("TBR", "BR", "SR", "TFR", "FR", "FRC"),
    }[position]
    match = _first_available(targets, candidates, _FOLD_GAIN)
    if match is not None:
        return (match,)
    if "FC" in targets:
        return (("FC", _HALF_GAIN),)
    if "TC" in targets:
        return (("TC", _FOLD_GAIN),)
    if "FL" in targets:
        return (("FL", _FOLD_GAIN),)
    return ()


def _centre_destination(
    position: str,
    targets: set[str],
) -> tuple[tuple[str, float], ...]:
    candidates = {
        "FC": ("FC", "TFC", "TC"),
        "TFC": ("TFC", "FC", "TC"),
        "TC": ("TC", "TFC", "FC"),
    }[position]
    match = _first_available(targets, candidates, _FOLD_GAIN)
    if match is not None:
        return (match,)
    if "FL" in targets and "FR" in targets:
        return (("FL", _FOLD_GAIN), ("FR", _FOLD_GAIN))
    if "FL" in targets:
        return (("FL", _FOLD_GAIN),)
    if "FR" in targets:
        return (("FR", _FOLD_GAIN),)
    return ()


def _back_centre_destination(
    targets: set[str],
) -> tuple[tuple[str, float], ...]:
    match = _first_available(targets, ("BC", "TBC"), _FOLD_GAIN)
    if match is not None:
        return (match,)
    left = _first_available(
        targets,
        ("BL", "SL", "TBL", "FL", "FLC"),
        _FOLD_GAIN,
    )
    right = _first_available(
        targets,
        ("BR", "SR", "TBR", "FR", "FRC"),
        _FOLD_GAIN,
    )
    if left is not None and right is not None:
        return (left, right)
    if left is not None:
        return (left,)
    if right is not None:
        return (right,)
    if "FC" in targets:
        return (("FC", _HALF_GAIN),)
    return _centre_destination("FC", targets)


def _lfe_destination(
    targets: set[str],
) -> tuple[tuple[str, float], ...]:
    if "LFE" in targets:
        return (("LFE", 1.0),)
    left = _first_available(targets, ("FL", "FLC"), _HALF_GAIN)
    right = _first_available(targets, ("FR", "FRC"), _HALF_GAIN)
    if left is not None and right is not None:
        return (left, right)
    match = left or right
    if match is not None:
        return (match,)
    if "FC" in targets:
        return (("FC", _HALF_GAIN),)
    # A hypothetical target made only of height or centre speakers still has
    # to receive the low-frequency content somewhere.
    return _centre_destination("FC", targets)


def contributions(
    position: str,
    target_positions: tuple[str, ...],
) -> tuple[tuple[str, float], ...]:
    """Map one source loudspeaker onto target loudspeakers.

    A source position that also exists in the target keeps unity gain.  Other
    positions are folded by category, so downmixing to stereo (or mono) keeps
    the centre, surround, height, and LFE content audible instead of dropping
    every channel after the first two.
    """
    targets = set(target_positions)
    if position in targets:
        return ((position, 1.0),)
    if position in {"FL", "FLC", "SL", "BL", "TFL", "TBL"}:
        routes = _left_destination(position, targets)
    elif position in {"FR", "FRC", "SR", "BR", "TFR", "TBR"}:
        routes = _right_destination(position, targets)
    elif position in {"FC", "TFC", "TC"}:
        routes = _centre_destination(position, targets)
    elif position == "LFE":
        routes = _lfe_destination(targets)
    elif position in {"BC", "TBC"}:
        routes = _back_centre_destination(targets)
    else:
        routes = ()
    if routes:
        return routes
    if not target_positions:
        return ()
    # A custom target made only of positions outside this source speaker's
    # family still has to receive the content somewhere deterministic.
    return ((target_positions[0], _FOLD_GAIN),)
