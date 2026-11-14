#!/usr/bin/env python3
"""Regression coverage for multichannel WAV layouts and delivery.

These tests exercise the media layer directly and through the HTTP boundary.
They are deliberately self-contained: media bytes are generated in memory and
no external tools are required.
"""

from __future__ import annotations

import base64
import hashlib
import http.client
import io
import json
import math
import struct
import sys
import tempfile
import threading
import unittest
import wave
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from avprovenance.http_api import build_http_server  # noqa: E402
from avprovenance.media.audio_ops import render_audio  # noqa: E402
from avprovenance.media.layouts import (  # noqa: E402
    NAMED_LAYOUTS,
    POSITION_BITS,
    positions_for_mask,
)
from avprovenance.media.wav import (  # noqa: E402
    decode_wav,
    encode_wav,
    inspect_wav,
)
from avprovenance.renditions.plan import compile_plan  # noqa: E402
from avprovenance.errors import PlanInvalid  # noqa: E402


PCM_SUBFORMAT_GUID = (
    b"\x01\x00\x00\x00\x00\x00\x10\x00"
    b"\x80\x00\x00\xaa\x00\x38\x9b\x71"
)


def mask_of(*positions: str) -> int:
    return sum(POSITION_BITS[position] for position in positions)


def make_extensible(
    channels: list[list[int]],
    *,
    channel_mask: int | None = None,
    sample_rate: int = 8000,
) -> bytes:
    channel_count = len(channels)
    frame_count = len(channels[0])
    if channel_mask is None:
        channel_mask = sum(1 << index for index in range(channel_count))
    interleaved = [
        sample
        for frame_index in range(frame_count)
        for sample in (channel[frame_index] for channel in channels)
    ]
    payload = struct.pack(f"<{len(interleaved)}h", *interleaved)
    fmt = struct.pack(
        "<HHIIHHHHI16s",
        0xFFFE,
        channel_count,
        sample_rate,
        sample_rate * channel_count * 2,
        channel_count * 2,
        16,
        22,
        16,
        channel_mask,
        PCM_SUBFORMAT_GUID,
    )
    body = (
        b"fmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data"
        + struct.pack("<I", len(payload))
        + payload
    )
    return b"RIFF" + struct.pack("<I", len(body) + 4) + b"WAVE" + body


def make_plain(channels: int, values: list[int], sample_rate: int = 8000) -> bytes:
    payload = io.BytesIO()
    with wave.open(payload, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(struct.pack(f"<{len(values)}h", *values))
    return payload.getvalue()


def make_tone(
    *,
    channels: int = 1,
    frequency: float = 440,
    amplitude: int = 10000,
    duration_seconds: float = 0.6,
    sample_rate: int = 8000,
) -> bytes:
    frame_count = int(duration_seconds * sample_rate)
    mono = [
        int(amplitude * math.sin(2 * math.pi * frequency * index / sample_rate))
        for index in range(frame_count)
    ]
    if channels == 1:
        values = mono
    else:
        values = []
        for sample in mono:
            values.extend([sample, int(sample * 0.5)])
    return make_plain(channels, values, sample_rate)


def parse_wave_chunks(data: bytes) -> dict[str, bytes]:
    position = 12
    chunks: dict[str, bytes] = {}
    while position + 8 <= len(data):
        chunk_id = data[position:position + 4]
        size = int.from_bytes(data[position + 4:position + 8], "little")
        chunks[chunk_id.decode("ascii")] = data[position + 8:position + 8 + size]
        position += 8 + size + (size & 1)
    return chunks


class LayoutDeclarationTests(unittest.TestCase):
    def test_plain_stereo_keeps_legacy_bytes_and_positions(self) -> None:
        source = make_tone(channels=2, frequency=330)
        self.assertEqual(encode_wav(decode_wav(source)), source)
        properties = inspect_wav(source).properties
        self.assertEqual(properties["channel_layout"], "stereo")
        self.assertEqual(properties["speaker_positions"], ["FL", "FR"])
        self.assertNotIn("channel_mask", properties)

    def test_plain_mono_is_front_centre(self) -> None:
        source = make_tone(channels=1)
        self.assertEqual(encode_wav(decode_wav(source)), source)
        properties = inspect_wav(source).properties
        self.assertEqual(properties["channel_layout"], "mono")
        self.assertEqual(properties["speaker_positions"], ["FC"])

    def test_extensible_roundtrip_preserves_declaration(self) -> None:
        source = make_extensible(
            [[10, 20, 30, 40]] * 6,
            channel_mask=mask_of("FL", "FR", "FC", "LFE", "BL", "BR"),
        )
        self.assertEqual(encode_wav(decode_wav(source)), source)
        properties = inspect_wav(source).properties
        self.assertEqual(properties["channels"], 6)
        self.assertEqual(properties["channel_layout"], "5.1")
        self.assertEqual(
            properties["speaker_positions"],
            ["FL", "FR", "FC", "LFE", "BL", "BR"],
        )
        self.assertEqual(properties["channel_mask"], 0x3F)

    def test_extensible_alternate_assignment_is_not_conventional(self) -> None:
        source = make_extensible(
            [[1, 2, 3, 4]] * 6,
            channel_mask=mask_of("FL", "FR", "FC", "LFE", "SL", "SR"),
        )
        properties = inspect_wav(source).properties
        self.assertEqual(properties["channel_layout"], "5.1(side)")
        self.assertEqual(
            properties["speaker_positions"],
            ["FL", "FR", "FC", "LFE", "SL", "SR"],
        )

    def test_frame_major_layout_uses_mask_bit_order(self) -> None:
        # The mask names FL, FR, FC; interleaved samples follow the same
        # ascending bit order, whatever order the caller typed the names in.
        source = make_extensible(
            [[1, 2, 3], [7, 8, 9], [4, 5, 6]],
            channel_mask=mask_of("FL", "FC", "FR"),
        )
        clip = decode_wav(source)
        self.assertEqual(clip.positions, ("FL", "FR", "FC"))
        self.assertEqual(clip.channels, ([1, 2, 3], [7, 8, 9], [4, 5, 6]))

    def test_mask_disagreeing_with_channel_count_is_rejected(self) -> None:
        source = make_extensible(
            [[1, 2]] * 3,
            channel_mask=mask_of("FL", "FR"),
        )
        with self.assertRaises(Exception):
            inspect_wav(source)

    def test_extensible_stereo_without_mask_uses_implicit_positions(self) -> None:
        # Some encoders write an extensible header with a zero mask for
        # stereo.  The WAVE convention still implies front left/right.
        source = make_extensible([[100, 200], [-100, -200]], channel_mask=0)
        properties = inspect_wav(source).properties
        self.assertEqual(properties["speaker_positions"], ["FL", "FR"])
        self.assertEqual(properties["channel_layout"], "stereo")
        self.assertNotIn("channel_mask", properties)

    def test_multichannel_extensible_without_mask_is_rejected(self) -> None:
        source = make_extensible([[1, 2]] * 6, channel_mask=0)
        with self.assertRaises(Exception):
            inspect_wav(source)

    def test_all_named_layouts_declare_their_positions(self) -> None:
        source = make_extensible(
            [[7, 8]] * 6,
            channel_mask=mask_of("FL", "FR", "FC", "LFE", "BL", "BR"),
        )
        for name, layout_mask in NAMED_LAYOUTS.items():
            with self.subTest(layout=name):
                output = render_audio(
                    source,
                    [{"type": "audio_layout", "target_layout": name}],
                    None,
                )
                descriptor = inspect_wav(output).properties
                expected = list(positions_for_mask(layout_mask))
                self.assertEqual(descriptor["speaker_positions"], expected)
                self.assertEqual(descriptor["channels"], len(expected))
                header_tag = struct.unpack_from(
                    "<H", parse_wave_chunks(output)["fmt "], 0
                )[0]
                if name in {"mono", "stereo"}:
                    self.assertEqual(header_tag, 0x0001)
                else:
                    self.assertEqual(header_tag, 0xFFFE)
                    declared = struct.unpack_from(
                        "<I", parse_wave_chunks(output)["fmt "], 20
                    )[0]
                    self.assertEqual(declared, layout_mask)


class AudioOperationLayoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.five_one = make_extensible(
            [
                [100, 200, 300, 400],      # FL
                [-100, -200, -300, -400],  # FR
                [1000, 1000, 1000, 1000],  # FC
                [5, 6, 7, 8],              # LFE
                [10, 20, 30, 40],          # BL
                [-10, -20, -30, -40],      # BR
            ],
            channel_mask=mask_of("FL", "FR", "FC", "LFE", "BL", "BR"),
        )

    def test_gain_and_normalize_keep_layout_declaration(self) -> None:
        output = render_audio(
            self.five_one,
            [
                {"type": "audio_gain", "db": -6},
                {"type": "audio_normalize", "target_peak": 0.5},
            ],
            None,
        )
        descriptor = inspect_wav(output).properties
        self.assertEqual(descriptor["channels"], 6)
        self.assertEqual(descriptor["channel_layout"], "5.1")
        self.assertEqual(descriptor["channel_mask"], 0x3F)
        peak = max(
            abs(sample)
            for channel in decode_wav(output).channels
            for sample in channel
        )
        self.assertLessEqual(abs(peak - int(round(0.5 * 32767))), 1)

    def test_stereo_downmix_keeps_every_position_audible(self) -> None:
        output = render_audio(
            self.five_one,
            [{"type": "audio_layout", "target_layout": "stereo"}],
            None,
        )
        clip = decode_wav(output)
        self.assertEqual(clip.channel_count, 2)
        # FL/FR survive at unity; FC, LFE, BL and BR also reach one side.
        self.assertGreater(clip.channels[0][0], 100)
        # A centre-only source is spread over both sides, not truncated.
        centre_only = make_extensible(
            [[0, 0]] * 2 + [[500, 500]] + [[0, 0]] * 3,
            channel_mask=mask_of("FL", "FR", "FC", "LFE", "BL", "BR"),
        )
        spread = decode_wav(
            render_audio(
                centre_only,
                [{"type": "audio_layout", "target_layout": "stereo"}],
                None,
            )
        )
        self.assertGreater(spread.channels[0][0], 0)
        self.assertGreater(spread.channels[1][0], 0)

    def test_surround_only_source_is_not_silently_dropped(self) -> None:
        surround_only = make_extensible(
            [[0, 0]] * 4 + [[500, 500], [500, 500]],
            channel_mask=mask_of("FL", "FR", "FC", "LFE", "BL", "BR"),
        )
        stereo = decode_wav(
            render_audio(
                surround_only,
                [{"type": "audio_layout", "target_layout": "stereo"}],
                None,
            )
        )
        self.assertGreater(stereo.channels[0][0], 0)
        self.assertGreater(stereo.channels[1][0], 0)

    def test_mix_aligns_by_position_not_index(self) -> None:
        primary = make_extensible(
            [[100] * 4, [0] * 4],
            channel_mask=mask_of("FL", "FR"),
        )
        # Secondary bit order is FR (index 0), FC (index 1).  The centre must
        # land on the primary centre panel rather than being treated as FL.
        secondary = make_extensible(
            [[500] * 4, [700] * 4],
            channel_mask=mask_of("FR", "FC"),
        )
        output = render_audio(
            primary,
            [{
                "type": "audio_mix",
                "secondary_gain": 0,
                "primary_weight": 1.0,
                "secondary_weight": 1.0,
            }],
            secondary,
        )
        properties = inspect_wav(output).properties
        self.assertEqual(properties["speaker_positions"], ["FL", "FR"])
        mixed = decode_wav(output)
        # FR (index 1) receives the secondary FR at unity; the secondary FC
        # is folded into both sides instead of taking one position outright.
        folded = int(round(700 * 0.7071067811865476))
        self.assertEqual(mixed.channels[1][0], 500 + folded)
        self.assertEqual(mixed.channels[0][0], 100 + folded)

    def test_equal_channel_counts_with_different_masks_are_distinct(self) -> None:
        primary = make_extensible(
            [[0] * 4] * 6,
            channel_mask=mask_of("FL", "FR", "FC", "LFE", "BL", "BR"),
        )
        secondary = make_extensible(
            [[1] * 4, [2] * 4, [3] * 4, [4] * 4, [5] * 4, [6] * 4],
            channel_mask=mask_of("FL", "FR", "FC", "LFE", "SL", "SR"),
        )
        output = decode_wav(
            render_audio(
                primary,
                [{
                    "type": "audio_mix",
                    "secondary_gain": 0,
                    "primary_weight": 1.0,
                    "secondary_weight": 1.0,
                }],
                secondary,
            )
        )
        properties = inspect_wav(
            render_audio(
                primary,
                [{
                    "type": "audio_mix",
                    "secondary_gain": 0,
                    "primary_weight": 1.0,
                    "secondary_weight": 1.0,
                }],
                secondary,
            )
        ).properties
        self.assertEqual(properties["speaker_positions"], [
            "FL", "FR", "FC", "LFE", "BL", "BR",
        ])
        # Side channels are folded into the matching back speakers, while
        # front and centre keep their own positions.
        self.assertEqual(output.channels[0][0], 1)
        self.assertEqual(output.channels[1][0], 2)
        self.assertEqual(output.channels[2][0], 3)
        self.assertEqual(output.channels[3][0], 4)
        self.assertGreater(output.channels[4][0], 0)
        self.assertGreater(output.channels[5][0], 0)
        self.assertLess(output.channels[4][0], 5)
        self.assertLess(output.channels[5][0], 6)

    def test_plain_stereo_mix_is_unchanged(self) -> None:
        primary = make_plain(2, [100, 200, 300, 400])
        secondary = make_plain(2, [10, 20, 30, 40])
        output = decode_wav(
            render_audio(
                primary,
                [{
                    "type": "audio_mix",
                    "secondary_gain": 0,
                    "primary_weight": 1.0,
                    "secondary_weight": 1.0,
                }],
                secondary,
            )
        )
        self.assertEqual(
            output.channels,
            ([110, 330], [220, 440]),
        )

    def test_legacy_recipe_bytes_match_reference_arithmetic(self) -> None:
        source = make_tone(channels=1, frequency=440, amplitude=10000)
        secondary = make_tone(channels=1, frequency=220, amplitude=7000)
        operations = [
            {
                "type": "media_trim",
                "start_seconds": 0.05,
                "duration_seconds": 0.3,
            },
            {"type": "audio_gain", "db": -3},
            {
                "type": "audio_mix",
                "secondary_gain": -6,
                "primary_weight": 0.75,
                "secondary_weight": 0.4,
            },
            {"type": "audio_normalize", "target_peak": 0.8},
        ]
        output = render_audio(source, operations, secondary)

        primary_values = list(
            struct.unpack("<4800h", parse_wave_chunks(source)["data"])
        )
        secondary_values = list(
            struct.unpack("<4800h", parse_wave_chunks(secondary)["data"])
        )
        start = int(round(0.05 * 8000))
        end = min(len(primary_values), start + max(1, int(round(0.3 * 8000))))
        gained = [
            max(-32768, min(32767, int(round(value * 10 ** (-3 / 20)))))
            for value in primary_values[start:end]
        ]
        factor = 10 ** (-6 / 20) * 0.4
        # Mixing follows the legacy rule of the longer source: the trimmed
        # primary is zero-padded up to the secondary length.
        mixed_frames = max(len(gained), len(secondary_values))
        mixed = [
            max(
                -32768,
                min(
                    32767,
                    int(round(
                        (gained[index] if index < len(gained) else 0) * 0.75
                        + secondary_values[index] * factor
                    )),
                ),
            )
            for index in range(mixed_frames)
        ]
        peak = max(abs(value) for value in mixed)
        scale = max(1.0, 0.8 * 32767) / peak
        normalized = [
            max(-32768, min(32767, int(round(value * scale))))
            for value in mixed
        ]
        payload = io.BytesIO()
        with wave.open(payload, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(8000)
            writer.writeframes(struct.pack(f"<{len(normalized)}h", *normalized))
        self.assertEqual(output, payload.getvalue())


class PlanValidationTests(unittest.TestCase):
    def test_target_layout_is_normalized_and_digested(self) -> None:
        plan = compile_plan(
            "audio",
            [{"type": "audio_layout", "target_layout": "  STEREO "}],
            has_secondary=False,
        )
        self.assertEqual(plan.operations, [{"type": "audio_layout", "target_layout": "stereo"}])

    def test_speaker_positions_are_canonicalized_to_bit_order(self) -> None:
        plan = compile_plan(
            "audio",
            [{
                "type": "audio_layout",
                "speaker_positions": ["FR", "FL", "FC"],
            }],
            has_secondary=False,
        )
        self.assertEqual(
            plan.operations,
            [{
                "type": "audio_layout",
                "speaker_positions": ["FL", "FR", "FC"],
            }],
        )

    def test_layout_rejects_both_declarations_or_neither(self) -> None:
        with self.assertRaises(PlanInvalid):
            compile_plan(
                "audio",
                [{
                    "type": "audio_layout",
                    "target_layout": "mono",
                    "speaker_positions": ["FC"],
                }],
                has_secondary=False,
            )
        with self.assertRaises(PlanInvalid):
            compile_plan(
                "audio",
                [{"type": "audio_layout"}],
                has_secondary=False,
            )

    def test_layout_rejects_unknown_or_repeated_positions(self) -> None:
        with self.assertRaises(PlanInvalid):
            compile_plan(
                "audio",
                [{"type": "audio_layout", "speaker_positions": ["XV"]}],
                has_secondary=False,
            )
        with self.assertRaises(PlanInvalid):
            compile_plan(
                "audio",
                [{"type": "audio_layout", "speaker_positions": ["FL", "FL"]}],
                has_secondary=False,
            )

    def test_layout_only_applies_to_audio(self) -> None:
        with self.assertRaises(PlanInvalid):
            compile_plan(
                "video",
                [{"type": "audio_layout", "target_layout": "mono"}],
                has_secondary=False,
            )


class HTTPMultichannelTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory(prefix="avprovenance-test-")
        self.server = build_http_server("127.0.0.1", 0, Path(self._temporary.name))
        self._thread = threading.Thread(
            target=self.server.serve_forever,
            kwargs={"poll_interval": 0.02},
            daemon=True,
        )
        self._thread.start()
        host, port = self.server.server_address[:2]
        self.host, self.port = str(host), int(port)

    def tearDown(self) -> None:
        self.server.shutdown()
        self._thread.join(timeout=5)
        self.server.server_close()
        self._temporary.cleanup()

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        connection = http.client.HTTPConnection(self.host, self.port, timeout=15)
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
            headers["Content-Length"] = str(len(body))
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            return response.status, json.loads(response.read().decode("utf-8"))
        finally:
            connection.close()

    def upload(self, filename: str, data: bytes) -> dict[str, Any]:
        status, body = self.request(
            "POST",
            "/v1/assets",
            {
                "filename": filename,
                "data_base64": base64.b64encode(data).decode("ascii"),
            },
        )
        self.assertEqual(status, 201, body)
        return body

    def test_multichannel_pipeline_and_reproduction(self) -> None:
        source_bytes = make_extensible(
            [
                [100, 200, 300, 400],
                [-100, -200, -300, -400],
                [1000, 1000, 1000, 1000],
                [5, 6, 7, 8],
                [10, 20, 30, 40],
                [-10, -20, -30, -40],
            ],
            channel_mask=mask_of("FL", "FR", "FC", "LFE", "BL", "BR"),
        )
        source = self.upload("surround-source.wav", source_bytes)
        self.assertEqual(source["descriptor"]["properties"]["channel_layout"], "5.1")

        secondary = self.upload(
            "surround-secondary.wav",
            make_extensible(
                [[1] * 4] * 6,
                channel_mask=mask_of("FL", "FR", "FC", "LFE", "SL", "SR"),
            ),
        )

        status, rendition = self.request(
            "POST",
            "/v1/renditions",
            {
                "primary_asset_id": source["id"],
                "secondary_asset_id": secondary["id"],
                "output_name": "surround-delivered.wav",
                "operations": [
                    {"type": "audio_gain", "db": -3},
                    {"type": "audio_normalize", "target_peak": 0.6},
                    {
                        "type": "audio_mix",
                        "secondary_gain": -12,
                        "primary_weight": 1.0,
                        "secondary_weight": 0.25,
                    },
                    {"type": "audio_layout", "target_layout": "5.1"},
                ],
            },
        )
        self.assertEqual(status, 201, rendition)
        self.assertEqual(rendition["status"], "completed")
        output_properties = rendition["output_asset"]["descriptor"]["properties"]
        self.assertEqual(output_properties["channel_layout"], "5.1")
        self.assertEqual(
            output_properties["speaker_positions"],
            ["FL", "FR", "FC", "LFE", "BL", "BR"],
        )

        output_asset = self.server.application.repository.get_asset(
            rendition["output_asset"]["id"]
        )
        stored = self.server.application.content_files.read(output_asset.content_key)
        descriptor = inspect_wav(stored).properties
        self.assertEqual(descriptor["channel_layout"], output_properties["channel_layout"])
        self.assertEqual(
            descriptor["speaker_positions"],
            output_properties["speaker_positions"],
        )

        status, check = self.request(
            "POST",
            f"/v1/renditions/{rendition['id']}/integrity-checks",
            {},
        )
        self.assertEqual(status, 201, check)
        self.assertEqual(check["status"], "verified")
        self.assertEqual(
            check["expected_descriptor"]["properties"]["speaker_positions"],
            ["FL", "FR", "FC", "LFE", "BL", "BR"],
        )

    def test_stereo_delivery_from_multichannel_source(self) -> None:
        source = self.upload(
            "surround-in.wav",
            make_extensible(
                [
                    [100, 200],
                    [-100, -200],
                    [1000, 1000],
                    [5, 6],
                    [10, 20],
                    [-10, -20],
                ],
                channel_mask=mask_of("FL", "FR", "FC", "LFE", "BL", "BR"),
            ),
        )
        status, rendition = self.request(
            "POST",
            "/v1/renditions",
            {
                "primary_asset_id": source["id"],
                "output_name": "stereo-out.wav",
                "operations": [
                    {"type": "audio_layout", "target_layout": "stereo"},
                ],
            },
        )
        self.assertEqual(status, 201, rendition)
        properties = rendition["output_asset"]["descriptor"]["properties"]
        self.assertEqual(properties["channels"], 2)
        self.assertEqual(properties["speaker_positions"], ["FL", "FR"])
        self.assertNotIn("channel_mask", properties)

        output_asset = self.server.application.repository.get_asset(
            rendition["output_asset"]["id"]
        )
        stored = self.server.application.content_files.read(output_asset.content_key)
        # Delivered stereo keeps a plain header, so any reader derives FL/FR.
        self.assertEqual(stored[20:22], b"\x01\x00")
        self.assertEqual(parse_wave_chunks(stored)["fmt "][:2], b"\x01\x00")

        status, check = self.request(
            "POST",
            f"/v1/renditions/{rendition['id']}/integrity-checks",
            {},
        )
        self.assertEqual(status, 201, check)
        self.assertEqual(check["status"], "verified")

    def test_multichannel_source_without_layout_still_completes(self) -> None:
        source = self.upload(
            "surround-plain.wav",
            make_extensible(
                [[9] * 4] * 6,
                channel_mask=mask_of("FL", "FR", "FC", "LFE", "BL", "BR"),
            ),
        )
        status, rendition = self.request(
            "POST",
            "/v1/renditions",
            {
                "primary_asset_id": source["id"],
                "output_name": "surround-gained.wav",
                "operations": [
                    {"type": "audio_gain", "db": 0},
                ],
            },
        )
        self.assertEqual(status, 201, rendition)
        self.assertEqual(
            rendition["output_asset"]["descriptor"]["properties"]["channel_mask"],
            0x3F,
        )


class DescriptorCompatibilityTests(unittest.TestCase):
    def test_reproduction_treats_legacy_descriptors_as_implicit_layouts(self) -> None:
        from avprovenance.models import MediaAsset, MediaDescriptor
        from avprovenance.renditions.verifier import reproduce_output

        source = make_plain(1, [100, 200, 300, 400])
        digest = hashlib.sha256(source).hexdigest()
        legacy = MediaAsset(
            id="asset_legacy",
            filename="legacy.wav",
            kind="audio",
            sha256=digest,
            size_bytes=len(source),
            content_key="content",
            descriptor=MediaDescriptor(
                kind="audio",
                mime_type="audio/wav",
                container="wav",
                properties={
                    "codec": "pcm_s16le",
                    "sample_rate": 8000,
                    "channels": 1,
                    "frame_count": 4,
                    "duration_seconds": 0.0005,
                },
            ),
            created_at="2026-01-01T00:00:00+00:00",
        )
        outcome = reproduce_output(
            output_asset=legacy,
            recorded_output_bytes=source,
            primary_bytes=source,
            operations=[{"type": "audio_gain", "db": 0}],
            secondary_bytes=None,
        )
        self.assertEqual(outcome.status, "verified")


if __name__ == "__main__":
    unittest.main()
