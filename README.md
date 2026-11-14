# AV Derivative Provenance

AV Derivative Provenance is a Python backend for deterministic audio and video
derivation. It accepts PCM WAV and planar Y4M media, applies bounded
operations, records content digests and lineage, and reproduces a completed
result on demand to verify that the stored bytes still match the plan.

The application uses only the Python standard runtime. It does not need
FFmpeg, network access, credentials, or an online database.

## What it does

- Ingests signed 16-bit PCM WAV with one to eighteen channels, including
  `WAVE_FORMAT_EXTENSIBLE` files that declare a speaker layout.
- Ingests planar 4:2:0 YUV4MPEG2 in the `C420` family.
- Applies audio trim, gain, normalization, layout delivery, and two-source
  mixing.
- Applies video trim, even-aligned crop, luma offset, and horizontal mirror.
- Persists immutable assets, rendition state, provenance edges, and integrity
  checks.
- Replays the stored operation plan and compares both bytes and media
  descriptors.

## Quick start

```bash
python3 -m compileall -q avprovenance scripts
python3 -m avprovenance --host 127.0.0.1 --port 8765 --data-dir .avprovenance-data
```

The startup line includes the active host, port, and data directory. Stop the
process with `Ctrl+C`.

## HTTP endpoints

### Health

```http
GET /health
```

### Upload an asset

```http
POST /v1/assets
Content-Type: application/json

{
  "filename": "source.wav",
  "data_base64": "<strict base64>"
}
```

### Read an asset

```http
GET /v1/assets/{asset_id}
```

### Create a rendition

```http
POST /v1/renditions
Content-Type: application/json

{
  "primary_asset_id": "asset_...",
  "secondary_asset_id": null,
  "output_name": "derived.wav",
  "operations": [
    {
      "type": "media_trim",
      "start_seconds": 0.1,
      "duration_seconds": 0.3
    },
    {"type": "audio_gain", "db": -3},
    {"type": "audio_normalize", "target_peak": 0.8}
  ]
}
```

Rendition creation is synchronous at the HTTP boundary. The durable transition
history still records `queued`, `running`, and the terminal state.

Audio-only operations:

- `media_trim` with `start_seconds` and `duration_seconds`
- `audio_gain` with `db`
- `audio_normalize` with `target_peak`
- `audio_layout` with either `target_layout` or an explicit
  `speaker_positions` array
- `audio_mix` with `secondary_gain`, `primary_weight`, and `secondary_weight`

`audio_layout` is how a caller delivers a target loudspeaker arrangement.
`target_layout` accepts `mono`, `stereo`, `2.1`, `3.0`, `4.0`, `5.0`, `5.1`,
`5.0(side)`, `5.1(side)`, `6.1`, `7.0`, and `7.1`. `speaker_positions` accepts
a channel-count-explicit list such as `["FL", "FR", "FC", "LFE", "BL", "BR"]`;
it is stored in WAVE channel-mask bit order.

### Speaker positions and channel order

Every audio descriptor reports `channel_layout` and `speaker_positions`.
Plain PCM WAV is interpreted as `mono` = `["FC"]` and `stereo` =
`["FL", "FR"]`, which keeps the original left/right relationship.
`WAVE_FORMAT_EXTENSIBLE` material uses its 32-bit channel mask, and the mask
bit order is also the interleaved sample order.

Channel identity is the speaker position, never the array index. Two sources
with the same channel count but different masks are distinct; mixing maps by
position and folds positions that the primary layout lacks. Downmixing to
stereo keeps centre, surround, height, and LFE content audible instead of
copying the first two channels.

Deliveries of mono and stereo are written as plain PCM WAV, and any other
`audio_layout` target is written as `WAVE_FORMAT_EXTENSIBLE` with a matching
channel mask. An independent reader therefore sees the same positions the
asset descriptor and lineage report. Reproducing a rendition replays the same
layout mapping, so an integrity check cannot silently move a pan position.

```json
{"type": "audio_layout", "target_layout": "stereo"}
```

```json
{"type": "audio_layout", "speaker_positions": ["FL", "FR", "LFE"]}
```

Video-only operations:

- `media_trim` with `start_frame` and `frame_count`
- `video_crop` with even `x`, `y`, `width`, and `height`
- `video_luma_offset` with `amount`
- `video_mirror`

### Read rendition state

```http
GET /v1/renditions/{rendition_id}
```

### Read lineage

```http
GET /v1/renditions/{rendition_id}/lineage
```

### Verify reproduction

```http
POST /v1/renditions/{rendition_id}/integrity-checks
Content-Type: application/json

{}
```

The successful response has `status: "verified"`. A byte or descriptor
difference returns HTTP `409` with code `INTEGRITY_MISMATCH`; the mismatch
record remains durable.

## Media operation examples

Audio mixing plan:

```json
[
  {"type": "media_trim", "start_seconds": 0.05, "duration_seconds": 0.3},
  {"type": "audio_gain", "db": -3},
  {
    "type": "audio_mix",
    "secondary_gain": -6,
    "primary_weight": 0.75,
    "secondary_weight": 0.4
  },
  {"type": "audio_normalize", "target_peak": 0.8}
]
```

Multi-channel delivery plan:

```json
[
  {"type": "media_trim", "start_seconds": 0.05, "duration_seconds": 0.3},
  {"type": "audio_gain", "db": -3},
  {"type": "audio_layout", "target_layout": "5.1"},
  {"type": "audio_normalize", "target_peak": 0.8}
]
```

Video plan:

```json
[
  {"type": "media_trim", "start_frame": 1, "frame_count": 4},
  {
    "type": "video_crop",
    "x": 8,
    "y": 6,
    "width": 32,
    "height": 18
  },
  {"type": "video_luma_offset", "amount": 12},
  {"type": "video_mirror"}
]
```

## Data layout

```text
.avprovenance-data/
├── state.json
└── content/
    └── <digest-prefix>/
        └── <digest-prefix>/
            └── <sha256>
```

`state.json` is replaced atomically. Binary objects are content-addressed by
SHA-256 and written through temporary siblings before replacement.

## Source layout

```text
avprovenance/
├── application.py
├── http_api.py
├── models.py
├── validation.py
├── media/
├── renditions/
├── storage/
└── transport/
```

The HTTP boundary calls the application layer. The application coordinates
plan handling, media operations, and persistence. The media layer does not
depend on HTTP or the repository.

## Workflow checks

The production workflow checks are:

```bash
python3 -m compileall -q avprovenance scripts tests
python3 scripts/workflow_checks.py ingest-media-asset
python3 scripts/workflow_checks.py execute-rendition
python3 scripts/workflow_checks.py verify-reproduction
```

The multichannel behavior is additionally covered by:

```bash
python3 -m unittest discover -s tests
```

Each check starts a temporary HTTP server, generates media bytes in memory,
exercises the public endpoints, and shuts the server down.

## Environment

No environment variables are required. Runtime choices use `--host`, `--port`,
and `--data-dir`. Incoming request bodies are limited to 36 MiB, decoded media
payloads to 24 MiB, WAV geometry to one or two channels, and Y4M geometry to
1920x1080 with at most 240 frames.
