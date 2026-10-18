# Verification

What was run to check this node pack, with the raw numbers. Everything used the real released model.

## 1. Setup

- Windows 11 Pro (build 26200), Intel Core Ultra 9 285K (24 threads), 64 GB RAM, NVIDIA RTX 5090 32 GB (driver 610.88).
- ComfyUI v0.38.0 (commit `6b747c0428c343e1417219641db93a4fb7cb69ae`).
- Model: `canberkkkkkk/ema-lightning` at `7a6ba1ad216bb2f1da9863f80ac8770a6a807632` (SHA-256 in `tools/model_files.sha256`).
- Two Python 3.12 environments with torch 2.14.1+cu130, ema-lightning 1.1.2, normalizer-tr 0.8.0, huggingface-hub 2.2.0:
  "official" (only the package, as its README installs it) and "comfy" (ComfyUI's requirements plus this pack's).
- The package's batch-size cache (`XDG_CACHE_HOME`) pointed to the same folder for both, so `auto` meant the same measured batch
  size: 128 on the RTX 5090, 4 on the CPU.

## 2. Node vs the official package

**Official side** (official environment): `EMA(device=...)` (the documented constructor; files from the Hugging Face cache at the
pinned revision), `say(text, speed=..., seed=..., sample_rate=...)`, with the scheduler batch size set the way the node's
`batch_size` option sets it. **Node side** (comfy environment, `scripts/run_parity.py`): the Loader and Text to Speech nodes
in-process, the model loaded from local files.

| Case | Text | Seed | Speed | Rate | Batch | GPU | CPU |
| --- | --- | --- | --- | --- | --- | --- | --- |
| short | "Merhaba, size nasıl yardımcı olabilirim?" | 0 | 1.0 | 48 kHz | auto | identical | identical |
| numbers | date, time and amount | 0 | 1.0 | 48 kHz | auto | identical | identical |
| long | three sentences | 7 | 1.0 | 48 kHz | auto | identical | identical |
| seed_42 | as short | 42 | 1.0 | 48 kHz | auto | identical | identical |
| fast | one sentence | 3 | 1.5 | 48 kHz | auto | identical | identical |
| slow_24k | one sentence | 3 | 0.75 | 24 kHz | auto | identical | identical |
| 16k | one sentence | 5 | 1.0 | 16 kHz | auto | identical | identical |
| batch_1 | two sentences | 9 | 1.0 | 48 kHz | 1 | identical | identical |
| batch_8 | two sentences | 9 | 1.0 | 48 kHz | 8 | identical | identical |

"identical" = every sample of the waveform and every word time. **9 of 9 on the GPU and 9 of 9 on the CPU.**

**CPU between processes.** On the CPU the package itself is deterministic within a process and across identical processes, but the
float rounding can change with what the process imported: the official package run after `import comfy.model_management` differed
from a plain process by up to 6.3e-7 (-124 dB) for the same text and seed; with the same imports it was identical again. One
comparison between the node (inside ComfyUI) and a plain official process showed such differences in 7 of 9 cases; repeating the
official side with ComfyUI's model management imported gave 9 of 9 identical, as did a later plain run. The GPU results do not
depend on this.

## 3. Real ComfyUI queue (comfy environment, GPU)

Both shipped workflows submitted unchanged through `POST /prompt` (`scripts/queue_client.py`):

| Workflow | Result |
| --- | --- |
| `tts_basic` (FLAC) | 5.68 s of audio, 14 word times; the saved 16-bit FLAC equals the official float audio for the same text and seed rounded to 16 bits (largest difference 1.5e-5, half a 16-bit step) |
| `tts_announcement_mp3` | 7.88 s of audio in 0.036 s (real-time factor 0.0046), 22 word times, MP3 saved |

## 4. Behaviour of the package worth knowing (checked)

- **Batch size.** For a three-sentence text, batch size 1 gave different audio from 2, 3, 4, 8, 16 and 128 (which all agreed;
  largest difference 9.3e-4); same on the CPU. A one-sentence text gave the same audio for every batch size.
- **lightning** (GPU): on this machine `torch.compile` found no Triton, so the package recorded CUDA graphs without compiling (its own
  fallback, with a warning): 190 graphs in 158 s. The announcement then took 12 ms and differed from the plain path by up to 6.8e-4.
  The package also enables cuDNN benchmark mode process-wide at that point.
- **The model card's sample clips** (`assets/sample-1.wav` to `sample-3.wav`, "say() at seed 0") were not reproduced sample for
  sample here (their lengths were, to the sample): they were made on another machine and setup, and this generator's audio depends on
  the device and settings. The clips' waveforms and ours are uncorrelated sample by sample (SNR about -2.5 dB) although the same
  words are spoken with the same timing.
- **Text normalization** lowercases with Turkish rules ("ComfyUI" -> "comfyuı") and turns numbers, times and amounts into words.

## 5. Clean install

Fresh ComfyUI v0.38.0 clone, fresh Python 3.12 virtual environment (torch 2.14.1+cu130, ComfyUI's `requirements.txt`, then this
pack's `requirements.txt`: ema-lightning 1.1.2, normalizer-tr 0.8.0; huggingface-hub stayed at 2.2.0), `git clone` of this
repository, `tools/download_model.py --models-dir ComfyUI/models/ema_lightning` (3 files, SHA-256 verified). Both shipped workflows
through the real queue on the GPU: the decoded FLAC of `tts_basic` has the same samples as the development environment's, and both
workflows return the same word times. The package's `auto` batch size, measured fresh in this install, was 8 (the development
environment had measured 128 on the same GPU); for these texts every batch size of 2 or more gives the same audio.

## 6. Tests

- `tests/` (CPU, real model): the runtime against the package's own objects for several texts, seeds, speeds and sample rates;
  option validation; the nodes as ComfyUI loads them (ComfyUI's own `validate_prompt` on the shipped workflows, the AUDIO output,
  Save Audio (Advanced) writing a FLAC); the download manifest; a scan of the package for environment / network / process code.
- `scripts/ci_queue_check.py`: a stock ComfyUI server on the CPU, both shipped workflows through the queue; the same seed gives the
  same decoded audio, another seed different audio; word times come back; FLAC and MP3 are saved.
