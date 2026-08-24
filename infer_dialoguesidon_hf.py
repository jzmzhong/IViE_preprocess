"""Dialogue Sidon inference with chunked processing from Hugging Face Hub.

Loads exported torch.export components from sarulab-speech/DialogueSidon on
Hugging Face Hub and runs diffusion-based speaker separation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import torch
import torch.nn.functional as F
import torchaudio
from diffusers import DPMSolverMultistepScheduler
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ID = "sarulab-speech/DialogueSidon"
MODEL_FILES = ["ssl_encoder.pt2", "diffusion_head.pt2", "vae_decoder.pt2", "metadata.json"]
SAMPLE_RATE_IN = 16_000

# ---------------------------------------------------------------------------
# Model loading from Hugging Face Hub
# ---------------------------------------------------------------------------

_cache: dict = {}


def load_models_from_hub(device: torch.device) -> dict:
    """Download and load exported model components from Hugging Face Hub."""
    cache_key = str(device)
    if cache_key in _cache:
        return _cache[cache_key]

    print(f"Downloading model files from {REPO_ID} ...")
    paths = {
        f: hf_hub_download(repo_id=REPO_ID, filename=f)
        for f in MODEL_FILES
    }

    with open(paths["metadata.json"]) as fp:
        meta = json.load(fp)

    ssl_encoder = torch.export.load(paths["ssl_encoder.pt2"]).module().to(device)
    diffusion_head = torch.export.load(paths["diffusion_head.pt2"]).module().to(device)
    vae_decoder = torch.export.load(paths["vae_decoder.pt2"]).module().to(device)

    latent_norm_mean = torch.tensor(
        meta["latent_norm_mean"], dtype=torch.float32, device=device
    ).view(1, 1, -1)
    latent_norm_std = torch.tensor(
        meta["latent_norm_std"], dtype=torch.float32, device=device
    ).view(1, 1, -1)

    scheduler = DPMSolverMultistepScheduler.from_config(
        meta["ddpm_config"],
        algorithm_type="dpmsolver++",
        timestep_spacing="linspace",
    )

    models = {
        "ssl_encoder": ssl_encoder,
        "diffusion_head": diffusion_head,
        "vae_decoder": vae_decoder,
        "latent_norm_mean": latent_norm_mean,
        "latent_norm_std": latent_norm_std,
        "latent_norm_initialized": meta["latent_norm_initialized"],
        "scheduler": scheduler,
        "latent_dim": meta["latent_dim"],
        "sample_rate": meta["sample_rate"],
    }
    _cache[cache_key] = models
    return models


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _pad_batch(
    features: list[torch.Tensor],
    pad_to_multiple_of: int = 2,
    padding_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a batch of feature tensors to the same length."""
    target_length = max(f.shape[0] for f in features)
    if pad_to_multiple_of:
        target_length = (
            (target_length + pad_to_multiple_of - 1)
            // pad_to_multiple_of
            * pad_to_multiple_of
        )
    batch_size = len(features)
    feature_dim = features[0].shape[1]
    device = features[0].device
    padded = torch.full(
        (batch_size, target_length, feature_dim),
        padding_value,
        dtype=torch.float32,
        device=device,
    )
    mask = torch.zeros((batch_size, target_length), dtype=torch.int64, device=device)
    for i, feat in enumerate(features):
        padded[i, : feat.shape[0]] = feat
        mask[i, : feat.shape[0]] = 1
    return padded, mask


def extract_fbank_features(
    waveforms: list[torch.Tensor],
    device: torch.device,
    num_mel_bins: int = 80,
    stride: int = 2,
) -> dict[str, torch.Tensor]:
    features = []
    for wav in waveforms:
        if wav.ndim > 1:
            wav = wav[0]
        feat = torchaudio.compliance.kaldi.fbank(
            wav.unsqueeze(0),
            sample_frequency=SAMPLE_RATE_IN,
            num_mel_bins=num_mel_bins,
            frame_length=25,
            frame_shift=10,
            dither=0.0,
            preemphasis_coefficient=0.97,
            remove_dc_offset=True,
            window_type="povey",
            use_energy=False,
            energy_floor=1.192092955078125e-07,
        )
        mean = feat.mean(0, keepdim=True)
        var = feat.var(0, keepdim=True)
        feat = (feat - mean) / torch.sqrt(var + 1e-5)
        features.append(feat.to(device))

    input_features, attention_mask = _pad_batch(features)
    b, t, c = input_features.shape
    t = (t // stride) * stride
    input_features = input_features[:, :t, :]
    attention_mask = attention_mask[:, :t]
    input_features = input_features.reshape(b, t // stride, c * stride)
    attention_mask = attention_mask[:, 1::stride]
    return {"input_features": input_features, "attention_mask": attention_mask}


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _normalize(latents: torch.Tensor, models: dict) -> torch.Tensor:
    """Normalize latents for the diffusion model."""
    if not models["latent_norm_initialized"]:
        return latents
    return (
        (latents.float() - models["latent_norm_mean"]) / models["latent_norm_std"]
    ).to(latents.dtype)


def _denormalize(latents: torch.Tensor, models: dict) -> torch.Tensor:
    """Denormalize latents after diffusion sampling."""
    if not models["latent_norm_initialized"]:
        return latents
    return (
        latents.float() * models["latent_norm_std"] + models["latent_norm_mean"]
    ).to(latents.dtype)


# ---------------------------------------------------------------------------
# Inference: single chunk
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _separate_chunk(
    wav: torch.Tensor,  # [1, T] at 16 kHz, already normalized
    num_steps: int,
    models: dict,
    device: torch.device,
) -> torch.Tensor:
    """Run separation on a single chunk. Returns [2, T_audio] at model sample rate."""
    latent_dim = models["latent_dim"]

    noisy_ssl = extract_fbank_features([wav.view(-1)], device)
    features, pred0, pred1 = models["ssl_encoder"](
        noisy_ssl["input_features"], noisy_ssl["attention_mask"]
    )

    predicted_latents = torch.cat([pred0, pred1], dim=-1)
    conditioning = torch.cat([_normalize(predicted_latents, models), features], dim=-1)

    seq_len = conditioning.shape[1]
    scheduler = models["scheduler"]
    scheduler.set_timesteps(num_steps, device=device)
    latents = torch.randn(
        (1, seq_len, latent_dim * 2), device=device, dtype=conditioning.dtype
    )
    for t in scheduler.timesteps:
        t_batch = torch.full((1,), int(t.item()), device=device, dtype=torch.long)
        latents = scheduler.step(
            models["diffusion_head"](latents, t_batch, conditioning), t, latents
        ).prev_sample

    latents = _denormalize(latents, models)
    spk1 = models["vae_decoder"](latents[:, :, :latent_dim].transpose(1, 2)).squeeze(
        0
    )  # [1, T]
    spk2 = models["vae_decoder"](latents[:, :, latent_dim:].transpose(1, 2)).squeeze(0)
    return torch.cat([spk1, spk2], dim=0)  # [2, T]


# ---------------------------------------------------------------------------
# Speaker permutation solving via waveform cosine similarity
# ---------------------------------------------------------------------------

def _channel_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute normalized cosine similarity between two audio channels."""
    a, b = a.reshape(-1), b.reshape(-1)
    a, b = a - a.mean(), b - b.mean()
    denom = torch.linalg.norm(a) * torch.linalg.norm(b)
    return float(torch.dot(a, b) / denom) if float(denom) > 1e-8 else 0.0


def _maybe_swap(
    prev_overlap: torch.Tensor, curr_chunk: torch.Tensor, overlap_samples: int
) -> tuple[torch.Tensor, bool]:
    """Check if speakers should be swapped based on overlap region similarity."""
    if overlap_samples <= 0 or prev_overlap.shape[0] != 2 or curr_chunk.shape[0] != 2:
        return curr_chunk, False
    curr_ov = curr_chunk[:, :overlap_samples]
    direct = _channel_similarity(prev_overlap[0], curr_ov[0]) + _channel_similarity(
        prev_overlap[1], curr_ov[1]
    )
    swapped = _channel_similarity(prev_overlap[0], curr_ov[1]) + _channel_similarity(
        prev_overlap[1], curr_ov[0]
    )
    if swapped > direct:
        return curr_chunk[[1, 0], :], True
    return curr_chunk, False


# ---------------------------------------------------------------------------
# Inference: chunked with streaming
# ---------------------------------------------------------------------------

@torch.inference_mode()
def separate_chunked(
    wav: torch.Tensor,  # [1, T] at original sample_rate
    sample_rate: int,
    num_steps: int,
    models: dict,
    device: torch.device,
    chunk_seconds: float = 20.0,
    overlap_seconds: float = 10.0,
) -> tuple[torch.Tensor, int]:
    """Separate a long waveform in overlapping chunks with speaker realignment."""
    out_sr = models["sample_rate"]

    # resample to 16 kHz
    if sample_rate != SAMPLE_RATE_IN:
        wav_16k = torchaudio.functional.resample(wav, sample_rate, SAMPLE_RATE_IN)
    else:
        wav_16k = wav
    wav_16k = wav_16k.to(device)

    chunk_samples = int(chunk_seconds * SAMPLE_RATE_IN)
    total_samples = wav_16k.shape[-1]

    if total_samples <= chunk_samples:
        # single-shot inference
        max_val = wav_16k.abs().max().clamp_min(1e-6)
        wav_norm = torch.nn.functional.pad(0.9 * wav_16k / max_val, (160, 160))
        separated = _separate_chunk(wav_norm, num_steps, models, device)
        return separated, out_sr

    # chunked streaming inference
    overlap_samples_in = int(overlap_seconds * SAMPLE_RATE_IN)
    hop_samples = chunk_samples - overlap_samples_in
    starts = list(range(0, total_samples, hop_samples))

    stitched: torch.Tensor | None = None
    prev_end_in = 0

    for idx, start in enumerate(starts):
        end = min(start + chunk_samples, total_samples)
        chunk = wav_16k[:, start:end]
        max_val = chunk.abs().max().clamp_min(1e-6)
        chunk_norm = torch.nn.functional.pad(0.9 * chunk / max_val, (160, 160))

        pred = _separate_chunk(chunk_norm, num_steps, models, device)  # [2, T_out]

        # match output length to input length (resampling ratio)
        target_out = max(1, round((end - start) * out_sr / SAMPLE_RATE_IN))
        if pred.shape[-1] > target_out:
            pred = pred[:, :target_out]
        elif pred.shape[-1] < target_out:
            pad = torch.zeros(2, target_out - pred.shape[-1], device=device)
            pred = torch.cat([pred, pad], dim=-1)

        if stitched is None:
            stitched = pred
            prev_end_in = end
            continue

        overlap_in = max(0, prev_end_in - start)
        overlap_out = max(
            0,
            min(
                round(overlap_in * out_sr / SAMPLE_RATE_IN),
                stitched.shape[-1],
                pred.shape[-1],
            ),
        )

        if overlap_out > 0:
            pred, _ = _maybe_swap(stitched[:, -overlap_out:], pred, overlap_out)
            fade = torch.linspace(0.0, 1.0, overlap_out, device=device).unsqueeze(0)
            blended = (
                stitched[:, -overlap_out:] * (1 - fade) + pred[:, :overlap_out] * fade
            )
            stitched = torch.cat(
                [stitched[:, :-overlap_out], blended, pred[:, overlap_out:]], dim=-1
            )
        else:
            stitched = torch.cat([stitched, pred], dim=-1)

        prev_end_in = end

    return stitched, out_sr


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def collect_wavs(input_dir: Path) -> list[Path]:
    """Collect all .wav files from a directory."""
    wavs = sorted(
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".wav"
    )
    if not wavs:
        raise FileNotFoundError(f"No .wav files found in: {input_dir}")
    return wavs


def mux_video_with_audio(
    input_video: Path, input_wav: Path, output_video: Path
) -> None:
    """Mux separated audio back into video using ffmpeg."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError(
            "ffmpeg is required for --output-video but was not found in PATH."
        )
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(input_video),
        "-i",
        str(input_wav),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output_video),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()[-2000:]}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DialogueSidon separation on wav folders or a single video/audio file. "
        "Loads models from Hugging Face Hub (sarulab-speech/DialogueSidon)."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing input wav files (batch mode).",
    )
    input_group.add_argument(
        "--input-video",
        type=Path,
        help="Input video/audio file path (single-file mode).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory where predicted wavs are written (batch mode).",
    )
    parser.add_argument(
        "--output-wav",
        type=Path,
        help="Output separated wav path (single-file mode).",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        help="Optional output video with replaced audio (single-file mode).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=30,
        help="Number of diffusion sampling steps.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=120.0,
        help="Chunk duration in seconds.",
    )
    parser.add_argument(
        "--overlap-seconds",
        type=float,
        default=10.0,
        help="Overlap duration in seconds.",
    )
    args = parser.parse_args()

    if args.input_dir is not None:
        if not args.input_dir.is_dir():
            raise NotADirectoryError(f"Input directory not found: {args.input_dir}")
        if args.output_dir is None:
            parser.error("--output-dir is required with --input-dir")
        if args.output_wav is not None or args.output_video is not None:
            parser.error("--output-wav/--output-video can only be used with --input-video")
    else:
        if args.input_video is None or not args.input_video.is_file():
            raise FileNotFoundError(f"Input video not found: {args.input_video}")
        if args.output_wav is None:
            parser.error("--output-wav is required with --input-video")
        if args.output_dir is not None:
            parser.error("--output-dir can only be used with --input-dir")

    device = torch.device(args.device)
    models = load_models_from_hub(device)

    print(f"device: {device}")
    print(f"repo_id: {REPO_ID}")
    print(
        f"chunk={args.chunk_seconds}s, overlap={args.overlap_seconds}s, steps={args.num_steps}"
    )

    def _process(wav: torch.Tensor, sr: int) -> tuple[torch.Tensor, int]:
        # import pdb; pdb.set_trace()
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        elif wav.ndim == 2:
            # if wav.shape[0] > 2 or (wav.shape[1] > 2 and wav.shape[0] <= 2):
            #     wav = wav.T
            wav = wav.mean(dim=0, keepdim=True)
        return separate_chunked(
            wav,
            sr,
            num_steps=args.num_steps,
            models=models,
            device=device,
            chunk_seconds=args.chunk_seconds,
            overlap_seconds=args.overlap_seconds,
        )

    if args.input_dir is not None:
        input_wavs = collect_wavs(args.input_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"num input wavs: {len(input_wavs)}")
        for input_wav in input_wavs:
            try:
                wav, sr = torchaudio.load(str(input_wav))
                predicted, out_sr = _process(wav, sr)
                output_wav = args.output_dir / input_wav.name
                torchaudio.save(str(output_wav), predicted.cpu(), sample_rate=out_sr)
                # print(f"{input_wav} -> {output_wav}")
            except Exception as e:
                print(f"Error processing {input_wav}: {e}", file=sys.stderr)
                continue
    else:
        wav, sr = torchaudio.load(str(args.input_video))
        predicted, out_sr = _process(wav, sr)
        args.output_wav.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(args.output_wav), predicted.cpu(), sample_rate=out_sr)
        # print(f"{args.input_video} -> {args.output_wav}")

        if args.output_video is not None:
            args.output_video.parent.mkdir(parents=True, exist_ok=True)
            mux_video_with_audio(args.input_video, args.output_wav, args.output_video)
            # print(f"{args.input_video} + {args.output_wav} -> {args.output_video}")


if __name__ == "__main__":
    main()
