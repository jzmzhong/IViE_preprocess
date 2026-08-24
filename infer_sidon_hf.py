from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio
import transformers
import spaces
from huggingface_hub import hf_hub_download

fe_path = hf_hub_download("sarulab-speech/sidon-v0.1", filename="feature_extractor_cuda.pt")
decoder_path = hf_hub_download("sarulab-speech/sidon-v0.1", filename="decoder_cuda.pt")

preprocessor = transformers.SeamlessM4TFeatureExtractor.from_pretrained(
    "facebook/w2v-bert-2.0"
)

_model_cache: dict[str, torch.jit.ScriptModule] = {}


def _load_models(device: str = "cuda") -> tuple[torch.jit.ScriptModule, torch.jit.ScriptModule]:
    """Load and cache Sidon modules on the target device."""
    cache_key = str(device)
    if cache_key in _model_cache:
        return _model_cache[cache_key]["fe"], _model_cache[cache_key]["decoder"]

    fe = torch.jit.load(fe_path, map_location=device).to(device)
    decoder = torch.jit.load(decoder_path, map_location=device).to(device)
    _model_cache[cache_key] = {"fe": fe, "decoder": decoder}
    return fe, decoder


@torch.inference_mode()
def _restore_waveform(
    sample_rate: int,
    waveform: torch.Tensor | np.ndarray,
    device: str = "cuda",
) -> torch.Tensor:
    """Restore a noisy waveform and return mono float tensor at 48 kHz."""
    fe, decoder = _load_models(device=device)

    if not isinstance(waveform, torch.Tensor):
        waveform = torch.tensor(waveform, dtype=torch.float32)
    else:
        waveform = waveform.float()

    if waveform.ndim == 2:
        # Handle both [channels, time] and [time, channels].
        if waveform.shape[0] <= 8 and waveform.shape[1] > waveform.shape[0]:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.mean(dim=1)

    if waveform.ndim != 1:
        waveform = waveform.view(-1)

    max_abs = waveform.abs().max().clamp_min(1e-6)
    waveform = 0.9 * (waveform / max_abs)
    target_n_samples = int(48_000 / sample_rate * waveform.shape[0])

    waveform = waveform.view(1, -1)
    wav = torchaudio.functional.highpass_biquad(waveform, sample_rate, 50)
    wav_16k = torchaudio.functional.resample(wav, sample_rate, 16_000)

    restoreds = []
    feature_cache = None
    wav_16k = torch.nn.functional.pad(wav_16k, (0, 24_000))
    for chunk in wav_16k.view(-1).split(16_000 * 96):
        inputs = preprocessor(
            torch.nn.functional.pad(chunk, (160, 160)), return_tensors="pt"
        ).to("cpu")
        feature = fe(inputs["input_features"].to(device))["last_hidden_state"]
        if feature_cache is not None:
            feature = torch.cat([feature_cache, feature], dim=1)
        restoreds.append(decoder(feature.transpose(1, 2)).view(-1)[:-960])
        feature_cache = feature[:, -1:]

    restored_wav = torch.cat(restoreds, dim=0)[:target_n_samples]
    return restored_wav


def collect_wavs(input_dir: Path) -> list[Path]:
    """Collect all wav files from a directory."""
    wavs = sorted(
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".wav"
    )
    if not wavs:
        raise FileNotFoundError(f"No .wav files found in: {input_dir}")
    return wavs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Sidon speech restoration on wav folders or a single wav file. "
        )
    )
    
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing input wav files (batch mode).",
    )
    input_group.add_argument(
        "--input-wav",
        type=Path,
        help="Input wav file path (single-file mode).",
    )
    
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--output-dir",
        type=Path,
        help="Directory where restored wavs are written (batch mode).",
    )
    output_group.add_argument(
        "--output-wav",
        type=Path,
        help="Output restored wav path (single-file mode).",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device used for inference.",
    )
    args = parser.parse_args()
    

    if args.input_dir is not None:
        if not args.input_dir.is_dir():
            raise NotADirectoryError(f"Input directory not found: {args.input_dir}")
        if args.output_dir is None:
            parser.error("--output-dir is required with --input-dir")
        if args.output_wav is not None:
            parser.error("--output-wav can only be used with --input-wav")
    else:
        if args.input_wav is None or not args.input_wav.is_file():
            raise FileNotFoundError(f"Input wav not found: {args.input_wav}")
        if args.output_wav is None:
            parser.error("--output-wav is required with --input-wav")
        if args.output_dir is not None:
            parser.error("--output-dir can only be used with --input-dir")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")

    print(f"device: {args.device}")

    if args.input_dir is not None:
        input_wavs = collect_wavs(args.input_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"#input wavs: {len(input_wavs)}")
        for input_wav in input_wavs:
            try:
                wav, sr = torchaudio.load(str(input_wav))
                restored = _restore_waveform(sr, wav, device=args.device)
                output_wav = args.output_dir / input_wav.name
                torchaudio.save(str(output_wav), restored.unsqueeze(0).cpu(), sample_rate=48_000)
                # print(f"{input_wav} -> {output_wav}")
            except Exception as e:
                print(f"Error processing {input_wav}: {e}", file=sys.stderr)
                continue
    else:
        wav, sr = torchaudio.load(str(args.input_wav))
        restored = _restore_waveform(sr, wav, device=args.device)
        args.output_wav.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(args.output_wav), restored.unsqueeze(0).cpu(), sample_rate=48_000)
        # print(f"{args.input_wav} -> {args.output_wav}")

if __name__ == "__main__":
    main()