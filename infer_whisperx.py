"""Run WhisperX transcription and alignment for all IViE channel WAV files."""

import argparse
import gc
import json
from pathlib import Path

import torch
import whisperx

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--input-root",
		type=Path,
		required=True,
	)
	parser.add_argument(
		"--output-root",
		type=Path,
		required=True,
	)
	parser.add_argument(
		"--audio-glob",
		default="*/*/*.wav",
		help="Glob pattern for WAV files relative to --input-root (default: %(default)s).",
	)
	parser.add_argument(
		"--model",
		default="large-v3",
		help="WhisperX model name (default: %(default)s).",
	)
	parser.add_argument(
		"--language",
		default="en",
		help="Language code used for transcription (default: %(default)s).",
	)
	parser.add_argument("--batch-size", type=int, default=16)
	parser.add_argument("--device", default="cuda")
	parser.add_argument("--compute-type", default="float16")
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Reprocess files whose output JSON already exists.",
	)
	return parser.parse_args()


def output_path(audio_path: Path, input_root: Path, output_root: Path) -> Path:
	relative_path = audio_path.relative_to(input_root)
	return output_root / relative_path.parent / f"{audio_path.stem}.json"


def main():
	args = parse_args()
	audio_paths = sorted(args.input_root.glob(args.audio_glob))
	if not audio_paths:
		raise FileNotFoundError(
			f"No WAV files found under {args.input_root / args.audio_glob}"
		)

	model = whisperx.load_model(
		args.model,
		args.device,
		compute_type=args.compute_type,
		language=args.language,
	)
	alignment_models = {}

	for index, audio_path in enumerate(audio_paths, start=1):
		destination = output_path(audio_path, args.input_root, args.output_root)
		if destination.exists() and not args.overwrite:
			print(f"[{index}/{len(audio_paths)}] Skipping existing {destination}")
			continue

		print(f"[{index}/{len(audio_paths)}] Processing {audio_path}")
		audio = whisperx.load_audio(str(audio_path))
		result = model.transcribe(
			audio,
			batch_size=args.batch_size,
			language=args.language,
		)

		if args.language not in alignment_models:
			alignment_models[args.language] = whisperx.load_align_model(
				language_code=args.language,
				device=args.device,
			)
		alignment_model, metadata = alignment_models[args.language]
		result = whisperx.align(
			result["segments"],
			alignment_model,
			metadata,
			audio,
			args.device,
			return_char_alignments=False,
		)

		destination.parent.mkdir(parents=True, exist_ok=True)
		with destination.open("w", encoding="utf-8") as output_file:
			json.dump(result, output_file, ensure_ascii=False)
		print(f"[{index}/{len(audio_paths)}] Wrote {destination}")

		del audio, result
		gc.collect()
		if args.device.startswith("cuda"):
			torch.cuda.empty_cache()


if __name__ == "__main__":
	main()