from pathlib import Path
import argparse
import re
import librosa
import numpy as np
import soundfile as sf

LIVERPOOL_PAIRS = [
    ("ch", "rb"),
    ("ds", "lm"),
    ("gw", "px"),
    ("ph", "sb"),
    ("tr", "lp"),
    ("nt", "js"),
]

CARDIFF_PAIRS = [
    ("nc", "kv"),
    ("hw", "mk"),
    ("hr", "ld"),
    ("sc", "hd"),
    ("xt", "er"),
    ("lh", "kt"),
]

SKIP_FILES = [
    "c-c2-f1_3.wav",
    "c-c2-m1_5.wav",
    "d-c2-f1_2.wav",
    "d-c2-m1_3.wav",
    "n-c2-f1_4.wav",
    "n-c2-m1_3.wav",
] # duplicate files in Cambridge, Dublin, and Newcastle free_conversation


def extract_paired_speaker(acc_idx: str, spk_idx_1: str) -> str:
    """
    Given a speaker index and accent index, return the paired speaker index.
    """
    if acc_idx == "s":  # Liverpool
        for spk_1, spk_2 in LIVERPOOL_PAIRS:
            if spk_1 == spk_idx_1:
                return spk_2
            elif spk_2 == spk_idx_1:
                return spk_1
    elif acc_idx == "w":  # Cardiff
        for spk_1, spk_2 in CARDIFF_PAIRS:
            if spk_1 == spk_idx_1:
                return spk_2
            elif spk_2 == spk_idx_1:
                return spk_1
    print(f"Warning: Could not find paired speaker for accent '{acc_idx}' and speaker '{spk_idx_1}'")
    return None

def breakdown_filename(task: str, accent: str, stem: str) -> tuple[str, str]:
    """
    Extract content_id and speaker or speaker-pair ID(s) from IViE filename stem.
    """
    if "-" in stem: # new format data from 7 accent regions
        
        # check file name format: accent-content-speaker or accent-content-speaker_pair
        parts = stem.split("-")
        if len(parts) != 3:
            print(f"Warning: Unexpected IViE new format filename stem: {stem}")
            return None
        acc_idx, content_idx, spk_stem = parts
        
        # extract speaker or speaker pair
        if re.match(r"^[fm][0-9]_[0-9]$", spk_stem): # conversation - speaker pair
            gender, spk_idx_1, _, spk_idx_2 = list(spk_stem)
            return content_idx, f"{gender}{spk_idx_1}_{gender}{spk_idx_2}"
        elif re.match(r"^[fm][0-9][fm][0-9]$", spk_stem): # conversation - speaker pair of different genders
            return content_idx, f"{spk_stem[:2]}_{spk_stem[2:]}"
        elif re.match(r"^[fm][0-9]$", spk_stem): # monologue - speaker
            return content_idx, spk_stem
        else:
            print("Warning: Unexpected IViE new format filename stem:", stem)
            return None
    
    else: # old format data from 2 accent regions (Liverpool, Cardiff)
        
        # check file name format: content|accent|speaker or content|accent|speaker_pair
        if re.match(r"^[a-z][0-9][ab]?[ws][a-z]{2}$", stem):
            content_idx, acc_idx, spk_idx_1 = stem[:-3], stem[-3], stem[-2:]
            if task in ["free_conversation", "map_task"]: # conversation - speaker pair
                spk_idx_2 = extract_paired_speaker(acc_idx, spk_idx_1)
                return content_idx, f"{spk_idx_1}_{spk_idx_2}" if spk_idx_2 else None
            else: # monologue - speaker
                return content_idx, spk_idx_1
        else:
            print("Warning: Unexpected IViE old format filename stem:", stem)
            return None


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/home/s2526235/data/IViE/raw"),
        help="Path to IViE wav root (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/s2526235/data/IViE/reorganised"),
        help="Path to reorganised IViE wav root (default: %(default)s)",
    )
    args = parser.parse_args()


    for accent_dir in args.input_dir.iterdir():
        accent = accent_dir.name
        if not accent_dir.is_dir():
            continue
        
        for task_dir in accent_dir.iterdir():
            task = task_dir.name
            if not task_dir.is_dir():
                continue
            
            print(f"Processing accent: {accent}, task: {task}")
            output_dir = Path(args.output_dir) / accent / task
            output_dir.mkdir(parents=True, exist_ok=True)

            wav_files = sorted([path for path in task_dir.glob("*.wav") if path.name not in SKIP_FILES])
            wav_files_info = [breakdown_filename(task, accent, Path(filename).stem) for filename in wav_files]

            if task in ["read_passages", "sentences"]: # read speech tasks
                
                # copy and rename audio clips
                for wav_file, info in zip(wav_files, wav_files_info):
                    if not info:
                        continue
                    content_id, speaker_id = info
                    output_filename = f"{accent}-{task}-{content_id}-{speaker_id}.wav"
                    output_path = output_dir / output_filename
                    try:
                        sf.write(output_path, *librosa.load(wav_file, sr=None, mono=True))
                    except Exception as e:
                        print(f"Error processing {wav_file}: {e}")
                        
            elif task in ["free_conversation", "map_task", "retold_passages"]: # conversation and spontaneous speech tasks

                # concat audio clips
                speaker_or_pairs = set([info[1] for info in wav_files_info if info])
                for speaker_or_pair in speaker_or_pairs:
                    output_filename = f"{accent}-{task}-{speaker_or_pair}.wav"
                    output_path = output_dir / output_filename
                    input_wav_files = [wav_file for wav_file, info in zip(wav_files, wav_files_info) if info and info[1] == speaker_or_pair]
                    waveforms = []
                    sample_rate = None
                    for wav_file in input_wav_files:
                        try:
                            waveform, current_sample_rate = librosa.load(wav_file, sr=None, mono=True)
                            if sample_rate is None:
                                sample_rate = current_sample_rate
                            elif current_sample_rate != sample_rate:
                                raise ValueError(
                                    f"Incompatible sample rate in {wav_file}: "
                                    f"{current_sample_rate} != {sample_rate}"
                                )
                            waveforms.append(waveform)
                        except Exception as e:
                            print(f"Error processing {wav_file}: {e}")
                    sf.write(output_path, np.concatenate(waveforms, axis=-1).T, sample_rate)
            
            else:
                print(f"Warning: Unrecognised task '{task}' in accent '{accent}'")
