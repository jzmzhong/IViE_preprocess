from pathlib import Path
import argparse
import soundfile as sf

def get_duration_seconds(wav_path: Path) -> float:
    info = sf.info(str(wav_path))
    return info.frames / info.samplerate
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/home/s2526235/data/IViE/reorganised"),
        help="Path to reorganised IViE wav root (default: %(default)s)",
    )
    args = parser.parse_args()

    # calculate the number of utterances and the duration for each accent and task
    accent_task_stats = {}
    for accent_dir in args.input_dir.iterdir():
        accent = accent_dir.name
        if not accent_dir.is_dir():
            continue
        
        for task_dir in accent_dir.iterdir():
            task = task_dir.name
            if not task_dir.is_dir():
                continue
            
            wav_files = sorted([path for path in task_dir.glob("*.wav")])
            num_utterances = len(wav_files)
            total_duration = sum(get_duration_seconds(wav_file) for wav_file in wav_files)
            
            accent_task_stats[(accent, task)] = {
                "num_utterances": num_utterances,
                "total_duration": total_duration,
            }
    
    # print the results
    for (accent, task), stats in accent_task_stats.items():
        print(f"Accent: {accent}, Task: {task}, Num Utterances: {stats['num_utterances']}, Total Duration (min): {stats['total_duration'] / 60:.2f}")
    print(f"Total number of utterances: {sum(stats['num_utterances'] for stats in accent_task_stats.values())}")
    print(f"Total duration (hr): {sum(stats['total_duration'] for stats in accent_task_stats.values()) / 3600:.2f}")