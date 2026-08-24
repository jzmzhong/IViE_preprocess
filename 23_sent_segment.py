
import os
import json
import librosa
import soundfile as sf

DATA_ROOT = "/home/s2526235/data/IViE"
ASR_ROOT = f"{DATA_ROOT}/asr_whisperx"

WAV_ROOT = f"{DATA_ROOT}/enhanced_separated_DialogueSidon_split"
TASKS = ["free_conversation", "map_task"]
SEGMENT = True

# WAV_ROOT = f"{DATA_ROOT}/enhanced_Sidon"
# TASKS = ["read_passages", "retold_passages"]
# SEGMENT = True

# WAV_ROOT = f"{DATA_ROOT}/enhanced_Sidon"
# TASKS = ["sentences"]
# SEGMENT = False

# Audio settings
SIL_BEFORE = 50 # ms of silence before the segment
SIL_AFTER = 50  # ms of silence after the segment
TAR_SR = 24000  # output sample rate

# Create output directories
OUTPUT_WAV_DIR = f"{DATA_ROOT}/segmented/wav"
OUTPUT_TXT_DIR = f"{DATA_ROOT}/segmented/txt"
os.makedirs(OUTPUT_WAV_DIR, exist_ok=True)
os.makedirs(OUTPUT_TXT_DIR, exist_ok=True)

# Process each accent folder
for accent in os.listdir(ASR_ROOT):
    accent_path = os.path.join(ASR_ROOT, accent)
    if not os.path.isdir(accent_path):
        continue
    
    # Process each content folder
    for task in os.listdir(accent_path):
        if task not in TASKS:
            continue
        task_path = os.path.join(accent_path, task)
        if not os.path.isdir(task_path):
            continue
        output_wav_dir = os.path.join(OUTPUT_WAV_DIR, accent, task)
        output_txt_dir = os.path.join(OUTPUT_TXT_DIR, accent, task)
        os.makedirs(output_wav_dir, exist_ok=True)
        os.makedirs(output_txt_dir, exist_ok=True)
        print(f"Processing accent: {accent}, task: {task}")

        # Process each json file
        for json_file in os.listdir(task_path):
            if not json_file.endswith('.json'):
                continue
            # print(f"Processing JSON file: {json_file}")

            # Read JSON file with timestamps
            json_path = os.path.join(task_path, json_file)
            with open(json_path, 'r') as f:
                alignment = json.load(f)
            
            # Load corresponding wav file
            wav_filename = json_file.replace('.json', '.wav')
            wav_path = os.path.join(WAV_ROOT, accent, task, wav_filename)
            if not os.path.exists(wav_path):
                print(f"Warning: {wav_path} not found")
                continue
            y, _ = librosa.load(wav_path, sr=TAR_SR, mono=True)

            if SEGMENT:
                # Segment audio based on timestamps
                for i, segment in enumerate(alignment['segments']):
                    start_time = float(segment.get('start', 0))
                    end_time = float(segment.get('end', 0))
                    text = segment.get('text', '').strip()
                    
                    # Convert time to samples
                    start_time = max(0, start_time - SIL_BEFORE / 1000)
                    end_time = min(len(y) / TAR_SR, end_time + SIL_AFTER / 1000)
                    start_sample = int(start_time * TAR_SR)
                    end_sample = int(end_time * TAR_SR)
                    
                    # Extract segment
                    segment = y[start_sample:end_sample]
                    
                    # Save wav segment
                    output_wav_name = wav_filename.replace('.wav', f'-{str(i+1).zfill(3)}.wav')
                    output_wav_path = os.path.join(output_wav_dir, output_wav_name)
                    sf.write(output_wav_path, segment, TAR_SR)
                    
                    # Save text segment
                    output_txt_name = wav_filename.replace('.wav', f'-{str(i+1).zfill(3)}.txt')
                    output_txt_path = os.path.join(output_txt_dir, output_txt_name)
                    with open(output_txt_path, 'w') as f:
                        f.write(str(text))
            else:

                if len(alignment['segments']) == 0:
                    print("Warning: No segments found in alignment for file:", json_file)
                    continue
                start_time = float(alignment['segments'][0].get('start', 0))
                end_time = float(alignment['segments'][-1].get('end', 0))
                text = ' '.join([seg.get('text', '').strip() for seg in alignment['segments']])

                # Convert time to samples
                start_time = max(0, start_time - SIL_BEFORE / 1000)
                end_time = min(len(y) / TAR_SR, end_time + SIL_AFTER / 1000)
                start_sample = int(start_time * TAR_SR)
                end_sample = int(end_time * TAR_SR)

                # Extract segment
                segment = y[start_sample:end_sample]

                # Save wav segment
                output_wav_path = os.path.join(output_wav_dir, wav_filename)
                sf.write(output_wav_path, segment, TAR_SR)

                # Save text segment
                output_txt_name = wav_filename.replace('.wav', '.txt')
                output_txt_path = os.path.join(output_txt_dir, output_txt_name)
                with open(output_txt_path, 'w') as f:
                    f.write(str(text))

            # print(f"Processed: {wav_filename}")
