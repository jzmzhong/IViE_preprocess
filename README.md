
## Step 0: Data Preparation and Analysis

```bash
bash 01_ivie_download_extract.sh # download to archives, and extract to raw
python 02_reorganise_files.py # concat audio clips for conversation and spontaneous speech tasks, and rename for all audio clips
python 03_ivie_stats.py # calculate the metainfo of the dataset, e.g. #speaker/speaker-pair/hours per task or/and per accent
```

## Step 1: Speech Enhancement and Speaker Separation

```bash
bash 11_sidon_setup.sh # setup Sidon and DialogueSidon environment and inference scripts
bash 12_speech_enhancement_sidon.sh # speech enhancement by Sidon/DialogueSidon for all five tasks
bash 13_two_channel_split.sh # split the two-channel audio file after speaker separation to two mono-channel audio files
```

## Step 2: ASR Transcription and Alignment

```bash
bash 21_whisperx_setup.sh # setup WhisperX environment and inference script
bash 22_transcribe_align.sh # inference whisperX to get transcription and alignments
python 23_sent_segment.py # segment audio and transcription files into sentence-level
```

## Step 3: Automatic Speaker Labelling

```bash
bash 31_speaker_label.py # automatic labelling of conversation speech tasks channels, using retold_passages as reference
```

## File Structure

```bash
├── archives # 01
├── raw # 01
├── enhanced_Sidon # 12 (only sentences, read_passages, and retold_passages tasks)
├── enhanced_separated_DialogueSidon # 12 (only free_conversation and map_task tasks)
├── enhanced_separated_DialogueSidon_channel_split # 13 (only free_conversation and map_task tasks)
├── asr_whisperx # 22
└── segmented
    ├── txt # 23
    ├── wav # 23
    └── speaker_labeling_results # 31
```

- each folder (except for `archives`) is in the structure of `./{accent}/{task}/*.wav` or `./{accent}/{task}/*.txt`
- TTS (Text-to-Speech) or Acent Identification (AID) data: `segmented`
- Full-duplex data: `enhanced_separated_DialogueSidon`