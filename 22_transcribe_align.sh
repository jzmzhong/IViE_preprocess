#!/usr/bin/env bash

source ~/.bashrc
WHISPERX_REPO=$HOME/tools/whisperX

cd $WHISPERX_REPO
export LD_LIBRARY_PATH=$WHISPERX_REPO/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib/:$LD_LIBRARY_PATH

for task in free_conversation map_task; do
    uv run python infer_whisperx.py \
        --input-root $HOME/data/IViE/enhanced_separated_DialogueSidon_split \
        --output-root $HOME/data/IViE/asr_whisperx \
        --audio-glob "*/$task/*.wav" \
        --model large-v3 \
        --language en \
        --batch-size 16 \
        --device cuda
done

for task in sentences read_passages retold_passages; do
    uv run python infer_whisperx.py \
        --input-root $HOME/data/IViE/enhanced_Sidon \
        --output-root $HOME/data/IViE/asr_whisperx \
        --audio-glob "*/$task/*.wav" \
        --model large-v3 \
        --language en \
        --batch-size 16 \
        --device cuda
done

# run this script on cluster
# e.g. run_l40s_cdt 22_transcribe_align.sh
