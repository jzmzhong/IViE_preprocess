#!/usr/bin/env bash

IN_DIR=$HOME/data/IViE/enhanced_separated_DialogueSidon
OUT_DIR=$HOME/data/IViE/enhanced_separated_DialogueSidon_split
mkdir -p "$OUT_DIR"

shopt -s globstar

for file in "$IN_DIR"/*/{free_conversation,map_task}/*.wav; do
  filename=$(basename "$file" .wav)
  relative_path="${file#$IN_DIR/}"
  relative_dir=$(dirname "$relative_path")
  mkdir -p "$OUT_DIR/$relative_dir"
  ffmpeg -i "$file" -map_channel 0.0.0 "$OUT_DIR/$relative_dir/${filename}-ch1.wav" -map_channel 0.0.1 "$OUT_DIR/$relative_dir/${filename}-ch2.wav"
done