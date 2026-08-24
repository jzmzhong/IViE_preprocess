#!/usr/bin/env bash

source ~/.bashrc

SIDON_REPO=$HOME/tools/Sidon

IN_DIR=$HOME/data/IViE/reorganised
OUT_DIR_1=$HOME/data/IViE/enhanced_separated_DialogueSidon
OUT_DIR_2=$HOME/data/IViE/enhanced_Sidon

if [ ! -d $OUT_DIR_1 ]; then
  mkdir -p $OUT_DIR_1
fi
if [ ! -d $OUT_DIR_2 ]; then
  mkdir -p $OUT_DIR_2
fi

echo "Running Sidon enhancement"
cd $SIDON_REPO

for subfolder in "$IN_DIR"/*; do
  echo "Processing subfolder: $subfolder"
  for subsubfolder in "$subfolder"/*; do
    echo "Processing subsubfolder: $subsubfolder"

    if [[ "$(basename "$subsubfolder")" == "free_conversation" || "$(basename "$subsubfolder")" == "map_task" ]]; then
      uv run python infer_dialoguesidon_hf.py \
        --input-dir "$subsubfolder" \
        --output-dir "$OUT_DIR_1/$(basename "$subfolder")/$(basename "$subsubfolder")"
    
    elif [[ "$(basename "$subsubfolder")" == "sentences" || "$(basename "$subsubfolder")" == "read_passages" || "$(basename "$subsubfolder")" == "retold_passages" ]]; then
      uv run python infer_sidon_hf.py \
        --input-dir "$subsubfolder" \
        --output-dir "$OUT_DIR_2/$(basename "$subfolder")/$(basename "$subsubfolder")"
    
    else
      echo "Skipping subsubfolder: $subsubfolder (not a recognised task)"
      continue
    fi
    
  done
done

# run this script on cluster
# e.g. run_l40s_cdt 12_speech_enhancement_sidon.sh
