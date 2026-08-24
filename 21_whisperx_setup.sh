#!/usr/bin/env bash

CUR_DIR=`pwd`

REPO_PARENT_DIR="$HOME/tools"
if [ ! -d "$REPO_PARENT_DIR" ]; then
  mkdir -p "$REPO_PARENT_DIR"
fi

# clone WhisperX repo
cd "$REPO_PARENT_DIR"
if [ ! -d whisperX ]; then
  echo "Cloning WhisperX repository"
  git clone https://github.com/m-bain/whisperX.git
fi

# uv install WhisperX dependencies
cd whisperX
uv sync --all-extras --dev

# copy inference scripts
cp "$CUR_DIR/infer_whisperx.py" "$REPO_PARENT_DIR/whisperX"