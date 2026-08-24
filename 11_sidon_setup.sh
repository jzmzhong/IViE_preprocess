#!/usr/bin/env bash

CUR_DIR=`pwd`

REPO_PARENT_DIR="$HOME/tools"
if [ ! -d "$REPO_PARENT_DIR" ]; then
  mkdir -p "$REPO_PARENT_DIR"
fi

# clone Sidon repo
cd "$REPO_PARENT_DIR"
if [ ! -d Sidon ]; then
  echo "Cloning Sidon repository"
  git clone https://github.com/sarulab-speech/Sidon.git
fi

# uv install Sidon dependencies
cd Sidon
if ! grep -q '^\[tool\.uv\.extra-build-dependencies\]$' pyproject.toml; then
  cat <<EOL >> pyproject.toml
[tool.uv.extra-build-dependencies]
flash-attn = [
    { requirement = "torch", match-runtime = true },
]
EOL
fi
uv sync

# copy inference scripts
cp "$CUR_DIR/infer_sidon_hf.py" "$REPO_PARENT_DIR/Sidon"
cp "$CUR_DIR/infer_dialoguesidon_hf.py" "$REPO_PARENT_DIR/Sidon"