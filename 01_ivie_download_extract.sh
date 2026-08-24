#!/usr/bin/env bash

set -euo pipefail

ROOT="$HOME/data/IViE"
RAW="$ROOT/raw"
ARCHIVES="$ROOT/archives"
BASE_NEW="https://www.phon.ox.ac.uk/files/apps/IViE/packages"
BASE_OLD="http://www.phon.ox.ac.uk/old_IViE/packages"

mkdir -p "$ARCHIVES"

regions=(Belfast Bradford Cambridge Dublin Leeds London Newcastle Cardiff Liverpool)
prefixes=(b p c m l j n w s)
contents=(sentences read_passages retold_passages map_task free_conversation)
new_suffixes=(s p r m c)
old_suffixes=(s p r m f)

for i in "${!regions[@]}"; do
  region="${regions[$i]}"
  prefix="${prefixes[$i]}"

  for j in "${!contents[@]}"; do
    content="${contents[$j]}"
    destination="$RAW/$region/$content"
    mkdir -p "$destination"

    if (( i < 7 )); then
      archive="${prefix}${new_suffixes[$j]}_new.tar"
      url="$BASE_NEW/$archive"
    else
      archive="${prefix}${old_suffixes[$j]}.tar.gz"
      url="$BASE_OLD/$archive"
    fi

    if [[ ! -s "$ARCHIVES/$archive" ]]; then
      echo "Downloading $archive"
      curl -fL --retry 3 --output "$ARCHIVES/$archive" "$url"
    fi

    echo "Extracting $archive -> $destination"
    tar -xf "$ARCHIVES/$archive" -C "$destination"
  done
done

echo "Downloaded: $(find "$ARCHIVES" -type f \( -name '*.tar' -o -name '*.tar.gz' \) | wc -l | tr -d ' ') archives"
echo "Created:    $(find "$RAW" -mindepth 2 -maxdepth 2 -type d | wc -l | tr -d ' ') content directories"
