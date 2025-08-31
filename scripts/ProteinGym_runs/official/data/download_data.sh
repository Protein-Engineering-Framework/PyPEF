#!/bin/bash

VERSION="v1.3"
FILES=(
  "DMS_substitutions.csv"
  "DMS_msa_files.zip"
  "ProteinGym_AF2_structures.zip"
  "cv_folds_singles_substitutions.zip"
  "cv_folds_multiples_substitutions.zip"
)

for FILENAME in "${FILES[@]}"; do
  echo "Downloading $FILENAME..."
  curl -O "https://marks.hms.harvard.edu/proteingym/ProteinGym_${VERSION}/${FILENAME}"

  # Only unzip and delete if it's a zip file
  if [[ "$FILENAME" == *.zip ]]; then
    echo "Unzipping $FILENAME..."
    unzip "$FILENAME"

    echo "Removing $FILENAME..."
    rm "$FILENAME"
  fi
done