#!/bin/bash
set -e

# === CONFIG ===
USER="bdc-ayushk"       # your GitHub username
REPO="test_repo"             # repo name
TOKEN="github_pat_11BSE37FY0HEcgflZ9HbrY_vHUtadLqR43K7J3mY4TEmD5ciPhfQj6roupdaQocjChYMWH67"     # your GitHub PAT
TAGS=("CTranslate2" "torch_arm64")   # tags to fetch

# === SCRIPT ===
for TAG in "${TAGS[@]}"; do
  echo "Downloading tag: $TAG"
  curl -L -H "Authorization: token $TOKEN" \
       -o "${TAG}.tar.gz" \
       "https://github.com/${USER}/${REPO}/archive/refs/tags/${TAG}.tar.gz"

  echo "Extracting: ${TAG}.tar.gz"
  mkdir -p "${TAG}"
  tar -xvzf "${TAG}.tar.gz" -C "${TAG}" --strip-components=1
done

echo "✅ All tags downloaded and extracted."
