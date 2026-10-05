#!/usr/bin/env bash
# DeepGuard AI - Push to GitHub Script
set -e

REPO_NAME="${1:-deepguard-ai}"
VISIBILITY="${2:---public}"

echo "===================================================================="
echo "  DEEPGUARD AI — GITHUB UPLOAD AUTOMATOR"
echo "  Target Repository: ${REPO_NAME}"
echo "===================================================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

git add .
git commit -m "Initial commit: DeepGuard AI Real-Time Call Deepfake Audio Detection Sentinel" || true
git branch -M main

if ! git remote get-url origin > /dev/null 2>&1; then
    echo "[*] Creating GitHub repository '${REPO_NAME}' via gh CLI..."
    gh repo create "${REPO_NAME}" "${VISIBILITY}" --source=. --remote=origin --push
else
    echo "[*] Pushing to origin main..."
    git push -u origin main
fi

echo "===================================================================="
echo "  UPLOAD COMPLETE!"
echo "===================================================================="
