@echo off
setlocal enabledelayedexpansion
title DeepGuard AI - Push to GitHub
echo ====================================================================
echo   DEEPGUARD AI - GITHUB UPLOAD AUTOMATOR
echo   Target Account: divyeshs-collab
echo   Target Repository: deepguard-ai
echo ====================================================================

cd /d "%~dp0"

echo [*] Staging all project files...
git add .

echo [*] Committing codebase...
git commit -m "Initial commit: DeepGuard AI Real-Time Call Deepfake Audio Detection Sentinel"

echo [*] Setting primary branch to main...
git branch -M main

echo [*] Checking if remote origin exists...
git remote get-url origin >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [*] Creating GitHub repository 'deepguard-ai' using GitHub CLI (gh)...
    gh repo create deepguard-ai --public --source=. --remote=origin --push
) else (
    echo [*] Remote origin already configured. Pushing to origin main...
    git push -u origin main
)

echo.
echo ====================================================================
echo   UPLOAD COMPLETE!
echo   Your repository is live at:
echo   https://github.com/divyeshs-collab/deepguard-ai
echo ====================================================================
pause
