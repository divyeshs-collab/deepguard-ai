@echo off
title DeepGuard AI - Deploy to Vercel
echo ====================================================================
echo   DEEPGUARD AI - VERCEL HOSTING DEPLOYMENT
echo ====================================================================

cd /d "%~dp0"

echo [*] Deploying DeepGuard AI Dashboard to Vercel via npx vercel...
echo [*] Note: If not logged into Vercel, it will prompt you in browser.
echo --------------------------------------------------------------------

npx vercel --prod

echo.
echo ====================================================================
echo   VERCEL DEPLOYMENT COMPLETE!
echo ====================================================================
pause
