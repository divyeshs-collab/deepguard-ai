"""
DeepGuard AI - Deployment Packaging Utility
Packages the complete DeepGuard AI project (codebase, trained CNN weights,
evaluation artifacts, and frontend) into a portable deployment ZIP archive
ready for instant deployment onto any remote host.
"""

import os
import zipfile
from pathlib import Path

def package_project():
    root_dir = Path(__file__).resolve().parent
    output_zip = root_dir.parent / "deepguard-ai-deployment.zip"

    exclude_patterns = [
        "__pycache__", ".pyc", ".git", "node_modules", ".DS_Store",
        ".venv", "venv", ".idea", ".vscode"
    ]

    print(f"[*] Packaging DeepGuard AI for remote host deployment...")
    print(f"[*] Source directory: {root_dir}")
    print(f"[*] Destination archive: {output_zip}")

    total_files = 0
    with zipfile.ZipFile(str(output_zip), 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(root_dir):
            # Prune excluded directories in-place
            dirs[:] = [d for d in dirs if not any(pat in d for pat in exclude_patterns)]

            for file in files:
                if any(file.endswith(ext) for ext in [".pyc", ".pyo", ".log"]):
                    continue
                file_path = Path(root) / file
                arcname = file_path.relative_to(root_dir)
                zipf.write(file_path, arcname)
                total_files += 1

    zip_size_mb = output_zip.stat().st_size / (1024 * 1024)
    print(f"[+] Successfully packaged {total_files} files into {output_zip} ({zip_size_mb:.2f} MB)")
    print(f"[+] To deploy to another host:")
    print(f"    1. Copy {output_zip.name} to the remote server/host.")
    print(f"    2. Unzip and run 'docker compose up -d' or 'bash deploy.sh'")

if __name__ == "__main__":
    package_project()
