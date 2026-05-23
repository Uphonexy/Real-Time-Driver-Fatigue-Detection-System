"""
download_model.py — Download the Dlib facial landmark model for WakeMate.

Downloads shape_predictor_68_face_landmarks.dat.bz2 from the official Dlib
website, extracts it, and places it in the project root.

Usage:
    python download_model.py

The file is ~100 MB compressed.  A progress bar is shown during download.
If the .dat file already exists, the script exits immediately.
"""

import os
import sys
import bz2
import urllib.request

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
BZ2_FILE  = "shape_predictor_68_face_landmarks.dat.bz2"
DAT_FILE  = "shape_predictor_68_face_landmarks.dat"
CHUNK     = 8192


# ── Progress hook ─────────────────────────────────────────────────────────────
def _progress(block_num: int, block_size: int, total_size: int):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded * 100 / total_size)
        bar_len  = 40
        filled   = int(bar_len * pct / 100)
        bar      = "█" * filled + "░" * (bar_len - filled)
        mb_done  = downloaded / 1_048_576
        mb_total = total_size / 1_048_576
        print(
            f"\r  [{bar}] {pct:5.1f}%  {mb_done:.1f}/{mb_total:.1f} MB",
            end="",
            flush=True,
        )
    else:
        mb_done = (block_num * block_size) / 1_048_576
        print(f"\r  Downloaded {mb_done:.1f} MB…", end="", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def download():
    # Already extracted
    if os.path.exists(DAT_FILE):
        size_mb = os.path.getsize(DAT_FILE) / 1_048_576
        print(f"[OK] {DAT_FILE} already exists ({size_mb:.1f} MB). Nothing to do.")
        return

    # Download
    print(f"Downloading {MODEL_URL}")
    print("This is ~99 MB — please wait…\n")
    try:
        urllib.request.urlretrieve(MODEL_URL, BZ2_FILE, reporthook=_progress)
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        sys.exit(1)
    print()  # newline after progress bar

    # Extract
    print(f"Extracting {BZ2_FILE}…")
    try:
        with bz2.open(BZ2_FILE, "rb") as f_in, open(DAT_FILE, "wb") as f_out:
            while True:
                chunk = f_in.read(CHUNK)
                if not chunk:
                    break
                f_out.write(chunk)
    except Exception as e:
        print(f"[ERROR] Extraction failed: {e}")
        sys.exit(1)

    # Cleanup
    try:
        os.remove(BZ2_FILE)
    except Exception:
        pass

    size_mb = os.path.getsize(DAT_FILE) / 1_048_576
    print(f"[OK] {DAT_FILE} ready ({size_mb:.1f} MB).")
    print("You can now launch WakeMate with:  python dashboard.py")


if __name__ == "__main__":
    download()
