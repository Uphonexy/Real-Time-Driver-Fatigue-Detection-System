"""
download_model.py — Download shape_predictor_68_face_landmarks.dat
Usage: python download_model.py
"""
import os, sys, bz2, urllib.request

MODEL_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
BZ2_FILE  = "shape_predictor_68_face_landmarks.dat.bz2"
DAT_FILE  = "shape_predictor_68_face_landmarks.dat"
CHUNK     = 8192

def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct    = min(100.0, downloaded * 100 / total_size)
        filled = int(40 * pct / 100)
        bar    = "█" * filled + "░" * (40 - filled)
        print(f"\r  [{bar}] {pct:5.1f}%  {downloaded/1_048_576:.1f}/{total_size/1_048_576:.1f} MB",
              end="", flush=True)

def download():
    if os.path.exists(DAT_FILE):
        print(f"[OK] {DAT_FILE} already exists. Nothing to do.")
        return
    print(f"Downloading {MODEL_URL}\nThis is ~99 MB — please wait…\n")
    try:
        urllib.request.urlretrieve(MODEL_URL, BZ2_FILE, reporthook=_progress)
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}"); sys.exit(1)
    print()
    print(f"Extracting {BZ2_FILE}…")
    try:
        with bz2.open(BZ2_FILE, "rb") as fin, open(DAT_FILE, "wb") as fout:
            while chunk := fin.read(CHUNK):
                fout.write(chunk)
    except Exception as e:
        print(f"[ERROR] Extraction failed: {e}"); sys.exit(1)
    try:
        os.remove(BZ2_FILE)
    except Exception:
        pass
    print(f"[OK] {DAT_FILE} ready ({os.path.getsize(DAT_FILE)/1_048_576:.1f} MB).")
    print("Launch with:  python dashboard.py")

if __name__ == "__main__":
    download()
