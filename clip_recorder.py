"""
clip_recorder.py — Short video clip recorder for drowsiness alarm events.

Records exactly 4 seconds (1s pre-alarm buffer + 3s post-alarm) at 480×360
resolution, 10 FPS, using the mp4v codec.  Clips are stored in a /clips
sub-folder inside the project root.

Only one clip is recorded at a time.  The frame buffer always runs so
pre-alarm frames are available whenever an alarm fires.

v2.0 addition: cleanup_old_clips(max_clips) — call on startup to prevent
the /clips directory from filling the user's hard drive.
"""

import cv2
import os
import glob
import collections
from datetime import datetime

from app_logger import get_logger

_log = get_logger("clip_recorder")

# ── Constants ────────────────────────────────────────────────────────────────
_CLIP_FPS    = 10
_CLIP_WIDTH  = 480
_CLIP_HEIGHT = 360
_PRE_FRAMES  = 10   # 1 s pre-alarm  (10 fps × 1 s)
_POST_FRAMES = 30   # 3 s post-alarm (10 fps × 3 s)
_CLIPS_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clips")
_FOURCC      = cv2.VideoWriter_fourcc(*"mp4v")


# ── Module-level helper (called at engine startup) ───────────────────────────

def cleanup_old_clips(max_clips: int = 50) -> int:
    """
    Delete the oldest clips so that at most *max_clips* remain in _CLIPS_DIR.

    Returns the number of clips deleted.
    """
    try:
        pattern = os.path.join(_CLIPS_DIR, "*.mp4")
        clips   = sorted(glob.glob(pattern), key=os.path.getmtime)
        excess  = clips[:-max_clips] if len(clips) > max_clips else []
        for f in excess:
            try:
                os.remove(f)
                _log.info("Deleted old clip: %s", f)
            except Exception as e:
                _log.warning("Could not delete clip %s: %s", f, e)
        if excess:
            _log.info("Clip cleanup: removed %d old clip(s). %d remain.", len(excess), max_clips)
        return len(excess)
    except Exception as e:
        _log.warning("cleanup_old_clips failed: %s", e)
        return 0


# ── Recorder class ───────────────────────────────────────────────────────────

class ClipRecorder:
    """
    Maintains a rolling frame buffer and writes short clips when a
    drowsiness alarm is confirmed.
    """

    def __init__(self):
        # Ring buffer — holds last PRE_FRAMES resized frames
        self.buffer: collections.deque = collections.deque(maxlen=_PRE_FRAMES)

        self.recording: bool       = False
        self.writer                = None
        self.frames_written: int   = 0
        self.clip_path: str | None = None

        try:
            os.makedirs(_CLIPS_DIR, exist_ok=True)
        except Exception as e:
            _log.warning("Could not create clips dir: %s", e)

    # ── Public API ───────────────────────────────────────────────────────────

    def add_frame(self, frame) -> "str | None":
        """
        Call every detection-loop iteration with the current frame.

        - Always resizes and appends to the pre-alarm buffer.
        - If recording is active, writes frame to the VideoWriter.
        - Returns the completed clip path when POST_FRAMES have been written,
          otherwise returns None.
        """
        try:
            small = cv2.resize(frame, (_CLIP_WIDTH, _CLIP_HEIGHT))
            self.buffer.append(small)

            if self.recording:
                if self.writer is not None:
                    self.writer.write(small)
                    self.frames_written += 1

                if self.frames_written >= _POST_FRAMES:
                    completed_path = self.clip_path
                    self._stop_recording()
                    return completed_path

        except Exception as e:
            _log.warning("add_frame error: %s", e)

        return None

    def start_recording(self) -> "str | None":
        """
        Call at the moment a drowsiness alarm is confirmed.

        Opens a new VideoWriter, flushes the pre-alarm buffer, and returns
        the clip file path, or None on failure.
        """
        if self.recording:
            return self.clip_path  # already recording — keep current clip

        try:
            timestamp      = datetime.now().strftime("%Y%m%d_%H%M%S")
            clip_name      = f"clip_{timestamp}.mp4"
            self.clip_path = os.path.join(_CLIPS_DIR, clip_name)

            self.writer = cv2.VideoWriter(
                self.clip_path, _FOURCC, _CLIP_FPS, (_CLIP_WIDTH, _CLIP_HEIGHT)
            )

            if not self.writer.isOpened():
                raise RuntimeError("VideoWriter could not be opened")

            # Write buffered pre-alarm frames
            for buffered_frame in list(self.buffer):
                self.writer.write(buffered_frame)

            self.recording      = True
            self.frames_written = 0

            _log.info("Recording started → %s", self.clip_path)
            return self.clip_path

        except Exception as e:
            _log.warning("start_recording failed: %s", e)
            self._stop_recording()
            return None

    def is_recording(self) -> bool:
        return self.recording

    def stop_recording(self):
        """Public alias for external callers."""
        self._stop_recording()

    # ── Internals ────────────────────────────────────────────────────────────

    def _stop_recording(self):
        try:
            if self.writer is not None:
                self.writer.release()
        except Exception as e:
            _log.warning("Error releasing VideoWriter: %s", e)
        finally:
            self.writer         = None
            self.recording      = False
            self.frames_written = 0
            if self.clip_path:
                _log.info("Clip saved → %s", self.clip_path)
