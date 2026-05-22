"""
clip_recorder.py — Short video clip recorder for drowsiness alarm events.

Records exactly 4 seconds (1s pre-alarm buffer + 3s post-alarm) at 480x360
resolution, 10 FPS, using the mp4v codec. Clips are stored in a /clips
sub-folder inside the project root.

Only one clip is recorded at a time. The frame buffer always runs so
pre-alarm frames are available whenever an alarm fires.
"""

import cv2
import os
import collections
from datetime import datetime


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
_CLIP_FPS        = 10
_CLIP_WIDTH      = 480
_CLIP_HEIGHT     = 360
_PRE_FRAMES      = 10   # 1 second of pre-alarm frames  (10 fps × 1 s)
_POST_FRAMES     = 30   # 3 seconds of post-alarm frames (10 fps × 3 s)
_CLIPS_DIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clips")
_FOURCC          = cv2.VideoWriter_fourcc(*"mp4v")


class ClipRecorder:
    """
    Maintains a rolling frame buffer and writes short clips when a
    drowsiness alarm is confirmed.
    """

    def __init__(self):
        # Ring buffer — holds last PRE_FRAMES resized frames
        self.buffer: collections.deque = collections.deque(maxlen=_PRE_FRAMES)

        self.recording: bool      = False
        self.writer               = None          # cv2.VideoWriter or None
        self.frames_written: int  = 0
        self.clip_path: str | None = None

        # Ensure output folder exists
        try:
            os.makedirs(_CLIPS_DIR, exist_ok=True)
        except Exception as e:
            print(f"[ClipRecorder WARNING] Could not create clips dir: {e}")

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def add_frame(self, frame) -> str | None:
        """
        Call this every detection-loop iteration with the current frame.

        - Always resizes and appends to the pre-alarm buffer.
        - If recording is active, writes frame to the VideoWriter.
        - When POST_FRAMES have been written, stops recording and returns
          the completed clip path.
        - Returns None when no clip has just finished.
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
            print(f"[ClipRecorder WARNING] add_frame error: {e}")

        return None

    def start_recording(self) -> str | None:
        """
        Call this at the moment a drowsiness alarm is confirmed.

        - Ignored if already recording.
        - Opens a new VideoWriter.
        - Flushes the pre-alarm buffer first (these are frames BEFORE alarm).
        - Resets frames_written counter.
        - Returns the clip file path, or None on failure.
        """
        if self.recording:
            return self.clip_path   # already recording — keep current clip

        try:
            timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
            clip_name  = f"clip_{timestamp}.mp4"
            self.clip_path = os.path.join(_CLIPS_DIR, clip_name)

            self.writer = cv2.VideoWriter(
                self.clip_path,
                _FOURCC,
                _CLIP_FPS,
                (_CLIP_WIDTH, _CLIP_HEIGHT),
            )

            if not self.writer.isOpened():
                raise RuntimeError("VideoWriter could not be opened")

            # Write buffered pre-alarm frames
            for buffered_frame in list(self.buffer):
                self.writer.write(buffered_frame)

            self.recording      = True
            self.frames_written = 0

            print(f"[ClipRecorder] Recording started → {self.clip_path}")
            return self.clip_path

        except Exception as e:
            print(f"[ClipRecorder WARNING] start_recording failed: {e}")
            self._stop_recording()
            return None

    def is_recording(self) -> bool:
        """Return True if a clip is currently being recorded."""
        return self.recording

    # ──────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────

    def _stop_recording(self):
        """Release the VideoWriter and reset recording state."""
        try:
            if self.writer is not None:
                self.writer.release()
        except Exception as e:
            print(f"[ClipRecorder WARNING] Error releasing writer: {e}")
        finally:
            self.writer         = None
            self.recording      = False
            self.frames_written = 0
            if self.clip_path:
                print(f"[ClipRecorder] Clip saved → {self.clip_path}")

    # Keep stop_recording as a public alias for external callers
    def stop_recording(self):
        self._stop_recording()
