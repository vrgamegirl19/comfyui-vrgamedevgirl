"""Embedded-audio stitch plays the music without jumps, repeats or silence at the cuts.

Live data, Cut and Gun S01-S12 (Beta2.0 a8f36fe, PREVIEW_API_001-012): every clip's audio is its song
window (starts on the window start, runs about the window's length), but the clips' frame counts do not
match the fractional windows. After #11 each clip's audio was cut or padded to frames/24, so at every cut
the music jumped by (window - frames/24): +10, -15, -5, +15, -16.7, +20, -28.3, +40, -0.7, -29.3, +21.7 ms,
and a clip whose audio ran shorter than its picture padded 7-50 ms of digital silence before the cut.
"""

import importlib
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

pkg_name = ROOT.name
video_files = importlib.import_module(f"{pkg_name}.runner.video_files")

FPS = 24
SR = 48000
# Cut and Gun S01-S12: song windows (s), clip frames, clip audio length (s) and clip audio sample rate.
WINDOWS = [(0.0, 6.76), (6.76, 11.62), (11.62, 17.49), (17.49, 25.13), (25.13, 31.28), (31.28, 36.3),
           (36.3, 42.98), (42.98, 50.02), (50.02, 54.436), (54.436, 60.49), (60.49, 65.47), (65.47, 71.47)]
FRAMES = [162, 117, 141, 183, 148, 120, 161, 168, 106, 146, 119, 144]
CLIP_AUDIO = [6.7338, 4.8762, 5.8747, 7.6161, 6.1533, 4.9923, 6.6874, 6.9892, 4.4160, 6.0587, 4.9459, 6.0160]
CLIP_RATES = [44100] * 8 + [48000, 48000, 44100, 48000]
# PREVIEW_API_001-012 jump at cuts 1-11 (ms), jumps.json: the music moved by window - frames/24.
LIVE_JUMPS_MS = [10.0, -15.0, -5.0, 15.0, -16.7, 20.0, -28.3, 40.0, -0.7, -29.3, 21.7]
SONG_SECONDS = 75.0
CROSSFADE = 0.010  # the join crossfade may use up to 10 ms after each cut


def have_tools():
    try:
        for tool in ("ffmpeg", "ffprobe"):
            subprocess.run([tool, "-version"], capture_output=True, check=True)
        import numpy  # noqa: F401
        return True
    except Exception:
        return False


def write_wav(path, samples, rate):
    import numpy as np
    data = np.clip(samples, -1.0, 1.0)
    if data.ndim == 1:
        data = np.stack([data, data], axis=1)
    with wave.open(path, "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes((data * 32767).astype("<i2").tobytes())


def noise_music(seconds, seed, rate=SR):
    """Band-limited noise with a slow envelope: unique at every 20 ms, so a cross-correlation finds one spot."""
    import numpy as np
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    x = rng.standard_normal(n + 32)
    x = np.convolve(x, np.ones(8) / 8, mode="valid")[:n]
    env = 0.6 + 0.4 * np.sin(np.arange(n) / rate * 2 * np.pi * 0.7)
    return (0.25 * x * env).astype("float32")


def decode(path, rate=SR):
    import numpy as np
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(rate), "-f", "f32le", "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32)


def frame_count(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return int(out)


def locate(block, ref, center, search):
    """Sample index in ``ref`` where ``block`` fits best near ``center``, and the normalized correlation."""
    import numpy as np
    lo = max(0, center - search)
    hi = min(len(ref), center + search + len(block))
    seg = ref[lo:hi]
    if len(seg) < len(block):
        return None, 0.0
    n = len(seg) + len(block)
    size = 1 << int(np.ceil(np.log2(n)))
    corr = np.fft.irfft(np.fft.rfft(seg, size) * np.conj(np.fft.rfft(block, size)), size)[: len(seg) - len(block) + 1]
    energy = np.cumsum(np.concatenate([[0.0], seg.astype("float64") ** 2]))
    norm = np.sqrt(energy[len(block):] - energy[:-len(block)]) * np.linalg.norm(block) + 1e-12
    corr = corr / norm
    k = int(np.argmax(corr))
    return lo + k, float(corr[k])


def cut_samples(frames=FRAMES):
    import numpy as np
    return [int(round(c * SR / FPS)) for c in np.cumsum([0] + list(frames))]


@unittest.skipUnless(have_tools(), "ffmpeg/ffprobe and numpy are needed")
class EmbeddedAudioSeamlessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="vrgdg_seamless_")
        cls.song_path = os.path.join(cls.tmp, "song.wav")
        write_wav(cls.song_path, noise_music(SONG_SECONDS, seed=7), SR)
        cls.song = decode(cls.song_path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.project = tempfile.mkdtemp(prefix="proj_", dir=self.tmp)
        self.clip_dir = os.path.join(self.project, "rendered_scene_videos")
        os.makedirs(self.clip_dir)

    def make_clip(self, index, frames, audio=None, rate=SR):
        """Clip ``index`` with ``frames`` frames and ``audio`` (mono float samples at ``rate``) or no audio stream."""
        path = os.path.join(self.clip_dir, f"video_{index:04d}-audio.mp4")
        video = path + ".v.mp4"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size=96x54:rate={FPS}",
                        "-frames:v", str(frames), "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", video],
                       capture_output=True, check=True)
        if audio is None:
            os.replace(video, path)
            return path
        wav = path + ".a.wav"
        write_wav(wav, audio, rate)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", video, "-i", wav, "-map", "0:v", "-map", "1:a",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", path], capture_output=True, check=True)
        os.remove(video)
        os.remove(wav)
        return path

    def song_clips(self, scenes=range(12)):
        """Clips whose audio is their song window, like Cut and Gun (audio runs CLIP_AUDIO s from the window start)."""
        import numpy as np
        song_hi = None
        paths = []
        for k in scenes:
            rate = CLIP_RATES[k]
            if song_hi is None or song_hi[0] != rate:
                song_hi = (rate, decode(self.song_path, rate))
            start = int(round(WINDOWS[k][0] * rate))
            audio = song_hi[1][start:start + int(round(CLIP_AUDIO[k] * rate))]
            paths.append(self.make_clip(k + 1, FRAMES[k], np.asarray(audio), rate))
        return paths

    def stitch(self, paths, scenes=range(12), song=True):
        scenes = list(scenes)
        payload = {
            "project_folder": self.project,
            "scene_paths": paths,
            "audio_path": "",
            "use_embedded_scene_audio": True,
            "scene_timing_items": [{"start": WINDOWS[k][0], "end": WINDOWS[k][1]} for k in scenes],
            "timeline_fps": FPS,
            "output_prefix": "SEAMLESS",
        }
        if song:
            payload["song_path"] = self.song_path
        return video_files._stitch_scene_videos(payload)["final_video_path"]

    # --- checks -----------------------------------------------------------------------------------

    def assert_no_digital_silence_at_cuts(self, audio, cuts):
        import numpy as np
        for k, cut in enumerate(cuts[1:-1], start=1):
            seg = audio[cut - 2400: cut + 2400]
            blocks = np.sqrt((seg[: len(seg) // 96 * 96].reshape(-1, 96) ** 2).mean(axis=1))  # 2 ms RMS
            self.assertGreater(blocks.min(), 0.2 * np.median(blocks), f"silence at cut {k} ({cut / SR:.4f} s)")

    def assert_length_matches_video(self, out, frames):
        total = sum(frames) / FPS
        audio_len = len(decode(out)) / SR
        self.assertGreater(audio_len, total - 0.001)
        self.assertLess(audio_len, total + 1024 / SR + 0.001)
        self.assertEqual(frame_count(out), sum(frames))

    def song_lags(self, audio, positions, expected, search=4800):
        """For each output sample position, (song sample - output sample) where a 20 ms block fits."""
        lags = []
        for pos, exp in zip(positions, expected):
            found, corr = locate(audio[pos:pos + 960], self.song, pos + exp, search)
            self.assertGreater(corr, 0.9, f"output at {pos / SR:.4f} s is not the song (r={corr:.3f})")
            lags.append(found - pos)
        return lags

    # --- tests ------------------------------------------------------------------------------------

    def test_song_clips_play_one_continuous_song(self):
        out = self.stitch(self.song_clips())
        audio = decode(out)
        cuts = cut_samples()
        self.assert_length_matches_video(out, FRAMES)
        self.assert_no_digital_silence_at_cuts(audio, cuts)
        # 20 ms blocks on both sides of every cut, and through the middle of each scene: one constant lag.
        positions = []
        for cut in cuts[1:-1]:
            positions += [cut - 1440, cut - 960, cut + 0, cut + 480, cut + 960]
        positions += [c + 4800 for c in cuts[:-1]]
        lags = self.song_lags(audio, positions, [0] * len(positions))
        jumps_ms = [round((b - a) / SR * 1000, 2) for a, b in zip(lags, lags[1:])]
        self.assertLessEqual(max(lags) - min(lags), 2, f"song jumps (ms): {jumps_ms}")
        self.assertLessEqual(abs(lags[0]), 2)  # the song is read from the timeline start (0 s here)
        # Each scene's window start lands within half a frame of its first frame (frame-exact timeline).
        for k, cut in enumerate(cuts[:-1]):
            self.assertLessEqual(abs(WINDOWS[k][0] * SR - cut), SR / FPS / 2 + 1, f"scene {k + 1}")

    def test_live_jumps_are_what_frames_minus_window_predicts(self):
        # Pins the fixture to the live measurement in jumps.json (window - frames/24 at every cut).
        for k, live in enumerate(LIVE_JUMPS_MS):
            predicted = (WINDOWS[k][1] - WINDOWS[k][0] - FRAMES[k] / FPS) * 1000
            self.assertAlmostEqual(predicted, live, delta=0.1)

    def test_per_scene_audio_fills_from_the_song_and_crossfades(self):
        # Scenes with their own (non-song) audio, mostly shorter than the picture: each starts on its first
        # frame, plays its own audio, and its shortfall continues the song instead of digital silence.
        import numpy as np
        scenes = range(6)
        paths, own = [], []
        for k in scenes:
            length = FRAMES[k] / FPS - (0.030 if k % 2 == 0 else -0.020)  # 30 ms short / 20 ms long
            audio = noise_music(length, seed=100 + k)
            own.append(audio)
            paths.append(self.make_clip(k + 1, FRAMES[k], audio, SR))
        out = self.stitch(paths, scenes=scenes)
        audio = decode(out)
        frames = FRAMES[:6]
        cuts = cut_samples(frames)
        self.assert_length_matches_video(out, frames)
        self.assert_no_digital_silence_at_cuts(audio, cuts)
        skip = int(CROSSFADE * SR)
        for k in scenes:
            cut, n = cuts[k], cuts[k + 1] - cuts[k]
            own_len = len(own[k])
            # Own audio starts on the first frame (lag 0 against the clip), after the join crossfade.
            pos = cut + skip
            found, corr = locate(audio[pos:pos + 960], np.asarray(own[k]), skip, 480)
            self.assertGreater(corr, 0.9, f"scene {k + 1} own audio")
            self.assertLessEqual(abs(found - skip), 1, f"scene {k + 1} starts {(found - skip) / SR * 1000:.2f} ms off")
            if own_len < n:
                # The shortfall is the song at the scene's window start + its own audio length.
                pos = cut + own_len + 120
                if pos + 960 <= cut + n:
                    song_pos = int(round(WINDOWS[k][0] * SR)) + own_len + 120
                    found, corr = locate(audio[pos:pos + 960], self.song, song_pos, 480)
                    self.assertGreater(corr, 0.9, f"scene {k + 1} shortfall is not the song")
                    self.assertLessEqual(abs(found - song_pos), 1, f"scene {k + 1} fill jumps")

    def test_gap_in_song_scenes_uses_per_scene_audio_with_crossfades(self):
        # Song clips but scene 5 left out: one continuous read cannot fit, so each scene plays its own
        # window, filled from the song, and the music only changes position inside the join crossfade.
        scenes = [0, 1, 2, 3, 5, 6]
        out = self.stitch(self.song_clips(scenes), scenes=scenes)
        audio = decode(out)
        frames = [FRAMES[k] for k in scenes]
        cuts = cut_samples(frames)
        self.assert_length_matches_video(out, frames)
        self.assert_no_digital_silence_at_cuts(audio, cuts)
        skip = int(CROSSFADE * SR)
        for i, k in enumerate(scenes):
            cut, end = cuts[i], cuts[i + 1]
            expected = int(round(WINDOWS[k][0] * SR)) - cut
            positions = [cut + skip, (cut + end) // 2, end - 960]
            lags = self.song_lags(audio, positions, [expected] * 3, search=480)
            for lag in lags:
                self.assertLessEqual(abs(lag - expected), 2, f"scene {k + 1}: lags {lags}, expected {expected}")

    def test_no_audio_clip_is_filled_from_the_song(self):
        paths = self.song_clips(range(4))
        os.remove(paths[2])
        paths[2] = self.make_clip(3, FRAMES[2], None)
        out = self.stitch(paths, scenes=range(4))
        audio = decode(out)
        cuts = cut_samples(FRAMES[:4])
        self.assert_no_digital_silence_at_cuts(audio, cuts)
        mid = (cuts[2] + cuts[3]) // 2
        found, corr = locate(audio[mid:mid + 960], self.song, mid, 1200)
        self.assertGreater(corr, 0.9)

    def test_no_audio_clip_without_a_song_stays_silent(self):
        import numpy as np
        paths = self.song_clips(range(3))
        os.remove(paths[1])
        paths[1] = self.make_clip(2, FRAMES[1], None)
        out = self.stitch(paths, scenes=range(3), song=False)
        audio = decode(out)
        cuts = cut_samples(FRAMES[:3])
        quiet = audio[cuts[1] + 2400: cuts[2] - 2400]
        self.assertLess(float(np.abs(quiet).max()), 1e-3)
        self.assert_length_matches_video(out, FRAMES[:3])


if __name__ == "__main__":
    unittest.main()
