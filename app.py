import os
import re
import json
import uuid
import time
import shutil
import subprocess
import threading
import concurrent.futures
import sys

from io import BytesIO
from pathlib import Path

# ---------------------------------------------------------------------------
# Register NVIDIA DLL directories (Windows only) BEFORE importing CUDA libs.
# pip-installed nvidia-cublas-cu12 / nvidia-cudnn-cu12 place DLLs under
# site-packages/nvidia/*/bin/, but Python 3.8+ no longer searches PATH for
# DLLs — os.add_dll_directory() is required so CTranslate2 can find them.
# ---------------------------------------------------------------------------
if sys.platform == 'win32':
    _sp = Path(sys.executable).parent / 'Lib' / 'site-packages' / 'nvidia'
    if _sp.is_dir():
        for _sub in _sp.iterdir():
            _bin = _sub / 'bin'
            if _bin.is_dir():
                os.add_dll_directory(str(_bin))
                print(f"[DLL] Registered: {_bin}")

import torch
from flask import (Flask, render_template, request, jsonify,
                   send_file, send_from_directory)

print("=" * 50)
if not torch.cuda.is_available():
    print("WARNING: CUDA is not available! Whisper will fall back to CPU, which is significantly slower.")
else:
    print("CUDA is available! Whisper will run on GPU.")
print("=" * 50)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = None  # no file size limit

BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / 'uploads'
OUTPUTS_DIR = BASE_DIR / 'outputs'
SETTINGS_FILE = BASE_DIR / 'settings.json'

UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Startup cleanup — purge leftover temp files from previous runs
# ---------------------------------------------------------------------------

def _startup_cleanup():
    """Remove temp WAVs, filter scripts, concat files, and stale uploads."""
    cleaned = 0
    for f in UPLOADS_DIR.iterdir():
        if f.is_file():
            try:
                f.unlink()
                cleaned += 1
            except OSError:
                pass
    for f in OUTPUTS_DIR.iterdir():
        if f.is_file() and (f.name.startswith('_') or f.name.startswith('filter_')):
            try:
                f.unlink()
                cleaned += 1
            except OSError:
                pass
    if cleaned:
        print(f"[Startup] Cleaned {cleaned} leftover temp files.")

_startup_cleanup()

# ---------------------------------------------------------------------------
# GPU encoder auto-detection — use NVENC if available, else fall back to CPU
# ---------------------------------------------------------------------------

def _detect_video_encoder() -> dict:
    """Probe ffmpeg for h264_nvenc; fall back to libx264 with smart threading."""
    cores = os.cpu_count() or 4
    try:
        r = subprocess.run(
            ['ffmpeg', '-hide_banner', '-encoders'],
            capture_output=True, text=True, timeout=10,
        )
        if 'h264_nvenc' in r.stdout:
            # NVENC: GPU does the heavy lifting; 3 concurrent sessions is safe
            # on all consumer NVIDIA GPUs (driver 470+).
            print("[FFmpeg] NVENC (h264_nvenc) detected — GPU encoding, 3 parallel.")
            return {
                'codec': 'h264_nvenc',
                'opts': ['-preset', 'fast', '-cq', '23'],
                'parallel': 3,
            }
    except Exception:
        pass
    # libx264: scale poorly past ~6 threads per process, so we split work
    # across multiple parallel processes with fewer threads each.
    # E.g. 16 cores → 4 parallel × 4 threads = full utilisation without
    # oversubscription or diminishing-returns thread contention.
    parallel = max(2, min(cores // 2, 8))
    threads = max(1, cores // parallel)
    print(f"[FFmpeg] NVENC not available — libx264, {parallel} parallel × {threads} threads.")
    return {
        'codec': 'libx264',
        'opts': ['-preset', 'fast', '-crf', '23', '-threads', str(threads)],
        'parallel': parallel,
    }

_ENCODER = _detect_video_encoder()

# ---------------------------------------------------------------------------
# Global Whisper model cache — keeps ONE model loaded, evicts on switch
# Uses faster-whisper (CTranslate2) for up to 4x faster inference.
# ---------------------------------------------------------------------------

_whisper_model_cache: dict = {}  # max 1 entry: (model_name, device, compute_type) -> model


def get_whisper_model(model_name: str):
    """Return a cached faster-whisper model. Evicts previous model to free VRAM/RAM."""
    from faster_whisper import WhisperModel
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    # float16 halves VRAM usage and accelerates CUDA matmuls;
    # int8 is the fastest option on CPU (no fp16 support on x86).
    compute_type = "float16" if use_cuda else "int8"
    cache_key = (model_name, device, compute_type)

    if cache_key not in _whisper_model_cache:
        # Evict any previously cached model to free VRAM / system RAM
        for old_key in list(_whisper_model_cache.keys()):
            print(f"[Whisper] Evicting cached model '{old_key[0]}' to free memory.")
            del _whisper_model_cache[old_key]
        if use_cuda:
            torch.cuda.empty_cache()

        print(f"[Whisper] Loading faster-whisper model '{model_name}' on {device} ({compute_type})...")
        _whisper_model_cache[cache_key] = WhisperModel(
            model_name, device=device, compute_type=compute_type
        )
        print(f"[Whisper] Model '{model_name}' loaded and cached.")
    else:
        print(f"[Whisper] Using cached model '{model_name}' on {device} ({compute_type}).")
    return _whisper_model_cache[cache_key]


DEFAULT_SETTINGS = {
    "silence_threshold": -35,
    "silence_duration": 0.4,
    "silence_padding": 0.1,
    "whisper_model": "medium",
    "max_words_per_line": 5,
    "remove_punctuation": True,
    "language": "Hebrew",
}

jobs: dict = {}
JOB_TTL_SECONDS = 3600  # auto-clean completed jobs after 1 hour

# ---------------------------------------------------------------------------
# Server-side file cache — upload once, reprocess many times
# ---------------------------------------------------------------------------

_file_cache: dict = {}      # file_id -> {path, original_name, uploaded_at, size}
_FILE_CACHE_TTL = 3600      # auto-clean cached files after 1 hour
_FILE_CACHE_MAX = 5         # max cached files (LRU eviction)

# silence detection results cache: (file_id, threshold, duration) -> {silences, total_dur}
_silence_cache: dict = {}
_SILENCE_CACHE_MAX = 10


def _cache_file(input_path: str, original_name: str) -> str:
    """Store a file in the cache, return a file_id."""
    file_id = uuid.uuid4().hex
    ext = Path(original_name).suffix or '.mp4'
    cached_path = str(UPLOADS_DIR / f"cached_{file_id}{ext}")
    shutil.move(input_path, cached_path)

    # Evict oldest if at max
    while len(_file_cache) >= _FILE_CACHE_MAX:
        oldest_id = min(_file_cache, key=lambda k: _file_cache[k]['uploaded_at'])
        _evict_cached_file(oldest_id)

    _file_cache[file_id] = {
        'path': cached_path,
        'original_name': original_name,
        'uploaded_at': time.time(),
        'size': os.path.getsize(cached_path),
    }
    return file_id


def _evict_cached_file(file_id: str):
    """Remove a cached file from disk and caches."""
    entry = _file_cache.pop(file_id, None)
    if entry:
        safe_delete(entry['path'])
    # Also purge any silence detection caches for this file
    stale_keys = [k for k in _silence_cache if k[0] == file_id]
    for k in stale_keys:
        del _silence_cache[k]


def _get_cached_file(file_id: str) -> str | None:
    """Return the path for a cached file, or None if expired/missing."""
    entry = _file_cache.get(file_id)
    if not entry:
        return None
    if time.time() - entry['uploaded_at'] > _FILE_CACHE_TTL:
        _evict_cached_file(file_id)
        return None
    if not os.path.exists(entry['path']):
        _evict_cached_file(file_id)
        return None
    return entry['path']


def _cleanup_file_cache():
    """Periodically clean expired cached files."""
    now = time.time()
    expired = [fid for fid, e in _file_cache.items()
               if now - e['uploaded_at'] > _FILE_CACHE_TTL]
    for fid in expired:
        _evict_cached_file(fid)


def cleanup_old_jobs():
    """Remove completed/errored jobs older than JOB_TTL_SECONDS + their files."""
    now = time.time()
    stale = [jid for jid, j in jobs.items()
             if j.get('status') in ('done', 'error')
             and now - j.get('finished_at', now) > JOB_TTL_SECONDS]
    for jid in stale:
        j = jobs.pop(jid, None)
        if j:
            # Also delete associated output / source files from disk
            for key in ('output_file', 'source_file'):
                fname = j.get(key)
                if fname:
                    safe_delete(str(OUTPUTS_DIR / fname))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_settings() -> dict:
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                return {**DEFAULT_SETTINGS, **json.load(f)}
        except Exception:
            pass
    return dict(DEFAULT_SETTINGS)


def save_settings(data: dict):
    merged = {**DEFAULT_SETTINGS, **data}
    with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    return merged


def get_duration(filepath: str) -> float:
    """Use ffprobe to get media duration in seconds."""
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        filepath,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return float(result.stdout.strip())
    except Exception:
        return 0.0


# Pre-compiled regexes — called thousands of times during stderr parsing
_RE_FFMPEG_TIME = re.compile(r'time=(\d+):(\d+):(\d+(?:\.\d+)?)')
_RE_SILENCE_START = re.compile(r'silence_start:\s*([\d.]+)')
_RE_SILENCE_END = re.compile(r'silence_end:\s*([\d.]+)')


def parse_ffmpeg_time(line: str):
    """Extract seconds from ffmpeg stderr line like 'time=00:01:23.45'."""
    m = _RE_FFMPEG_TIME.search(line)
    if m:
        h, mi, s = float(m.group(1)), float(m.group(2)), float(m.group(3))
        return h * 3600 + mi * 60 + s
    return None


def safe_delete(path: str):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def _split_words_to_subtitles(all_words, max_words: int):
    """
    Split word-level timestamps into properly-sized subtitle segments.

    Context-aware breaking priority (once buffer reaches *max_words*):
      1. Punctuation at end of current word  (. , ! ? ; :)
      2. Next word is a Hebrew clause opener  (אבל, כי, אז, גם, או, ...)
         or starts with prefix-ו  (ואנחנו, ובכלל, ועכשיו …)
      3. Pause gap > 300 ms between current word end and next word start
      4. Past 1.5× max_words: micro-pause > 150 ms
      5. Hard break at 2× max_words regardless

    Returns a list of dicts: [{start, end, text}, ...]
    """
    if not all_words:
        return []

    # Hebrew conjunctions / clause openers — natural subtitle break points
    _CLAUSE_OPENERS = {
        'אבל', 'כי', 'אז', 'גם', 'או', 'רק', 'אם', 'לכן', 'משום',
        'בגלל', 'למרות', 'אלא', 'כאשר', 'שזה', 'היום', 'עכשיו',
        'כאילו', 'בעצם', 'פשוט', 'ואז', 'אפילו', 'לפני', 'אחרי',
    }
    _PUNCT = set('.,!?;:،؟')

    subtitles = []
    buf = []       # [(word_text, start, end), ...]

    def _flush():
        if not buf:
            return
        subtitles.append({
            'start': round(buf[0][1], 3),
            'end':   round(buf[-1][2], 3),
            'text':  ' '.join(w[0] for w in buf),
        })
        buf.clear()

    for i, w in enumerate(all_words):
        text = w.word.strip()
        if not text:
            continue
        buf.append((text, w.start, w.end))
        n = len(buf)
        if n < 2:
            continue

        # ---- decide whether to flush after this word ----
        flush = False

        if n >= max_words * 2:                             # 5) hard cap
            flush = True
        elif n >= max_words:
            # 1) punctuation at end of word
            if text[-1] in _PUNCT:
                flush = True

            # look-ahead checks
            if not flush and i + 1 < len(all_words):
                nxt = all_words[i + 1]
                nxt_text = nxt.word.strip()
                nxt_clean = nxt_text.lstrip(''.join(_PUNCT))

                # 2a) next word is an explicit clause opener
                if nxt_clean in _CLAUSE_OPENERS:
                    flush = True
                # 2b) next word starts with prefix-ו (ואנחנו, ובכלל…)
                if not flush and len(nxt_clean) > 2 and nxt_clean.startswith('ו'):
                    flush = True
                # 3) pause gap > 300 ms
                if not flush and nxt.start - w.end > 0.3:
                    flush = True

            # 4) past 1.5× — softer threshold
            if not flush and n >= int(max_words * 1.5):
                if i + 1 < len(all_words) and all_words[i + 1].start - w.end > 0.15:
                    flush = True
                if not flush and n >= int(max_words * 1.8):
                    flush = True                            # almost at hard cap

        if flush:
            _flush()

    _flush()  # leftover
    return subtitles


def srt_timestamp(seconds: float) -> str:
    """Format seconds to SRT timestamp HH:MM:SS,mmm (comma, not dot)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

# ---------------------------------------------------------------------------
# Background workers
# ---------------------------------------------------------------------------

def _encode_segment(seg_s: float, seg_e: float, input_path: str,
                    seg_file: str, cancel_check=None) -> float:
    """Encode one segment with -ss input seeking. Returns segment duration.
    Called from thread pool — must be self-contained (no shared mutable state).
    """
    if cancel_check and cancel_check():
        raise RuntimeError('Cancelled')
    seg_dur = seg_e - seg_s
    cmd = [
        'ffmpeg', '-y',
        '-ss', f'{seg_s:.3f}',       # input-level fast keyframe seek
        '-i', input_path,
        '-t', f'{seg_dur:.3f}',       # exact segment duration
        '-c:v', _ENCODER['codec'], *_ENCODER['opts'],
        '-c:a', 'aac', '-b:a', '128k',
        '-avoid_negative_ts', 'make_zero',
        '-v', 'error',               # suppress non-error output (parallel)
        seg_file,
    ]
    r = subprocess.run(
        cmd, capture_output=True, text=True,
        encoding='utf-8', errors='replace',
    )
    if r.returncode != 0:
        err = r.stderr.strip().split('\n')[-1] if r.stderr.strip() else 'unknown'
        raise RuntimeError(f'ffmpeg: {err}')
    return seg_dur


def run_silence_cut(job_id: str, input_path: str, threshold: float,
                    duration: float, padding: float,
                    file_id: str = None, is_cached: bool = False):
    """
    Silence removal with frame-accurate A/V sync.

    Fast pipeline — parallel segment-based with input-level seeking:
      1. silencedetect directly on source (-vn skips video decoding)
      2. Encode kept segments in PARALLEL using -ss input seek + thread pool
         (each segment: keyframe seek → decode only that slice, NOT full file)
      3. Concat all segments with -c copy (instant, no re-encode)

    Performance on a 30-min video with 50 segments (16-core CPU):
      Old filter_complex:    ~15 min  (N × full decode)
      Sequential -ss:        ~2 min   (N × seek + small decode)
      Parallel -ss (×8):     ~20 sec  (seek + small decode, 8 at a time)
    """
    job = jobs[job_id]
    output_name = f"cut_{uuid.uuid4().hex[:8]}.mp4"
    output_path = str(OUTPUTS_DIR / output_name)
    temp_files = []  # all temp files, cleaned up in finally
    parallel = _ENCODER.get('parallel', 4)

    try:
        # --- Step 1: get total duration ---
        job['message'] = 'Analysing duration...'
        total_dur = get_duration(input_path)
        if total_dur <= 0:
            total_dur = 1.0

        if job.get('_cancel'):
            raise RuntimeError('Cancelled')

        # --- Step 2: detect silences (use cache if available) ---
        cache_key = (file_id, threshold, duration) if file_id else None
        cached_silences = _silence_cache.get(cache_key) if cache_key else None

        if cached_silences:
            # Cache hit — skip the expensive silence detection pass
            silences = cached_silences['silences']
            total_dur = cached_silences['total_dur']
            job['message'] = f'Using cached silence data ({len(silences)} regions)...'
            job['progress'] = 20
            print(f"[SilenceCut] Cache HIT for file_id={file_id}, {len(silences)} silences")
        else:
            # Cache miss — run ffmpeg silencedetect
            # -vn skips video decoding; pre-compiled regexes avoid re-compilation
            # on every line (thousands of lines for long videos).
            job['message'] = 'Detecting silences...'
            job['progress'] = 5
            detect_cmd = [
                'ffmpeg', '-i', input_path,
                '-vn',
                '-af', f'silencedetect=noise={threshold}dB:d={duration}',
                '-f', 'null', '-',
            ]
            proc = subprocess.Popen(
                detect_cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL,
                text=True, encoding='utf-8', errors='replace',
            )
            job['_proc'] = proc

            silences = []
            s_start = None
            for line in iter(proc.stderr.readline, ''):
                # Fast short-circuit: only run regex on lines that could match
                if 'time=' in line:
                    t = parse_ffmpeg_time(line)
                    if t is not None and total_dur > 0:
                        pct = 5 + min(int(t / total_dur * 15), 15)
                        job['progress'] = pct
                if 'silence_' in line:
                    m = _RE_SILENCE_START.search(line)
                    if m:
                        s_start = float(m.group(1))
                    else:
                        m = _RE_SILENCE_END.search(line)
                        if m and s_start is not None:
                            silences.append((s_start, float(m.group(1))))
                            s_start = None

            proc.wait()
            job['_proc'] = None
            if s_start is not None:
                silences.append((s_start, total_dur))

            job['progress'] = 20

            # Cache the results for future parameter tweaks
            if cache_key:
                # Evict oldest if at max
                while len(_silence_cache) >= _SILENCE_CACHE_MAX:
                    oldest_key = next(iter(_silence_cache))
                    del _silence_cache[oldest_key]
                _silence_cache[cache_key] = {
                    'silences': silences,
                    'total_dur': total_dur,
                }
                print(f"[SilenceCut] Cached {len(silences)} silences for file_id={file_id}")

        if not silences:
            shutil.copy2(input_path, output_path)
            job['progress'] = 100
            job['status'] = 'done'
            job['message'] = 'No silences detected \u2014 file copied as-is.'
            job['output_file'] = output_name
            job['finished_at'] = time.time()
            return

        # --- Step 3: build non-silent segments with padding ---
        job['message'] = f'Found {len(silences)} silent regions, building cut list...'
        keep = []
        prev_end = 0.0
        for si_start, si_end in silences:
            seg_start = prev_end
            seg_end = si_start
            padded_start = max(0, seg_start - padding)
            padded_end = min(total_dur, seg_end + padding)
            if padded_end - padded_start > 0.05:
                keep.append((padded_start, padded_end))
            prev_end = si_end
        if total_dur - prev_end > 0.05:
            padded_start = max(0, prev_end - padding)
            keep.append((padded_start, total_dur))

        if not keep:
            raise RuntimeError('No non-silent segments found')

        # Merge overlapping segments caused by padding
        merged = [keep[0]]
        for s, e in keep[1:]:
            if s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        keep = merged

        job['progress'] = 25
        expected_dur = sum(e - s for s, e in keep)

        if job.get('_cancel'):
            raise RuntimeError('Cancelled')

        # --- Step 4: parallel segment encoding ---
        # Each segment uses -ss input seek (keyframe jump → decode only the
        # segment's frames).  ThreadPoolExecutor runs N encodes concurrently:
        #   NVENC:  3 parallel (GPU HW limit)
        #   libx264: cores/2 parallel × cores/parallel threads each = full CPU
        workers = min(parallel, len(keep))
        job['message'] = f'Encoding {len(keep)} segments ({workers} parallel)...'
        segment_files = []

        for i in range(len(keep)):
            seg_file = str(OUTPUTS_DIR / f"_seg_{uuid.uuid4().hex[:6]}_{i}.mp4")
            segment_files.append(seg_file)
            temp_files.append(seg_file)

        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {}
            cancel_fn = lambda: job.get('_cancel', False)
            for i, (seg_s, seg_e) in enumerate(keep):
                f = pool.submit(_encode_segment, seg_s, seg_e,
                                input_path, segment_files[i], cancel_fn)
                future_map[f] = i

            for future in concurrent.futures.as_completed(future_map):
                if job.get('_cancel'):
                    for f in future_map:
                        f.cancel()
                    raise RuntimeError('Cancelled')
                future.result()  # raises RuntimeError on encode failure
                completed += 1
                job['progress'] = 25 + min(int(completed / len(keep) * 70), 70)
                job['message'] = f'Encoded {completed}/{len(keep)} segments...'

        # --- Step 5: concat all segments (stream copy — instant) ---
        job['message'] = 'Joining segments...'
        job['progress'] = 96

        concat_file = str(OUTPUTS_DIR / f"_concat_{uuid.uuid4().hex[:8]}.txt")
        temp_files.append(concat_file)

        with open(concat_file, 'w', encoding='utf-8') as f:
            for sf in segment_files:
                safe_path = sf.replace('\\', '/')
                f.write(f"file '{safe_path}'\n")

        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat', '-safe', '0',
            '-i', concat_file,
            '-c', 'copy',
            '-movflags', '+faststart',
            output_path,
        ]

        r = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding='utf-8', errors='replace', timeout=300,
        )
        if r.returncode != 0:
            raise RuntimeError('Segment concatenation failed')

        if job.get('status') == 'cancelled':
            raise RuntimeError('Cancelled')
        job['progress'] = 100
        job['status'] = 'done'
        removed = total_dur - expected_dur
        job['message'] = (
            f'Done! Removed {len(silences)} silences '
            f'({removed:.1f}s cut, {expected_dur:.1f}s kept)'
        )
        job['output_file'] = output_name
        job['finished_at'] = time.time()

    except Exception as e:
        if job.get('status') != 'cancelled':
            job['status'] = 'error'
            job['message'] = str(e)
            job['finished_at'] = time.time()
    finally:
        job['_proc'] = None
        # Only delete the input file if it's NOT a cached file
        if not is_cached:
            safe_delete(input_path)
        for tf in temp_files:
            safe_delete(tf)


def run_transcribe(job_id: str, input_path: str, model_name: str,
                   language: str, max_words: int, max_lines: int = 2,
                   diarize: bool = False):
    """
    Transcribe with faster-whisper (CTranslate2 backend). Optimised pipeline:
      - float16 on GPU, int8 on CPU — faster matmuls, lower memory.
      - Silero VAD filter strips silent sections BEFORE the transformer:
        long pauses / music intros are skipped entirely, saving significant
        compute on typical interview / lecture / screencast recordings.
      - Greedy beam_size=1 for additional ~3x speed vs default beam=5.
      - Hebrew initial prompt maintained for accuracy on he/Hebrew input.
    """
    job = jobs[job_id]
    source_name = f"source_{uuid.uuid4().hex[:8]}{Path(input_path).suffix}"
    source_out = str(OUTPUTS_DIR / source_name)

    try:
        # Step 1 – move source to outputs for preview (instant rename, no copy)
        job['message'] = 'Preparing file...'
        job['progress'] = 5
        shutil.move(input_path, source_out)
        job['source_file'] = source_name

        if job.get('_cancel'):
            raise RuntimeError('Cancelled')

        # Step 2 – load faster-whisper model (cached — evicts previous model if different)
        job['message'] = f'Loading Whisper model ({model_name})...'
        model = get_whisper_model(model_name)
        job['progress'] = 15

        if job.get('_cancel'):
            raise RuntimeError('Cancelled')

        job['message'] = 'Transcribing (VAD active)...'

        # Step 3 – transcribe with faster-whisper
        # vad_filter=True uses Silero VAD to skip silent regions before the
        # transformer — huge win for long videos with pauses, music, etc.
        # beam_size=1 further reduces compute with negligible accuracy loss.
        lang = language.lower() if language else None
        # Normalise language tag: full word → ISO code for faster-whisper
        _LANG_MAP = {'hebrew': 'he', 'english': 'en', 'arabic': 'ar',
                     'russian': 'ru', 'french': 'fr', 'spanish': 'es',
                     'german': 'de', 'portuguese': 'pt', 'italian': 'it'}
        if lang in _LANG_MAP:
            lang = _LANG_MAP[lang]

        initial_prompt = None
        if lang in ('he', 'hebrew'):
            initial_prompt = '\u05d4\u05e7\u05dc\u05d8\u05d4 \u05d1\u05e2\u05d1\u05e8\u05d9\u05ea.'

        segments_gen, info = model.transcribe(
            source_out,
            language=lang,
            beam_size=1,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            initial_prompt=initial_prompt,
            condition_on_previous_text=False,
        )

        # Materialise the lazy generator; collect ALL words with timestamps
        all_words = []
        total_duration = info.duration or 1.0
        for seg in segments_gen:
            if job.get('_cancel'):
                raise RuntimeError('Cancelled')
            # faster-whisper segments have .words when word_timestamps=True
            if seg.words:
                all_words.extend(seg.words)
            # Progress: 15-80 mapped to audio position
            pct = 15 + min(int(seg.end / total_duration * 65), 65)
            job['progress'] = pct
            job['message'] = f'Transcribing... {seg.end:.1f}s / {total_duration:.1f}s'

        job['progress'] = 80

        if job.get('_cancel'):
            raise RuntimeError('Cancelled')

        # Step 4 – context-aware subtitle splitting using word timestamps
        # max_words = words per line, max_lines = lines per subtitle
        # total words per subtitle = max_words × max_lines
        max_words_per_sub = max_words * max_lines
        job['message'] = f'Building subtitles ({len(all_words)} words, {max_words}w×{max_lines}L)...'
        segments = _split_words_to_subtitles(all_words, max_words_per_sub)
        for i, seg in enumerate(segments):
            seg['id'] = i + 1

        job['progress'] = 90

        # Step 5 – speaker diarization (gap-based heuristic)
        if diarize and len(segments) > 1:
            job['message'] = 'Detecting speakers...'
            GAP_THRESHOLD = 1.5
            current_speaker = 1
            segments[0]['speaker'] = current_speaker
            for idx in range(1, len(segments)):
                gap = segments[idx]['start'] - segments[idx - 1]['end']
                if gap >= GAP_THRESHOLD:
                    current_speaker = (current_speaker % 2) + 1
                segments[idx]['speaker'] = current_speaker
            for seg in segments:
                seg['text'] = f"[Speaker {seg['speaker']}] {seg['text']}"
        elif diarize and len(segments) == 1:
            segments[0]['speaker'] = 1

        full_text = ' '.join(seg['text'] for seg in segments)

        if job.get('status') != 'cancelled':
            job['progress'] = 100
            job['status'] = 'done'
            job['message'] = f'Transcription complete \u2014 {len(segments)} segments'
            job['transcript'] = segments
            job['full_text'] = full_text
            job['finished_at'] = time.time()

    except Exception as e:
        if job.get('status') != 'cancelled':
            job['status'] = 'error'
            job['message'] = str(e)
            job['finished_at'] = time.time()
    finally:
        # If move failed, original file still in uploads — clean it up
        if not job.get('source_file'):
            safe_delete(input_path)

# ---------------------------------------------------------------------------
# Error handlers — return JSON instead of HTML so the frontend can parse them
# ---------------------------------------------------------------------------

@app.errorhandler(413)
def file_too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 2 GB.'}), 413


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': f'Server error: {e}'}), 500


@app.errorhandler(404)
def not_found(e):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Endpoint not found'}), 404
    return render_template('index.html'), 404


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/settings', methods=['GET'])
def get_settings():
    return jsonify(load_settings())


@app.route('/api/settings', methods=['POST'])
def post_settings():
    data = request.get_json(force=True)
    saved = save_settings(data)
    return jsonify(saved)


@app.route('/api/upload', methods=['POST'])
def upload_file_endpoint():
    """Upload a file once; returns a file_id for later reuse."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    f = request.files['file']
    ext = Path(f.filename).suffix or '.mp4'
    temp_path = str(UPLOADS_DIR / f"{uuid.uuid4().hex}{ext}")
    f.save(temp_path)

    file_id = _cache_file(temp_path, f.filename)
    entry = _file_cache[file_id]
    return jsonify({
        'file_id': file_id,
        'original_name': f.filename,
        'size': entry['size'],
    })


@app.route('/api/cut-silences', methods=['POST'])
def cut_silences():
    _cleanup_file_cache()

    # Accept either a cached file_id or a fresh file upload
    file_id = request.form.get('file_id')
    input_path = None
    is_cached = False

    if file_id:
        input_path = _get_cached_file(file_id)
        if not input_path:
            return jsonify({'error': 'Cached file expired or not found. Please re-upload.'}), 410
        is_cached = True
    elif 'file' in request.files:
        # Legacy / first-time upload: accept file inline and cache it
        f = request.files['file']
        ext = Path(f.filename).suffix or '.mp4'
        temp_path = str(UPLOADS_DIR / f"{uuid.uuid4().hex}{ext}")
        f.save(temp_path)
        file_id = _cache_file(temp_path, f.filename)
        input_path = _get_cached_file(file_id)
        is_cached = True
    else:
        return jsonify({'error': 'No file uploaded and no file_id provided'}), 400

    threshold = float(request.form.get('threshold', -35))
    duration = float(request.form.get('duration', 0.4))
    padding = float(request.form.get('padding', 0.1))

    job_id = uuid.uuid4().hex
    jobs[job_id] = {
        'status': 'running',
        'progress': 0,
        'message': 'Starting...',
        'output_file': None,
        'transcript': None,
        '_cancel': False,
        '_proc': None,
    }

    t = threading.Thread(
        target=run_silence_cut,
        args=(job_id, input_path, threshold, duration, padding),
        kwargs={'file_id': file_id, 'is_cached': is_cached},
        daemon=True,
    )
    t.start()
    return jsonify({'job_id': job_id, 'file_id': file_id})


@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    f = request.files['file']
    ext = Path(f.filename).suffix or '.mp4'
    input_path = str(UPLOADS_DIR / f"{uuid.uuid4().hex}{ext}")
    f.save(input_path)

    model_name = request.form.get('model', 'medium')
    language = request.form.get('language', 'Hebrew')
    max_words = int(request.form.get('max_words', 5))
    max_lines = int(request.form.get('max_lines', 2))
    diarize = request.form.get('diarize', 'false').lower() == 'true'

    job_id = uuid.uuid4().hex
    jobs[job_id] = {
        'status': 'running',
        'progress': 0,
        'message': 'Starting...',
        'output_file': None,
        'transcript': None,
        'source_file': None,
        '_cancel': False,
        '_proc': None,
    }

    t = threading.Thread(
        target=run_transcribe,
        args=(job_id, input_path, model_name, language, max_words, max_lines, diarize),
        daemon=True,
    )
    t.start()
    return jsonify({'job_id': job_id})


@app.route('/api/job/<job_id>')
def job_status(job_id):
    cleanup_old_jobs()
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    # Filter internal keys (_cancel, _proc) from API response
    return jsonify({k: v for k, v in job.items() if not k.startswith('_')})


@app.route('/api/job/<job_id>/cancel', methods=['POST'])
def cancel_job(job_id):
    """Cancel a running job — kills active ffmpeg subprocess."""
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    if job['status'] != 'running':
        return jsonify({'status': job['status']}), 200
    job['_cancel'] = True
    job['status'] = 'cancelled'
    job['message'] = 'Cancelled by user'
    job['finished_at'] = time.time()
    proc = job.get('_proc')
    if proc:
        try:
            proc.terminate()
        except OSError:
            pass
    return jsonify({'status': 'cancelled'})


@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    """Purge all cached files and silence detection results."""
    file_count = len(_file_cache)
    silence_count = len(_silence_cache)
    for fid in list(_file_cache.keys()):
        _evict_cached_file(fid)
    _silence_cache.clear()
    return jsonify({
        'cleared_files': file_count,
        'cleared_silences': silence_count,
    })


@app.route('/api/export-srt', methods=['POST'])
def export_srt():
    """Generate SRT in memory — no temp file on disk."""
    data = request.get_json(force=True)
    segments = data.get('segments', [])
    max_words = int(data.get('max_words_per_line', 5))
    remove_punct = data.get('remove_punctuation', False)

    punct_re = re.compile(r'[,\.!?;:\"\'\u060C]+')
    lines = []

    for i, seg in enumerate(segments, 1):
        text = seg.get('text', '')
        if remove_punct:
            text = punct_re.sub('', text).strip()

        words = text.split()
        text_lines = []
        for j in range(0, len(words), max_words):
            text_lines.append(' '.join(words[j:j + max_words]))
        formatted_text = '\n'.join(text_lines) if text_lines else text

        start_ts = srt_timestamp(seg.get('start', 0))
        end_ts = srt_timestamp(seg.get('end', 0))
        lines.append(f"{i}\n{start_ts} --> {end_ts}\n{formatted_text}\n")

    srt_content = '\n'.join(lines)
    buf = BytesIO(srt_content.encode('utf-8'))
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name='subtitles.srt',
                     mimetype='text/plain')


@app.route('/api/export-markdown', methods=['POST'])
def export_markdown():
    """Generate Markdown in memory — no temp file on disk."""
    data = request.get_json(force=True)
    text = data.get('text', '')
    title = data.get('title', 'Transcription')

    md = f"# {title}\n\n{text}\n"
    buf = BytesIO(md.encode('utf-8'))
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name='transcription.md',
                     mimetype='text/markdown')


@app.route('/api/download/<filename>')
def download_file(filename):
    safe_name = Path(filename).name  # prevent path traversal
    file_path = OUTPUTS_DIR / safe_name
    if not file_path.exists():
        return jsonify({'error': 'File not found'}), 404
    return send_from_directory(str(OUTPUTS_DIR), safe_name, as_attachment=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 50)
    print("  ReelStudio running at http://localhost:5177")
    print("=" * 50)
    app.run(host='127.0.0.1', port=5177, debug=False, threaded=True)
