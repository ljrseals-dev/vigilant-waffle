#!/usr/bin/env python3
"""
Lumen Vortex — Sacred Generative Resonance Engine
A contemplative web sanctuary for the Vortex Music Engine.
"""

from __future__ import annotations

import os
import random
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import scipy.io.wavfile as wavfile
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Engine lives one level up
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))
from vme_core import (
    DownwashResolver,
    NeuralTimbreSynthesizer,
    generate_raw_pitch,
    best_key_24,
    inject_leap_return,
)

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"
AUDIO_DIR = STATIC / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Lumen Vortex",
    description="Sacred generative resonance — a contemplative interface to the Vortex Music Engine.",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

# Single shared synthesizer (isolated training RNG, safe to reuse)
_synth = NeuralTimbreSynthesizer(num_harmonics=16)
_last_metrics: Optional[dict] = None
_MAX_AUDIO_FILES = 12


class GenerateRequest(BaseModel):
    seed: int = Field(42, ge=0, le=999999)
    notes: int = Field(128, ge=32, le=400)
    tempo: float = Field(72, ge=40, le=160)  # slower, more meditative default
    intention: str = Field("stillness", description="Contemplative mood label")


class MetricsOut(BaseModel):
    stepwise_pct: float
    repeat_pct: float
    mean_interval: float
    key_name: str
    key_correlation: float
    seed: int
    notes: int
    intention: str


def _prune_audio():
    files = sorted(AUDIO_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime)
    while len(files) > _MAX_AUDIO_FILES:
        files.pop(0).unlink(missing_ok=True)


@app.get("/", response_class=HTMLResponse)
async def sanctuary():
    index = STATIC / "index.html"
    if not index.exists():
        raise HTTPException(404, "Sanctuary not found")
    return FileResponse(index)


@app.post("/api/resonate")
async def resonate(req: GenerateRequest):
    global _last_metrics

    # Deterministic stream for this intention
    random.seed(req.seed)
    np.random.seed(req.seed)
    torch.manual_seed(req.seed)

    resolver = DownwashResolver()
    pitches = []
    last_raw = None
    for _ in range(req.notes):
        raw = generate_raw_pitch(last_raw)
        last_raw = raw
        resolver.maybe_change_chord()
        pitches.append(resolver.fit_pitch_to_chord(raw))

    # Leap-return contour injection — clears the mean-interval gap
    pitches = inject_leap_return(pitches)

    intervals = [abs(pitches[i + 1] - pitches[i]) for i in range(len(pitches) - 1)]
    stepwise = 100.0 * sum(1 for i in intervals if i <= 2) / max(1, len(intervals))
    repeats = 100.0 * sum(1 for i in intervals if i == 0) / max(1, len(intervals))
    mean_int = float(np.mean(intervals)) if intervals else 0.0
    key_name, key_corr = best_key_24(pitches)

    # Render audio
    sample_rate = 44100
    time_step = 60.0 / req.tempo
    note_dur = time_step * 0.92
    buffer = []
    for p in pitches:
        wave = _synth.synthesize(p, velocity=72, duration=note_dur, sample_rate=sample_rate)
        buffer.append(wave)
        buffer.append(np.zeros(int(sample_rate * 0.04)))

    full = np.concatenate(buffer)
    peak = np.max(np.abs(full))
    if peak > 0:
        full = full / peak * 0.82
    audio_i16 = (full * 32767).astype(np.int16)

    uid = uuid.uuid4().hex[:10]
    fname = f"lumen_{req.seed}_{uid}.wav"
    path = AUDIO_DIR / fname
    wavfile.write(str(path), sample_rate, audio_i16)
    _prune_audio()

    _last_metrics = {
        "stepwise_pct": round(stepwise, 1),
        "repeat_pct": round(repeats, 1),
        "mean_interval": round(mean_int, 2),
        "key_name": key_name,
        "key_correlation": round(float(key_corr), 3),
        "seed": req.seed,
        "notes": req.notes,
        "intention": req.intention,
        "audio_url": f"/static/audio/{fname}",
        "duration_sec": round(len(audio_i16) / sample_rate, 1),
    }
    return _last_metrics


@app.get("/api/metrics")
async def metrics():
    if _last_metrics is None:
        raise HTTPException(404, "No resonance has been invoked yet.")
    return _last_metrics


@app.get("/api/health")
async def health():
    return {"status": "still", "engine": "Lumen Vortex 1.0"}
