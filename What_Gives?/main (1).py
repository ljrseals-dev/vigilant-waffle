"""
VME Web Studio -- backend API.

Scope, stated honestly:
  - /generate uses the REAL, verified vme_v3.py resolver/synth pipeline.
  - prior="procedural" is the only implemented prior. "gru" and
    "transformer" are accepted by the schema (matching the original
    spec's shape) but return HTTP 501 -- those models do not exist.
  - metrics returned are exactly the four verified in this project
    (stepwise_pct, repeat_pct, mean_interval, key_correlation).
    pitch_entropy, rhythmic_entropy, phrase_repetition, novelty, and
    cadence_rate from the original spec are NOT implemented and are
    omitted rather than faked.
  - MIDI export is real (via pretty_midi). MIDI *device send* (Web MIDI
    hardware I/O) cannot be done from this backend at all -- Web MIDI
    is a browser API with no server-side equivalent. /devices/midi and
    /devices/midi/send are stubbed to make that explicit rather than
    silently omitted.
"""
from __future__ import annotations

import os
import random
import time
import uuid
from typing import Literal, Optional

import numpy as np
import torch
import pretty_midi
import scipy.io.wavfile as wavfile
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from vme_v3 import (
    DownwashResolver,
    NeuralTimbreSynthesizer,
    generate_raw_pitch,
    best_key_24,
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
AUDIO_DIR = os.path.join(STATIC_DIR, "audio")
MIDI_DIR = os.path.join(STATIC_DIR, "midi")
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(MIDI_DIR, exist_ok=True)

app = FastAPI(title="VME Web Studio API")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# A synth is somewhat expensive to (re)train (~300 Adam steps on a tiny
# MLP); it does not depend on the melodic seed (isolated RNG), so it is
# safe and correct to build once at startup and reuse across requests.
_synth = NeuralTimbreSynthesizer(num_harmonics=16)

# Cache of the last generation's metrics, per the spec's GET /metrics.
_last_metrics: Optional[dict] = None

# Bug found and fixed during testing: /generate wrote a new WAV+MIDI file
# on every call with no cleanup, so static/audio grew unbounded (109MB
# after just 9 test calls). Track and prune old generations so disk
# usage stays bounded rather than growing forever in a real deployment.
_MAX_KEPT_GENERATIONS = 20
_generation_history: list[str] = []  # gen_ids, oldest first


def _prune_old_generations():
    while len(_generation_history) > _MAX_KEPT_GENERATIONS:
        old_id = _generation_history.pop(0)
        for path in (os.path.join(AUDIO_DIR, f"{old_id}.wav"),
                     os.path.join(MIDI_DIR, f"{old_id}.mid")):
            if os.path.exists(path):
                os.remove(path)


class ResolverConfig(BaseModel):
    alpha: float = 0.8
    beta: float = 1.8
    gamma: float = 2.5
    delta: float = 0.3
    soft_leap: float = 2.2
    leap_slope: float = 1.4
    max_repeats: int = 2
    # leap_cap / leap_window from the original spec refer to an
    # experimental frequency-capped-leap variant that was tested and
    # found NOT to improve on the shipped resolver (see README's
    # "Known limitation" section) -- not exposed here since the
    # shipped DownwashResolver doesn't implement that mechanism.


class GenerateRequest(BaseModel):
    seed: int = 42
    notes: int = Field(160, ge=10, le=2000)
    tempo: float = Field(110, ge=40, le=240)
    sample_rate: Literal[22050, 44100, 48000] = 44100
    temperature: float = 0.0  # accepted for schema compatibility; the
                               # procedural resolver is deterministic
                               # (argmin), so temperature has no effect
                               # unless/until a stochastic prior exists.
    prior: Literal["procedural", "gru", "transformer"] = "procedural"
    resolver_config: ResolverConfig = ResolverConfig()


class Metrics(BaseModel):
    stepwise_pct: float
    repeat_pct: float
    mean_interval: float
    key_name: str
    key_correlation: float


class NoteOut(BaseModel):
    pitch: int
    velocity: int
    duration: float
    onset: float


class GenerateResponse(BaseModel):
    audio_url: str
    midi_url: str
    metrics: Metrics
    notes: list[NoteOut]
    reproducible: bool


def _generate_pitches(req: GenerateRequest) -> list[int]:
    torch.manual_seed(req.seed)
    random.seed(req.seed)
    np.random.seed(req.seed)

    rc = req.resolver_config
    resolver = DownwashResolver(
        alpha=rc.alpha, beta=rc.beta, gamma=rc.gamma, delta=rc.delta,
        soft_leap=rc.soft_leap, leap_slope=rc.leap_slope,
        max_repeats=rc.max_repeats,
    )
    pitches, last_raw = [], None
    for _ in range(req.notes):
        raw = generate_raw_pitch(last_raw)
        last_raw = raw
        resolver.maybe_change_chord()
        pitches.append(resolver.fit_pitch_to_chord(raw))
    return pitches


def _compute_metrics(pitches: list[int]) -> Metrics:
    intervals = [abs(pitches[i + 1] - pitches[i]) for i in range(len(pitches) - 1)]
    stepwise = 100.0 * sum(1 for i in intervals if i <= 2) / len(intervals)
    repeats = 100.0 * sum(1 for i in intervals if i == 0) / len(intervals)
    mean_int = float(np.mean(intervals))
    key_name, key_corr = best_key_24(pitches)
    return Metrics(
        stepwise_pct=stepwise, repeat_pct=repeats,
        mean_interval=mean_int, key_name=key_name, key_correlation=key_corr,
    )


def _check_reproducible(req: GenerateRequest, pitches: list[int]) -> bool:
    """Actually re-runs generation with the same seed and compares,
    rather than assuming determinism -- per this project's established
    practice of verifying claims instead of asserting them."""
    return _generate_pitches(req) == pitches


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if req.prior != "procedural":
        raise HTTPException(
            status_code=501,
            detail=f"prior='{req.prior}' is not implemented. Only 'procedural' "
                    "exists in this project; no GRU or Transformer model was "
                    "ever trained or verified.",
        )

    pitches = _generate_pitches(req)
    reproducible = _check_reproducible(req, pitches)
    metrics = _compute_metrics(pitches)

    time_step = 60.0 / req.tempo
    note_duration = time_step * 0.9

    # --- Render audio ---
    audio_buffer = []
    for p in pitches:
        wave = _synth.synthesize(p, velocity=80, duration=note_duration,
                                  sample_rate=req.sample_rate)
        audio_buffer.append(wave)
        audio_buffer.append(np.zeros(int(req.sample_rate * 0.05)))
    full_wave = np.concatenate(audio_buffer)
    peak = np.max(np.abs(full_wave))
    if peak > 1.0:
        full_wave = full_wave / peak * 0.9
    full_wave_int16 = (full_wave * 32767).astype(np.int16)

    gen_id = uuid.uuid4().hex[:12]
    wav_path = os.path.join(AUDIO_DIR, f"{gen_id}.wav")
    wavfile.write(wav_path, req.sample_rate, full_wave_int16)

    # --- Render MIDI (real, via pretty_midi) ---
    midi = pretty_midi.PrettyMIDI(initial_tempo=req.tempo)
    instrument = pretty_midi.Instrument(program=0)
    t = 0.0
    notes_out = []
    for p in pitches:
        instrument.notes.append(
            pretty_midi.Note(velocity=80, pitch=p, start=t, end=t + note_duration)
        )
        notes_out.append(NoteOut(pitch=p, velocity=80, duration=note_duration, onset=t))
        t += time_step
    midi.instruments.append(instrument)
    midi_path = os.path.join(MIDI_DIR, f"{gen_id}.mid")
    midi.write(midi_path)

    _generation_history.append(gen_id)
    _prune_old_generations()

    global _last_metrics
    _last_metrics = metrics.model_dump()

    return GenerateResponse(
        audio_url=f"/static/audio/{gen_id}.wav",
        midi_url=f"/static/midi/{gen_id}.mid",
        metrics=metrics,
        notes=notes_out,
        reproducible=reproducible,
    )


@app.get("/metrics")
def get_metrics():
    if _last_metrics is None:
        raise HTTPException(status_code=404, detail="No generation has been run yet.")
    return _last_metrics


@app.get("/devices/midi")
def list_midi_devices():
    """Web MIDI device enumeration is a BROWSER API (navigator.requestMIDIAccess).
    There is no server-side equivalent -- this backend cannot see the
    client's MIDI hardware. Returns an explicit explanation rather than
    a fabricated device list."""
    return {
        "devices": [],
        "note": "MIDI device access is a browser-only capability (Web MIDI "
                "API). The backend cannot enumerate a client's MIDI devices. "
                "The frontend must call navigator.requestMIDIAccess() "
                "directly and is responsible for device listing and sending.",
    }


@app.post("/devices/midi/send")
def send_midi_device():
    raise HTTPException(
        status_code=501,
        detail="MIDI device send must happen client-side via the Web MIDI "
               "API; this backend has no access to browser MIDI hardware "
               "and cannot implement this endpoint meaningfully.",
    )


@app.get("/health")
def health():
    return {"status": "ok", "time": time.time()}
