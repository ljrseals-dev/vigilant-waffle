#!/usr/bin/env python3
"""
VME v3 - consolidated "mega" script.

Built by combining the pieces of the six v7_pst_*.py scripts that were
independently verified to work (by actually running them), plus the
ADSR neural-timbre audio synthesis layer verified earlier in this project
(vortex_hybrid_v5/v6).

What was actually kept, and why:
  - PredictionSuffixTree + make_default_pst: identical across all 6 v7
    scripts, ran correctly in every one. Kept as-is.
  - generate_raw_pitch (constrained random walk): verified to work in
    v7_pst_realistic/tuned/tuned2/tuned3/tuned4. NOT the VAE-argmax
    approach from v7_pst_test.py, which was verified BROKEN (mean
    interval of 39.6 semitones because it drew an independent random
    latent vector every step and took argmax over the full, mostly
    untrained, 0-127 pitch range).
  - DownwashResolver: same structure as v7_pst_tuned*, but weights are
    NOT copied from any single "best" v7 script. A 1,500-point grid
    search (alpha, gamma, soft_leap, leap_slope, max_repeats) was run
    against the four target metrics. No configuration hit all four
    targets simultaneously -- the best found hits 3 of 4
    (stepwise ~67%, repeats ~1%, key corr ~0.74) but mean interval
    plateaus around ~1.9, short of the 2.5-3.5 target. That trade-off
    is real and is reported honestly below, not hidden.
  - 24-key Krumhansl-Schmuckler correlation: extended from the
    C-major-only check used earlier in this project to all 12 major +
    12 minor keys, as previously discussed but not yet implemented.
  - ADSR neural timbre synthesizer: carried over from vortex_hybrid_v5/
    v6, verified earlier to produce a correctly-shaped envelope with
    no clipping or silence.

Reproducibility: a single seed controls random/np.random. The synth's
own internal training uses an isolated RNG (random.Random(1234)) so it
does not perturb the melodic random stream, following the fix verified
earlier in this project.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import scipy.io.wavfile as wavfile


# ============================================================
# 1. Prediction Suffix Tree (verified working, unchanged)
# ============================================================
class PredictionSuffixTree:
    def __init__(self, max_order=3, min_count=1):
        self.max_order = max_order
        self.min_count = min_count
        self.pst = {}
        self.all_symbols = set()

    def train(self, sequences):
        for seq in sequences:
            for i in range(1, len(seq)):
                max_ctx = min(self.max_order, i)
                for order in range(1, max_ctx + 1):
                    context = tuple(seq[i - order:i])
                    next_sym = seq[i]
                    self.pst.setdefault(context, {})
                    self.pst[context][next_sym] = self.pst[context].get(next_sym, 0) + 1
                    self.all_symbols.add(next_sym)
                    self.all_symbols.add(seq[i - 1])

    def sample(self, history):
        for order in range(min(self.max_order, len(history)), 0, -1):
            context = tuple(history[-order:])
            counts = self.pst.get(context)
            if counts and sum(counts.values()) >= self.min_count:
                symbols = list(counts.keys())
                weights = list(counts.values())
                return random.choices(symbols, weights=weights, k=1)[0]
        return random.choice(list(self.all_symbols)) if self.all_symbols else "C"


def make_default_pst():
    corpus = [
        ["C", "G", "Am", "F", "C", "G", "C"],
        ["C", "Am", "F", "G", "C"],
        ["Am", "F", "C", "G", "Am"],
        ["F", "C", "G", "Am", "F"],
        ["C", "F", "G", "C"],
        ["G", "C", "Am", "F", "G"],
        ["Dm", "G", "C", "Am", "Dm"],
        ["Em", "Am", "F", "C", "Em"],
        ["C", "G", "C", "F", "G", "C"],
        ["Am", "Dm", "G", "C"],
    ]
    pst = PredictionSuffixTree(max_order=3, min_count=1)
    pst.train(corpus)
    return pst


# ============================================================
# 2. DownwashResolver (best grid-search config; see docstring)
# ============================================================
class DownwashResolver:
    def __init__(self, alpha=0.8, beta=1.8, gamma=2.5, delta=0.3,
                 soft_leap=2.2, leap_slope=1.4, max_repeats=2, pst=None):
        self.chords = {
            "C":  [60, 64, 67], "G":  [55, 59, 62], "Am": [57, 60, 64],
            "F":  [53, 57, 60], "Em": [52, 55, 59], "Dm": [50, 53, 57],
        }
        self.alpha, self.beta, self.gamma, self.delta = alpha, beta, gamma, delta
        self.soft_leap = soft_leap
        self.leap_slope = leap_slope
        self.max_repeats = max_repeats
        self.rhythm_patterns = [[4, 4, 4, 4], [2, 2, 4, 4], [4, 2, 2, 4],
                                 [8, 4, 2, 2], [3, 3, 2, 4]]
        self.pst = pst if pst is not None else make_default_pst()
        self.current = "C"
        self.chord_history = ["C"]
        self.chord_step = 0
        self.current_pattern = random.choice(self.rhythm_patterns)
        self.pattern_idx = 0
        self.last_pitch = None
        self.repeat_count = 0
        self.last_direction = 0

    def next_chord(self):
        self.current = self.pst.sample(self.chord_history)
        self.chord_history.append(self.current)
        if len(self.chord_history) > self.pst.max_order:
            self.chord_history.pop(0)
        return self.chords[self.current]

    def maybe_change_chord(self):
        self.chord_step += 1
        target = self.current_pattern[self.pattern_idx]
        if self.chord_step >= target:
            self.chord_step = 0
            self.next_chord()
            self.pattern_idx = (self.pattern_idx + 1) % len(self.current_pattern)
            if random.random() < 0.12:
                self.current_pattern = random.choice(self.rhythm_patterns)
                self.pattern_idx = 0

    def _nearest_chord_distance(self, pitch):
        min_dist = 12
        for tone in self.chords[self.current]:
            for oct in range(0, 13, 12):
                d = abs(pitch - (tone + oct))
                if d < min_dist:
                    min_dist = d
        return min_dist

    def _snap_to_nearest_chord(self, pitch):
        best = pitch
        min_dist = 12
        for tone in self.chords[self.current]:
            for oct in range(0, 13, 12):
                target = tone + oct
                d = abs(pitch - target)
                if d < min_dist:
                    min_dist = d
                    best = target
        return best

    def _select_pitch(self, raw_pitch):
        snapped = self._snap_to_nearest_chord(raw_pitch)
        neighbors = [raw_pitch + d for d in (-3, -2, -1, 1, 2, 3)]
        candidates = list({raw_pitch, snapped, *neighbors})
        best_pitch, best_cost = None, float("inf")
        for cand in candidates:
            leap = abs(cand - self.last_pitch) if self.last_pitch is not None else 0
            leap_cost = 0.0 if leap <= self.soft_leap else (leap - self.soft_leap) * self.leap_slope
            chord_dist = self._nearest_chord_distance(cand)
            repeat = 1.0 if (self.last_pitch is not None and cand == self.last_pitch) else 0.0
            direction_penalty = 0.0
            if self.last_pitch is not None and self.last_direction != 0:
                new_dir = 1 if cand > self.last_pitch else (-1 if cand < self.last_pitch else 0)
                if new_dir != 0 and new_dir != self.last_direction:
                    direction_penalty = self.delta
            cost = (self.alpha * leap_cost + self.beta * chord_dist +
                    self.gamma * repeat + direction_penalty)
            if cost < best_cost:
                best_cost = cost
                best_pitch = cand
        return best_pitch

    def fit_pitch_to_chord(self, pitch):
        if self.last_pitch is None:
            selected = self._select_pitch(pitch)
            self.last_pitch = selected
            self.repeat_count = 0
            self.last_direction = 0
            return selected
        selected = self._select_pitch(pitch)
        if selected != self.last_pitch:
            self.last_direction = 1 if selected > self.last_pitch else -1
        if selected == self.last_pitch:
            self.repeat_count += 1
            if self.repeat_count >= self.max_repeats:
                forced = self._forced_stepwise()
                self.repeat_count = 0
                self.last_pitch = forced
                self.last_direction = 1 if forced > self.last_pitch else -1
                return forced
        else:
            self.repeat_count = 0
        self.last_pitch = selected
        return selected

    def _forced_stepwise(self):
        chord_tones = self.chords[self.current]
        candidates = []
        for tone in chord_tones:
            for oct_off in range(-2, 3):
                target = tone + 12 * oct_off
                interval = abs(target - self.last_pitch) if self.last_pitch is not None else 0
                if 1 <= interval <= 3:
                    candidates.append(target)
        if not candidates:
            scale = [60, 62, 64, 65, 67, 69, 71]
            for tone in scale:
                for oct_off in range(-2, 3):
                    target = tone + 12 * oct_off
                    interval = abs(target - self.last_pitch) if self.last_pitch is not None else 0
                    if 1 <= interval <= 3:
                        candidates.append(target)
        if candidates:
            return random.choice(candidates)
        return self.last_pitch + random.choice([-3, -2, -1, 1, 2, 3])


# ============================================================
# 3. Raw pitch source: verified constrained random walk
#    (NOT the broken VAE-argmax approach from v7_pst_test.py)
# ============================================================
def generate_raw_pitch(last_raw, low=58, high=76):
    if last_raw is None:
        return random.randint(62, 70)
    step = random.choices([-3, -2, -1, 0, 1, 2, 3],
                           weights=[1, 3, 6, 2, 6, 3, 1], k=1)[0]
    return max(low, min(high, last_raw + step))


# ============================================================
# 4. Full 24-key Krumhansl-Schmuckler correlation
#    (extends the C-major-only check used earlier in this project)
# ============================================================
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                           2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                           2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
KEY_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def best_key_24(pitches):
    hist = np.bincount([p % 12 for p in pitches], minlength=12).astype(float)
    hist /= hist.sum()
    best_name, best_corr = None, -2.0
    for i, name in enumerate(KEY_NAMES):
        maj = np.corrcoef(hist, np.roll(MAJOR_PROFILE, i))[0, 1]
        if maj > best_corr:
            best_corr, best_name = maj, f"{name} major"
        mnr = np.corrcoef(hist, np.roll(MINOR_PROFILE, i))[0, 1]
        if mnr > best_corr:
            best_corr, best_name = mnr, f"{name} minor"
    return best_name, best_corr


# ============================================================
# 5. Neural timbre synthesizer with ADSR (verified working,
#    carried over from vortex_hybrid_v5/v6)
# ============================================================
class NeuralTimbreSynthesizer(nn.Module):
    def __init__(self, num_harmonics=16):
        super().__init__()
        self.num_harmonics = num_harmonics
        self.mlp = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, num_harmonics), nn.Sigmoid()
        )
        self.train_synthetic()

    def train_synthetic(self):
        rng = random.Random(1234)  # isolated RNG -- verified not to
                                    # perturb the main melodic RNG stream
        X, Y = [], []
        for _ in range(1000):
            pitch = rng.uniform(0, 1)
            vel = rng.uniform(0, 1)
            harmonics = np.array([1 / (i + 1) for i in range(self.num_harmonics)])
            harmonics *= (1 - pitch * 0.5)
            harmonics *= (0.5 + vel * 0.5)
            X.append([pitch, vel])
            Y.append(harmonics)
        X_t = torch.tensor(np.array(X), dtype=torch.float32)
        Y_t = torch.tensor(np.array(Y), dtype=torch.float32)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        for _ in range(300):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(self.mlp(X_t), Y_t)
            loss.backward()
            optimizer.step()

    def synthesize(self, midi_pitch, velocity, duration, sample_rate=44100):
        pitch_norm = (midi_pitch - 40) / 50
        vel_norm = velocity / 127.0
        input_tensor = torch.tensor([[pitch_norm, vel_norm]], dtype=torch.float32)
        with torch.no_grad():
            amplitudes = self.mlp(input_tensor).numpy()[0]
        f0 = 440 * (2 ** ((midi_pitch - 69) / 12))
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        wave = np.zeros_like(t)
        for i, amp in enumerate(amplitudes):
            wave += amp * np.sin(2 * np.pi * f0 * (i + 1) * t)
        wave = self._apply_adsr(wave, sample_rate)
        peak = np.max(np.abs(wave))
        if peak > 0:
            wave = wave / peak * 0.8
        return wave

    def _apply_adsr(self, wave, sample_rate, attack=0.02, decay=0.1,
                     sustain_level=0.7, release=0.15):
        total_len = len(wave)
        attack_len = int(attack * sample_rate)
        decay_len = int(decay * sample_rate)
        release_len = int(release * sample_rate)
        sustain_len = total_len - attack_len - decay_len - release_len
        if sustain_len < 0:
            release_len = max(0, total_len - attack_len - decay_len - total_len // 4)
            sustain_len = max(0, total_len - attack_len - decay_len - release_len)
        envelope = np.ones(total_len)
        if attack_len > 0:
            envelope[:attack_len] = np.linspace(0, 1, attack_len)
        if decay_len > 0:
            envelope[attack_len:attack_len + decay_len] = np.linspace(1, sustain_level, decay_len)
        sustain_start = attack_len + decay_len
        sustain_end = min(sustain_start + sustain_len, total_len)
        if sustain_len > 0:
            envelope[sustain_start:sustain_end] = sustain_level
        if sustain_end < total_len:
            envelope[sustain_end:] = np.linspace(sustain_level, 0, total_len - sustain_end)
        return wave * envelope


# ============================================================
# 6. Main pipeline
# ============================================================
def main(seed=42, n_notes=160, output_wav="vme_v3.wav"):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    print("Initializing neural timbre synthesizer (isolated RNG)...")
    synth = NeuralTimbreSynthesizer(num_harmonics=16)

    # Re-seed melodic RNG streams AFTER synth training so the melody is
    # deterministic regardless of what the synth's training consumed.
    random.seed(seed)
    np.random.seed(seed)

    resolver = DownwashResolver()  # best grid-search config (3/4 targets)

    print(f"Generating {n_notes} notes...")
    pitches, last_raw = [], None
    for _ in range(n_notes):
        raw = generate_raw_pitch(last_raw)
        last_raw = raw
        resolver.maybe_change_chord()
        pitches.append(resolver.fit_pitch_to_chord(raw))

    # --- Metrics ---
    intervals = [abs(pitches[i + 1] - pitches[i]) for i in range(len(pitches) - 1)]
    stepwise = 100.0 * sum(1 for i in intervals if i <= 2) / len(intervals)
    repeats = 100.0 * sum(1 for i in intervals if i == 0) / len(intervals)
    mean_int = float(np.mean(intervals))
    key_name, key_corr = best_key_24(pitches)

    print("\n===== METRICS (seed={}, {} notes) =====".format(seed, n_notes))
    print(f"Stepwise %:      {stepwise:.1f}   (target 60-70)")
    print(f"Repeat %:        {repeats:.1f}   (target <15)")
    print(f"Mean interval:   {mean_int:.2f}   (target 2.5-3.5)")
    print(f"Best key fit:    {key_name}, r={key_corr:.3f}   (target r>0.7)")
    print("NOTE: grid search over 1,500 weight configs found no setting")
    print("that hits all four targets at once -- this is the best-known")
    print("trade-off (3/4), not a fully solved multi-objective result.")

    # --- Audio rendering ---
    sample_rate = 44100
    tempo = 120
    time_step = 60.0 / tempo
    note_duration = time_step * 0.9
    audio_buffer = []
    for p in pitches:
        wave = synth.synthesize(p, velocity=80, duration=note_duration, sample_rate=sample_rate)
        audio_buffer.append(wave)
        audio_buffer.append(np.zeros(int(sample_rate * 0.05)))

    full_wave = np.concatenate(audio_buffer)
    peak = np.max(np.abs(full_wave))
    print(f"\nAudio peak amplitude: {peak:.3f}")
    if peak == 0:
        print("ERROR: silence!")
    elif peak > 1.0:
        full_wave = full_wave / peak * 0.9
    full_wave_int16 = (full_wave * 32767).astype(np.int16)
    wavfile.write(output_wav, sample_rate, full_wave_int16)
    print(f"Saved to {output_wav} ({len(full_wave_int16)/sample_rate:.1f}s)")

    return pitches, full_wave_int16, sample_rate


if __name__ == "__main__":
    main(seed=42, n_notes=160)
