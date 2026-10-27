# VME v3 — Vortex Music Engine

A hybrid generative melody engine combining a variable-order Prediction
Suffix Tree (chord progression), a multi-term cost-based pitch resolver
(the "Downwash Resolver"), a constrained-random-walk raw pitch source,
and a small neural timbre synthesizer with an ADSR envelope.

This is a **finalized, honestly-scoped result**, not a claim of a fully
solved system. Read the Known Limitation section below before assuming
all four target metrics are met — they are not, and that is by design
after exhausting the reasonable search space (see below).

## Running it

```bash
pip install torch numpy scipy
python3 vme_v3.py
```

Produces `vme_v3.wav` (80 seconds, seed 42, 160 notes) and prints the
four benchmark metrics.

## Final configuration

`DownwashResolver(alpha=0.8, beta=1.8, gamma=2.5, delta=0.3, soft_leap=2.2,
leap_slope=1.4, max_repeats=2)` — found via a 1,500-point grid search
over these parameters, selected as the best available trade-off across
the four target metrics (see below).

Raw pitch source: constrained random walk (`generate_raw_pitch`), not
a VAE-argmax approach — an earlier VAE-based variant was tested and
found to produce a mean interval of ~40 semitones due to sampling an
independent random latent vector every step with no continuity, and
was discarded.

## Verified metrics (seed=42, 160 notes, reproducible across repeated runs)

| Metric | Result | Target | Met? |
|---|---|---|---|
| Stepwise motion (≤2 semitones) | 67.3% | 60–70% | ✅ |
| Repeated notes | 1.3% | <15% | ✅ |
| Mean melodic interval | 1.91 semitones | 2.5–3.5 | ❌ |
| Krumhansl-Schmuckler key correlation (best of 24 keys) | 0.741 (A minor) | >0.7 | ✅ |

**3 of 4 targets met.** Mean interval is the one open gap.

## Known limitation: mean interval, and why it's structural, not a tuning gap

A 3,000-configuration randomized search was run (varying alpha, beta,
gamma, delta, soft_leap, leap_slope, max_repeats, and two additional
leap-frequency-cap parameters explored in a later experiment) against
500-note generations. Result:

- **0 of 3,000 configurations achieved stepwise ≥60% AND mean interval
  ≥2.5 simultaneously.**
- Among configs with stepwise ≥60%, the best mean interval found was
  1.99.
- Among configs with mean interval ≥2.5, the best stepwise found was
  31.7%.

This is a genuine Pareto-frontier result for the current candidate
generation scheme (raw pitch, snapped chord tone, ±1/±2 stepwise
neighbors, and optional 4–7 semitone chord-tone leaps under a
frequency cap): **the two targets cannot both be hit within this
design.** It is not a matter of finding better weights.

Two follow-on attempts to fix this via wider "leap" candidates were
tried and both failed instructively:

1. Giving leap candidates a strong cost discount caused them to
   dominate nearly every choice, collapsing the melody into an
   arpeggio (stepwise motion fell to 3.8%, worse than the original
   pre-fix baseline).
2. A frequency-capped version of the same idea correctly prevented
   that collapse, but a bug in the cap's bookkeeping (counting all
   >2-semitone moves against the budget, rather than only the
   intended 4–7 semitone category) caused 3-semitone jumps to
   consume the leap budget before genuine wide leaps could be used.
   Fixing that one-line bug raised the mean interval to 2.23 but
   dropped stepwise motion to 49.7% — trading one metric for the
   other rather than improving both.

**Conclusion:** further weight tuning on this candidate space is a
dead end, confirmed empirically, not assumed. The mean interval of
1.91 is a real property of this system's melodic style (conservative,
mostly stepwise) rather than a defect.

## If you want to push further

The only identified path that could plausibly move the Pareto
frontier is expanding the candidate space itself — e.g., allowing
occasional 4–7 semitone "leap-and-return" contour candidates that are
not restricted to chord tones, paired with a contour-resolution cost
term that rewards stepping back after a leap. This was scoped but not
implemented or tested; it is a genuine bet, not a verified fix.
