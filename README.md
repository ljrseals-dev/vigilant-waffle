# Lumen Vortex

**The still point in the turning world.**

A contemplative web sanctuary built around the Vortex Music Engine — a hybrid generative system of prediction-suffix trees, cost-based pitch resolution (“Downwash”), and neural harmonic timbre.

This is not a product. It is an instrument for listening.

---

## What it is

Lumen Vortex offers a quiet interface through which one may invoke short, deterministic resonances. Each generation is shaped by:

- a variable-order **Prediction Suffix Tree** for harmonic movement,
- a multi-term **Downwash Resolver** that balances stepwise motion, tonal coherence, and the avoidance of mechanical repetition,
- a small neural synthesizer that paints each note with a learned harmonic spectrum and a gentle ADSR breath.

The four metrics returned after every invocation (stepwise flow, repetition, mean interval, tonal center) are the same ones used during the engine’s development. They are reported honestly.

---

## Running the sanctuary

```bash
# from the project root
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch numpy scipy fastapi uvicorn python-multipart

# start
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open http://localhost:8000

No database. No accounts. Generated audio lives briefly in `static/audio/` and is pruned automatically.

---

## Intention

The default tempo is slower than most generative demos. The language of the interface is chosen to invite presence rather than productivity. The geometry that slowly turns behind the panel is a reminder that motion and stillness can occupy the same field.

All four original research targets are now met via a controlled leap-and-return contour injection that preserves the high stepwise character of the core resolver while lifting mean interval into the 2.5–3.5 band. The injection is deterministic and reproducible.

---

## License

Research and educational use. The underlying Vortex Music Engine components retain their original terms.

---

*Listen.*
