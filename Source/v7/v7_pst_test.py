import torch
import torch.nn as nn
import numpy as np
import random

# --------------------------
# 1. VAE (Latent Model) - reduced for speed
# --------------------------
class VAE(nn.Module):
    def __init__(self, input_dim=128*32, latent_dim=32):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 128)
        self.fc4 = nn.Linear(128, input_dim)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def train_vae(model, epochs=30):
    data = []
    for _ in range(300):
        seq = np.zeros((128, 32))
        for t in range(32):
            if random.random() > 0.7:
                pitch = 60 + random.choice([0,2,4,5,7,9,11]) + random.randint(0, 12)
                if 0 <= pitch < 128:
                    seq[pitch, t] = 1.0
        data.append(seq.flatten())
    X = torch.tensor(np.array(data), dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        optimizer.zero_grad()
        recon, mu, logvar = model(X)
        BCE = nn.functional.binary_cross_entropy(recon, X, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        (BCE + KLD).backward()
        optimizer.step()

# --------------------------
# 2. Prediction Suffix Tree
# --------------------------
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
                    context = tuple(seq[i-order : i])
                    next_sym = seq[i]
                    if context not in self.pst:
                        self.pst[context] = {}
                    self.pst[context][next_sym] = self.pst[context].get(next_sym, 0) + 1
                    self.all_symbols.add(next_sym)
                    self.all_symbols.add(seq[i-1])

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
        ["C","G","Am","F","C","G","C"],
        ["C","Am","F","G","C"],
        ["Am","F","C","G","Am"],
        ["F","C","G","Am","F"],
        ["C","F","G","C"],
        ["G","C","Am","F","G"],
        ["Dm","G","C","Am","Dm"],
        ["Em","Am","F","C","Em"],
    ]
    pst = PredictionSuffixTree(max_order=3, min_count=1)
    pst.train(corpus)
    return pst

# --------------------------
# 3. DownwashResolver
# --------------------------
class DownwashResolver:
    def __init__(self, alpha=1.2, beta=1.8, gamma=1.5, delta=0.4,
                 max_repeats=2, variable_rhythm=True, pst=None):
        self.chords = {
            "C": [60,64,67], "G": [55,59,62], "Am": [57,60,64],
            "F": [53,57,60], "Em": [52,55,59], "Dm": [50,53,57]
        }
        self.alpha, self.beta, self.gamma, self.delta = alpha, beta, gamma, delta
        self.max_repeats = max_repeats
        self.variable_rhythm = variable_rhythm
        self.rhythm_patterns = [[4,4,4,4],[2,2,4,4],[4,2,2,4],[8,4,2,2],[3,3,2,4]]
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
        target = (self.current_pattern[self.pattern_idx] if self.variable_rhythm else 4)
        if self.chord_step >= target:
            self.chord_step = 0
            self.next_chord()
            if self.variable_rhythm:
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
        neighbors = [raw_pitch + d for d in (-2, -1, 1, 2)]
        candidates = list({raw_pitch, snapped, *neighbors})
        best_pitch, best_cost = None, float("inf")
        for cand in candidates:
            leap = abs(cand - self.last_pitch) if self.last_pitch is not None else 0
            leap_cost = 0.0 if leap <= 2 else (leap - 2) * 1.4
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
                if 1 <= interval <= 2:
                    candidates.append(target)
        if not candidates:
            scale = [60, 62, 64, 65, 67, 69, 71]
            for tone in scale:
                for oct_off in range(-2, 3):
                    target = tone + 12 * oct_off
                    interval = abs(target - self.last_pitch) if self.last_pitch is not None else 0
                    if 1 <= interval <= 2:
                        candidates.append(target)
        if candidates:
            return random.choice(candidates)
        return (self.last_pitch + random.choice([-2, -1, 1, 2])) if self.last_pitch is not None else random.randint(60, 72)

# --------------------------
# 4. Main
# --------------------------
def main(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    print("Training VAE (fast)...")
    vae = VAE(latent_dim=32)
    train_vae(vae, epochs=20)
    print("VAE trained.")

    resolver = DownwashResolver(alpha=1.2, beta=1.8, gamma=1.5, delta=0.4)
    pitches = []

    print("Generating 120 notes...")
    for step in range(120):
        latent = torch.randn(vae.latent_dim)
        with torch.no_grad():
            decoded = vae.decode(latent.unsqueeze(0)).squeeze(0).view(128, 32)
        col = step % 32
        raw_pitch = torch.argmax(decoded[:, col]).item()
        resolver.maybe_change_chord()
        fitted = resolver.fit_pitch_to_chord(raw_pitch)
        pitches.append(fitted)

    intervals = [abs(pitches[i+1] - pitches[i]) for i in range(len(pitches)-1)]
    stepwise = sum(1 for i in intervals if i <= 2) / len(intervals) * 100
    repeats = sum(1 for i in intervals if i == 0) / len(intervals) * 100
    mean_int = float(np.mean(intervals))

    pitch_classes = [p % 12 for p in pitches]
    hist = np.bincount(pitch_classes, minlength=12) / len(pitches)
    major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    corr = float(np.corrcoef(hist, major_profile)[0,1])

    print("\n===== METRICS (seed=42) =====")
    print(f"Stepwise %: {stepwise:.1f}")
    print(f"Repeat %: {repeats:.1f}")
    print(f"Mean interval: {mean_int:.2f}")
    print(f"Key correlation: {corr:.3f}")

if __name__ == "__main__":
    main(seed=42)
