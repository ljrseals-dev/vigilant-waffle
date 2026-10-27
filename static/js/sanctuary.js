(() => {
  const $ = (sel) => document.querySelector(sel);

  const notesSlider = $("#notes");
  const tempoSlider = $("#tempo");
  const notesVal = $("#notes-val");
  const tempoVal = $("#tempo-val");
  const invokeBtn = $("#invoke");
  const btnText = $(".btn-text");
  const btnSpinner = $(".btn-spinner");
  const result = $("#result");

  notesSlider.addEventListener("input", () => {
    notesVal.textContent = notesSlider.value;
  });
  tempoSlider.addEventListener("input", () => {
    tempoVal.textContent = tempoSlider.value;
  });

  const blessings = {
    stillness: "May this silence between the notes hold you.",
    devotion: "Offered like incense — rising, dissolving.",
    wonder: "Listen as if the cosmos itself is breathing.",
    release: "What is held can be set free in sound.",
    awakening: "The vortex turns; the center remains.",
  };

  invokeBtn.addEventListener("click", async () => {
    const seed = parseInt($("#seed").value, 10) || 42;
    const notes = parseInt(notesSlider.value, 10);
    const tempo = parseFloat(tempoSlider.value);
    const intention = $("#intention").value;

    invokeBtn.disabled = true;
    btnText.classList.add("hidden");
    btnSpinner.classList.remove("hidden");
    result.classList.add("hidden");

    try {
      const res = await fetch("/api/resonate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ seed, notes, tempo, intention }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Resonance failed");
      }

      const data = await res.json();

      $("#m-step").textContent = `${data.stepwise_pct}%`;
      $("#m-rep").textContent = `${data.repeat_pct}%`;
      $("#m-int").textContent = data.mean_interval;
      $("#m-key").textContent = `${data.key_name} (${data.key_correlation})`;

      const audio = $("#audio");
      audio.src = data.audio_url;
      audio.load();
      $("#duration").textContent = `${data.duration_sec}s · seed ${data.seed}`;

      $("#blessing").textContent = blessings[intention] || blessings.stillness;

      result.classList.remove("hidden");
    } catch (e) {
      alert("The vortex could not open: " + e.message);
    } finally {
      invokeBtn.disabled = false;
      btnText.classList.remove("hidden");
      btnSpinner.classList.add("hidden");
    }
  });
})();
