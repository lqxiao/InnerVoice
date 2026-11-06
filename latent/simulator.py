"""Phase 4a: text-only playback simulator.

A trajectory is a list of events in wall-clock order, produced by the streaming decoder:
    {"type": "think", "dt": compute_seconds}
    {"type": "speak", "dt": compute_seconds, "text": chunk_text, "ok": bool, "meaningful": bool}
Audio model: each spoken chunk has playback duration audio_sec(text). Audio plays from a FIFO buffer.
Timeline: compute runs continuously; chunk i becomes playable at the moment its generation finishes.
Metrics:
    ttfc         time until first *meaningful* chunk begins playing
    stalls       list of silence gaps between consecutive chunks (buffer under-runs); P50/P95 reported
    gpu_time     total compute time
    hidden_steps number of think events
    premature    number of spoken chunks that were wrong at the time they were spoken (unsupported statements)
"""
from dataclasses import dataclass, field
from typing import List, Dict
import statistics

@dataclass
class AudioModel:
    sec_per_word: float = 0.35     # ~170 wpm TTS
    sec_per_char: float = 0.0      # alternative, 0 -> use words
    overhead: float = 0.15         # TTS startup / inter-sentence pause per chunk

    def duration(self, text: str) -> float:
        if self.sec_per_char > 0:
            return self.overhead + self.sec_per_char * len(text)
        return self.overhead + self.sec_per_word * max(1, len(text.split()))

@dataclass
class SimResult:
    ttfc: float
    stalls: List[float]
    gpu_time: float
    hidden_steps: int
    n_chunks: int
    premature: int
    total_audio: float
    end_time: float
    timeline: List[Dict] = field(default_factory=list)

    @property
    def p50_stall(self):
        return statistics.median(self.stalls) if self.stalls else 0.0

    @property
    def p95_stall(self):
        if not self.stalls:
            return 0.0
        s = sorted(self.stalls); i = min(len(s) - 1, int(round(0.95 * (len(s) - 1))))
        return s[i]

    def as_dict(self):
        return {"ttfc": self.ttfc, "p50_stall": self.p50_stall, "p95_stall": self.p95_stall, "max_stall": max(self.stalls) if self.stalls else 0.0,
                "n_stalls": sum(1 for s in self.stalls if s > 1e-6), "gpu_time": self.gpu_time, "hidden_steps": self.hidden_steps,
                "n_chunks": self.n_chunks, "premature": self.premature, "total_audio": self.total_audio, "end_time": self.end_time}

def simulate(events: List[Dict], audio: AudioModel = AudioModel(), think_cost_override: float = None) -> SimResult:
    """Replay events. think_cost_override: substitute a fixed per-step compute cost (for cost-sensitivity analysis)."""
    t = 0.0                 # compute clock
    play_end = 0.0          # time at which the audio buffer runs dry
    ttfc, stalls, gpu, steps, prem, total_audio, n_chunks = None, [], 0.0, 0, 0, 0.0, 0
    timeline = []
    for ev in events:
        dt = think_cost_override if (ev["type"] == "think" and think_cost_override is not None) else ev["dt"]
        t += dt; gpu += dt
        if ev["type"] == "think":
            steps += 1
            timeline.append({"t": t, "ev": "think"})
            continue
        dur = audio.duration(ev["text"])
        start = max(t, play_end)          # chunk can start when generated and previous audio done
        if n_chunks > 0:
            stalls.append(max(0.0, start - play_end))   # silence between chunks
        if ttfc is None and ev.get("meaningful", True):
            ttfc = start
        if not ev.get("ok", True):
            prem += 1
        play_end = start + dur; total_audio += dur; n_chunks += 1
        timeline.append({"t": t, "ev": "speak", "start": start, "end": play_end, "text": ev["text"], "ok": ev.get("ok", True)})
    if ttfc is None:
        ttfc = t
    return SimResult(ttfc=ttfc, stalls=stalls, gpu_time=gpu, hidden_steps=steps, n_chunks=n_chunks, premature=prem,
                     total_audio=total_audio, end_time=max(t, play_end), timeline=timeline)

def buffer_remaining(play_end: float, t: float) -> float:
    return max(0.0, play_end - t)

def aggregate(results: List[SimResult]) -> Dict:
    import numpy as np
    d = {}
    for k in ["ttfc", "gpu_time", "hidden_steps", "premature", "n_chunks", "end_time"]:
        d[f"mean_{k}"] = float(np.mean([getattr(r, k) for r in results]))
    all_stalls = [s for r in results for s in r.stalls]
    d["p50_stall"] = float(np.median(all_stalls)) if all_stalls else 0.0
    d["p95_stall"] = float(np.percentile(all_stalls, 95)) if all_stalls else 0.0
    d["p95_stall_per_traj"] = float(np.mean([r.p95_stall for r in results]))
    d["stall_rate"] = float(np.mean([s > 0.05 for s in all_stalls])) if all_stalls else 0.0
    return d

if __name__ == "__main__":
    ev = [{"type": "think", "dt": 0.05}] * 3 + [{"type": "speak", "dt": 0.3, "text": "A1: 42", "ok": True}] + [{"type": "think", "dt": 0.05}] * 8 + [{"type": "speak", "dt": 0.3, "text": "A2: Yes", "ok": False}]
    r = simulate(ev); print(r.as_dict())
