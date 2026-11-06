"""Minimal rollout server around vllm.LLM (runs INSIDE the vllm docker image, one process per GPU):
    POST /generate  {"prompt_token_ids": [[...], ...], "n": 8, "temperature": 1.0, "max_tokens": 8000, "top_p": 1.0}
                    -> {"outputs": [[{"token_ids": [...], "finish_reason": "stop"|"length"}, ...] per prompt]}
    POST /reload    {"path": "<dir with model.safetensors>"}  -> {"loaded": [n_params per worker]}   (in-place hot reload)
    GET  /health
"""
import argparse, json, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--port", type=int, default=8001); ap.add_argument("--max_model_len", type=int, default=13000)
ap.add_argument("--gpu_mem", type=float, default=0.9); ap.add_argument("--max_num_seqs", type=int, default=128); ap.add_argument("--enforce_eager", action="store_true")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

from vllm import LLM, SamplingParams
llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem, max_num_seqs=a.max_num_seqs,
          enable_prefix_caching=True, seed=a.seed, generation_config="vllm", enforce_eager=a.enforce_eager,
          worker_extension_cls="innervoice_vllm.worker_ext.InnerVoiceWorkerExt")
tok = llm.get_tokenizer()
STOP = sorted({tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")})
LOCK = threading.Lock()


class H(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def _send(self, code, obj):
        body = json.dumps(obj).encode(); self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        self._send(200, {"ok": True, "model": a.model})
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        t0 = time.time()
        try:
            with LOCK:
                if self.path == "/generate":
                    sp = SamplingParams(n=req.get("n", 1), temperature=req.get("temperature", 1.0), top_p=req.get("top_p", 1.0), top_k=-1,
                                        max_tokens=req["max_tokens"], stop_token_ids=STOP, skip_special_tokens=False, seed=req.get("seed"))
                    outs = llm.generate([{"prompt_token_ids": p} for p in req["prompt_token_ids"]], sp, use_tqdm=False)
                    res = [[{"token_ids": list(o.token_ids), "finish_reason": o.finish_reason} for o in out.outputs] for out in outs]
                    self._send(200, {"outputs": res, "t": time.time() - t0})
                elif self.path == "/reload":
                    n = llm.collective_rpc("reload_weights", args=(req["path"],))
                    self._send(200, {"loaded": n, "t": time.time() - t0})
                else:
                    self._send(404, {"error": self.path})
        except Exception as e:
            import traceback; traceback.print_exc(); self._send(500, {"error": repr(e)})


print(f"SERVER_READY port={a.port}", flush=True)
ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()
