"""Worker extension (vLLM `worker_extension_cls`): in-place weight hot-reload from a safetensors directory, called through
LLM.collective_rpc("reload_weights", args=(path,)). Parameters are updated in place so CUDA graphs stay valid."""
import glob, os, torch


class InnerVoiceWorkerExt:
    def reload_weights(self, path: str) -> int:
        from safetensors.torch import safe_open
        files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        assert files, f"no safetensors in {path}"
        def it():
            for f in files:
                with safe_open(f, "pt", device="cpu") as sf:
                    for k in sf.keys():
                        yield k, sf.get_tensor(k)
        loaded = self.model_runner.model.load_weights(it())
        torch.cuda.synchronize()
        return len(loaded)
