"""vLLM general plugin registering the InnerVoice architecture (loaded in every engine/worker process)."""
ARCH = "InnerVoiceQwen3ForCausalLM"

def register():
    from vllm import ModelRegistry
    if ARCH not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(ARCH, "innervoice_vllm.model:InnerVoiceQwen3ForCausalLM")
