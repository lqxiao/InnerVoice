"""Upload an exported InnerVoice model dir (+ README) to the Hugging Face Hub.
    python hf_release/upload.py --dir /path/export --repo liqiangx/InnerVoice-Qwen3-4B [--readme hf_release/README.md] [--only_readme]"""
import argparse, os
from huggingface_hub import HfApi
ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--repo", required=True); ap.add_argument("--readme", default="")
ap.add_argument("--only_readme", action="store_true"); ap.add_argument("--token_file", default=os.path.expanduser("~/hugingface.key")); ap.add_argument("--private", action="store_true")
a = ap.parse_args()
api = HfApi(token=open(a.token_file).read().strip())
api.create_repo(a.repo, repo_type="model", exist_ok=True, private=a.private)
if not a.only_readme:
    api.upload_folder(folder_path=a.dir, repo_id=a.repo, repo_type="model", ignore_patterns=[".*", "*.tmp", "README.md"], commit_message="InnerVoice SFT weights (merged, HF remote code)")
if a.readme:
    api.upload_file(path_or_fileobj=a.readme, path_in_repo="README.md", repo_id=a.repo, repo_type="model", commit_message="model card")
print("uploaded ->", f"https://huggingface.co/{a.repo}")
