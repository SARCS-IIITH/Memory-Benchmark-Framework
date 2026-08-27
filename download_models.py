import yaml
import os
import subprocess

# ⚠️ Change this to the mount path of your external SSD in WSL
# (e.g., if your external SSD is the D: drive in Windows, use "/mnt/d/ai-models")
EXTERNAL_SSD_PATH = "/mnt/d/ai-models"

# Tell Hugging Face to cache all downloads to the external SSD
os.environ["HF_HOME"] = EXTERNAL_SSD_PATH

# Enable Rust-based fast downloading
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

YAML_FILE = "configs/model-registry.yaml"

def main():
    print(f"Reading model registry from {YAML_FILE}...")
    with open(YAML_FILE, "r") as f:
        registry = yaml.safe_load(f)
    models = registry.get("models", {})
    
    for key, info in models.items():
        repo = info.get("repo")
        if not repo:
            continue
            
        print(f"\n🚀 Downloading {key} ({repo})...")
        
        # Download the repository using huggingface-cli.
        # We exclude legacy formats (.bin, .pt) to save space and ensure we only pull
        # the BF16 .safetensors which are the default for these modern models.
        cmd = [
            "huggingface-cli", "download", repo,
            "--exclude", "*.bin", "*.pt", "*.h5", "*.msgpack"
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"✅ Successfully downloaded {repo}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to download {repo}. Error: {e}")

if __name__ == "__main__":
    # Ensure the target directory exists
    os.makedirs(EXTERNAL_SSD_PATH, exist_ok=True)
    main()
