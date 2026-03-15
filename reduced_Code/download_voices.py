import requests
from pathlib import Path

BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
INDEX_URL = f"{BASE_URL}/voices.json"
OUT_DIR = Path("piper-voices-de")

def download_file(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [skip] {dest}")
        return
    print(f"  [dl]   {dest}")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

print("Fetching voice index...")
voices = requests.get(INDEX_URL).json()

file_paths = [
    rel_path
    for key, voice in voices.items()
    if voice.get("language", {}).get("family") == "de"
    for rel_path in voice["files"].keys()  # keys are the paths!
]

print(f"Found {len(file_paths)} German files\n")

for rel_path in file_paths:
    download_file(f"{BASE_URL}/{rel_path}", OUT_DIR / rel_path)

print("\nDone!")