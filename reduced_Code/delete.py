from pathlib import Path

DATA_DIR = Path("data")

deleted = 0
for f in DATA_DIR.rglob("*mls*"):
    if f.is_file():
        f.unlink()
        print(f"  [del] {f}")
        deleted += 1

# Remove empty directories
for d in sorted(DATA_DIR.rglob("*"), reverse=True):
    if d.is_dir() and not any(d.iterdir()):
        d.rmdir()
        print(f"  [rmdir] {d}")

print(f"\nDeleted {deleted} files.")