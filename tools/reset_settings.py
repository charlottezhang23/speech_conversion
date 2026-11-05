"""Delete the runtime settings file so defaults are regenerated."""
from pathlib import Path

p = Path(__file__).resolve().parent.parent / "config" / "settings.json"
if p.exists():
    p.unlink()
print("settings reset:", not p.exists())
