#!/usr/bin/env python3
"""
Patches coqui-tts to work with transformers >= 4.46.

The function `isin_mps_friendly` was removed from transformers.pytorch_utils
in v4.46. coqui-tts 0.27.x imports it at module load time from a file called
tacotron2.py. This script adds it back as a local shim so the import works.
"""
import sys
from pathlib import Path

# Find the venv site-packages
venv_site = None
for p in sys.path:
    if "site-packages" in p:
        venv_site = Path(p)
        break

if not venv_site:
    print("❌ Could not find site-packages in sys.path")
    sys.exit(1)

print(f"site-packages: {venv_site}")

# ── Fix 1: add isin_mps_friendly back to transformers.pytorch_utils ──────────
pytorch_utils = venv_site / "transformers" / "pytorch_utils.py"
if not pytorch_utils.exists():
    print(f"❌ Not found: {pytorch_utils}")
    sys.exit(1)

content = pytorch_utils.read_text()
shim = '''
# Shim added by patch_coqui.py — removed in transformers 4.46, still needed by coqui-tts
def isin_mps_friendly(elements, test_elements):
    """MPS-compatible replacement for torch.isin (added as shim for coqui-tts)."""
    import torch
    if elements.device.type == "mps":
        return torch.stack([elements == t for t in test_elements.flatten()]).any(dim=0)
    return torch.isin(elements, test_elements)
'''

if "isin_mps_friendly" in content:
    print("✅ isin_mps_friendly already present in pytorch_utils.py — no patch needed")
else:
    pytorch_utils.write_text(content + shim)
    print(f"✅ Patched: {pytorch_utils}")

# ── Verify the import now works ───────────────────────────────────────────────
print("\nVerifying import...")
try:
    from TTS.api import TTS
    print("✅ from TTS.api import TTS  — SUCCESS")
except Exception as e:
    print(f"❌ Still failing: {e}")
    print("\nRun this to see the full traceback:")
    print("  python -c \"from TTS.api import TTS\" 2>&1 | head -40")
