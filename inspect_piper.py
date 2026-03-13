"""Run this to show every public method on PiperVoice and the full source."""
import inspect
from piper import PiperVoice

print("=== PiperVoice public methods ===")
for name, obj in inspect.getmembers(PiperVoice, predicate=inspect.isfunction):
    if not name.startswith("_"):
        sig = inspect.signature(obj)
        print(f"  {name}{sig}")

print("\n=== Full source of voice.py ===")
print(inspect.getsource(inspect.getmodule(PiperVoice)))