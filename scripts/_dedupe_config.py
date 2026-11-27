"""Remove the duplicated ensemble config that landed inside reset_forecast."""

import pathlib

path = pathlib.Path("backend/prediction.py")
lines = path.read_text(encoding="utf-8").split("\n")

# Find the second occurrence of the config block and delete it.
marker = "        # Uncertainty is reported instead of being hidden. Each replicate re-runs the"
positions = [i for i, ln in enumerate(lines) if marker in ln]
print(f"config block found at line indices: {positions}")

if len(positions) >= 2:
    start = positions[1]
    # The block runs from the comment to `self.final_sigma = None`.
    end = None
    for i in range(start, min(start + 20, len(lines))):
        if "self.final_sigma = None" in lines[i]:
            end = i
            break
    if end is not None:
        print(f"removing lines {start}..{end}")
        del lines[start:end + 1]

text = "\n".join(lines)
path.write_text(text, encoding="utf-8")
print(f"remaining 'ensemble_replicates = 0': {text.count('self.ensemble_replicates = 0')}")
