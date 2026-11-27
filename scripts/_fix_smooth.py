"""Fix the separable convolution: pad each axis separately."""

import pathlib

path = pathlib.Path("backend/density.py")
text = path.read_text(encoding="utf-8")

old = '''        padded = np.pad(field, kr, mode="constant", constant_values=0.0)

        # Convolve along x for every row.
        horizontal = np.zeros_like(field)
        for i, k in enumerate(kernel_1d):
            horizontal += k * padded[:, i:i + w]

        # Then along y.
        smoothed = np.zeros_like(field)
        vertical = np.pad(horizontal, ((kr, kr), (0, 0)), mode="constant", constant_values=0.0)
        for i, k in enumerate(kernel_1d):
            smoothed += k * vertical[i:i + h, :]

        return smoothed'''

new = '''        # Each axis is padded on its own. Padding both at once and slicing would
        # carry the other axis's padding into the slice width.
        horizontal = np.zeros_like(field)
        pad_x = np.pad(field, ((0, 0), (kr, kr)), mode="constant", constant_values=0.0)
        for i, k in enumerate(kernel_1d):
            horizontal += k * pad_x[:, i:i + w]

        smoothed = np.zeros_like(field)
        pad_y = np.pad(horizontal, ((kr, kr), (0, 0)), mode="constant", constant_values=0.0)
        for i, k in enumerate(kernel_1d):
            smoothed += k * pad_y[i:i + h, :]

        return smoothed'''

n = text.count(old)
text = text.replace(old, new)
path.write_text(text, encoding="utf-8")
print(f"fixed separable convolution: {n}")
