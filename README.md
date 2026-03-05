# HPLC Calibration Standards - Peak Integration Utility

This repository includes `hplc_peak_integration.py`, a reusable script for HPLC-style traces stored as comma-separated `.txt` files with no header.

## Features
- Robust loading with malformed-line handling (`on_bad_lines='skip'`)
- Automatic main-peak detection
- Area integration via trapezoidal rule **without baseline zeroing**
- Trace plotting with shaded integrated region
- Modular functions for reuse in notebooks or other scripts

## Install (recommended)
Install in editable mode so imports work from any folder:

```bash
python -m pip install -e .
```

After this, these both work globally:

```bash
hplc-integrate /path/to/trace.txt
python -m hplc_peak_integration /path/to/trace.txt
```

## Run as script (no install)
From the repository root:

```bash
python hplc_peak_integration.py /path/to/trace.txt
```

## Reuse from Python
```python
from hplc_peak_integration import analyze_hplc_file, plot_peak_integration
import matplotlib.pyplot as plt

trace, region, area = analyze_hplc_file("sample.txt")
fig, ax = plot_peak_integration(
    trace["x"].to_numpy(),
    trace["y"].to_numpy(),
    region,
    area,
)
plt.show()
```

## Troubleshooting
If you see `ModuleNotFoundError: No module named 'hplc_peak_integration'`, it usually means Python cannot find this repository on `PYTHONPATH`.

Use one of these fixes:
1. Run commands from the repo root, **or**
2. Install with `python -m pip install -e .` (recommended).
