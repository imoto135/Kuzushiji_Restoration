#!/usr/bin/env python3
"""
Check PyTorch checkpoint files (.state / .pth) for loadability.

Usage: python scripts/check_states_torch.py /absolute/or/relative/path/to/dir

Prints one line per file: GOOD or BAD, filename, size, and some top-level keys (if available).
"""
import os
import sys
import traceback

def human_size(n):
    for unit in ['B','KB','MB','GB','TB']:
        if n < 1024.0:
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PB"

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/check_states_torch.py /path/to/dir")
        sys.exit(2)
    d = sys.argv[1]
    if not os.path.isdir(d):
        print("Not a directory:", d)
        sys.exit(2)
    files = sorted([os.path.join(d,f) for f in os.listdir(d) if f.endswith('.state') or f.endswith('.pth')])
    if not files:
        print("NO_STATE_FILES", d)
        sys.exit(1)

    # Import torch lazily so that script can print a clear error if env lacks it
    try:
        import torch
    except Exception as e:
        print("TORCH_IMPORT_FAILED:", e)
        sys.exit(3)

    for p in files:
        try:
            sz = os.path.getsize(p)
            print(f"CHECKING {p} ({human_size(sz)})...")
            obj = torch.load(p, map_location='cpu')
            keys = None
            if isinstance(obj, dict):
                keys = list(obj.keys())
            elif hasattr(obj, '__dict__'):
                keys = list(obj.__dict__.keys())
            else:
                keys = [type(obj).__name__]
            print("GOOD", p, human_size(sz), "keys:", keys[:20])
        except Exception as e:
            print("BAD", p, type(e).__name__, str(e))
            traceback.print_exc()

if __name__ == '__main__':
    main()
