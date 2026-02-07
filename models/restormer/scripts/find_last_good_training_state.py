#!/usr/bin/env python3
import os, sys, torch

root = sys.argv[1] if len(sys.argv) > 1 else 'experiments/UNet_Restormer_Hiragana_Run01/training_states'
if not os.path.isdir(root):
    print('ROOT_NOT_FOUND', root)
    sys.exit(2)

files = sorted([os.path.join(root,f) for f in os.listdir(root) if f.endswith('.pth')], key=os.path.getmtime, reverse=True)
if not files:
    print('NO_PTH_FILES', root)
    sys.exit(1)

bad = []
for p in files:
    try:
        print('TRY', p)
        ckpt = torch.load(p, map_location='cpu')
        print('GOOD', p)
        sys.stdout.flush()
        # Print the good file and exit with 0
        print('LAST_GOOD', p)
        sys.exit(0)
    except Exception as e:
        print('BAD', p, repr(e))
        bad.append(p)

print('NO_VALID_CHECKPOINT')
sys.exit(3)
