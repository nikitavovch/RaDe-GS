import numpy as np
import sys

npz_path = sys.argv[1] if len(sys.argv) > 1 else '/workspace/out/Text23D_exp_spatialgen/Text23D_exp_spatialgen_16view_sample_0/val/scene_00000/inference_results.npz'

npz = np.load(npz_path, allow_pickle=True)
print('=' * 60)
print('Файлы в NPZ:')
print('=' * 60)
for key in npz.files:
    print(f'  - {key}')

print()
print('=' * 60)
print('Структура данных:')
print('=' * 60)
key = npz.files[0]
data = npz[key].item()
print(f'\nКлючи в данных ({key}):')
for k in sorted(data.keys()):
    v = data[k]
    if hasattr(v, 'shape'):
        print(f'  {k:30s}: shape={str(v.shape):20s} dtype={v.dtype}')
    elif isinstance(v, list) and len(v) > 0:
        if hasattr(v[0], 'shape'):
            print(f'  {k:30s}: list[{len(v)}], first shape={v[0].shape}')
        else:
            print(f'  {k:30s}: list[{len(v)}], type={type(v[0])}')
    else:
        print(f'  {k:30s}: type={type(v).__name__}')
