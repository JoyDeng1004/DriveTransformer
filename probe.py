# probe.py
import os, sys, inspect
sys.path.insert(0, os.getcwd())
import adzoo.drivetransformer.mmdet3d_plugin

from mmcv import Config
from mmcv.datasets import build_dataset, build_dataloader
from mmcv.models import build_model

CONFIG_PATH = 'adzoo/drivetransformer/configs/drivetransformer/drivetransformer_large.py'

cfg = Config.fromfile(CONFIG_PATH)
dataset = build_dataset(cfg.data.val)
loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0,
                           dist=False, shuffle=False)

# 1. 看 model.forward 签名
model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
print('=== model.forward signature ===')
print(inspect.signature(model.forward))
print('=== model.forward source (first 30 lines) ===')
src = inspect.getsource(model.forward)
print('\n'.join(src.split('\n')[:30]))

# 2. 看 dataloader 出的 data 结构
data = next(iter(loader))
print('\n=== data keys ===')
print(list(data.keys()))
print('\n=== data structure ===')
for k, v in data.items():
    if hasattr(v, 'data'):
        inner = v.data
        if isinstance(inner, list):
            print(f'  {k}: DataContainer(list of {len(inner)}, first elem type = {type(inner[0])})')
        else:
            print(f'  {k}: DataContainer({type(inner).__name__}, '
                  f'shape={inner.shape if hasattr(inner, "shape") else "N/A"})')
    else:
        print(f'  {k}: {type(v).__name__}')

# 3. 看一下 simple_test 或 forward_test
for fn_name in ['simple_test', 'forward_test', 'forward_dummy']:
    fn = getattr(model, fn_name, None)
    if fn is not None:
        print(f'\n=== model.{fn_name} signature ===')
        print(inspect.signature(fn))