"""Separate UCR classification expansion; never modifies historical evidence."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'experiments/results/vldb_ucr_classification_expansion_20260915_v1'
OLD = ROOT / 'experiments/vldb/external_utility/results/ucr_quick_panel_v5'
METHODS = ['raw', 'point_only', 'pearson', 'acf', 'joint', 'sperr', 'sz2016']
DATASETS = ['ECG200', 'ArrowHead', 'GunPoint', 'Earthquakes', 'ShapeletSim', 'Meat']
SEEDS = [20260915, 20260916, 20260917]
PROTOCOL = {
    'schema': 'nclab.ucr.classification-expansion.v1',
    'scope': 'six-dataset UCR classification panel, not EDBT six-source full matrix',
    'reuse': '24 historical v5 SCS2 artifacts plus separately frozen Point+Pearson and Point+ACF supplements',
    'methods': METHODS, 'datasets': DATASETS, 'seeds': SEEDS,
    'supplemental_ablations': ['point_plus_pearson', 'point_plus_acf'],
    'split': {'seed': 20260915, 'within_official_TRAIN': True,
              'validation_per_class': 'max(1, ceil(0.2*n_class)); remaining FIT',
              'labels_from': 'official TRAIN only'},
    'one_nn': 'centered unit-L2 shape distance, all original TRAIN, stable first tie',
    'linear': {'model': 'multinomial/binary logistic regression', 'C': [.01,.1,1.,10.,100.],
               'selection': 'minimum raw TRAIN-VAL log loss; smaller C on ties',
               'max_iter': 3000, 'refit': 'all uncompressed official TRAIN'},
    'resnet': {'architecture': 'Wang2017 ResNet-TSC via hfawaz reference; explicit PyTorch port',
               'channels': [64,128,128], 'kernels_per_block': [8,5,3],
               'shortcut': '1x1+BN when channel count changes; BN otherwise',
               'pooling': 'global average', 'init': 'Glorot uniform; zero bias',
               'batch_norm': {'eps': .001, 'pytorch_momentum': .01},
               'optimizer': 'Adam', 'learning_rate': .001, 'adam_eps': 1e-7,
               'max_epochs': 500, 'early_stop_patience': 50, 'min_epochs': 50,
               'min_delta': 1e-5, 'batch_size': 16,
               'reduce_lr': {'factor': .5, 'patience': 20, 'min_lr': 1e-5},
               'selection': 'minimum raw TRAIN-VAL cross entropy',
               'refit': 'fresh initialization on all raw TRAIN for selected epochs; replay selection LR schedule',
               'augmentation': None, 'class_weights': None, 'pretraining': None,
               'test_selection': False},
    'normalization': 'inherited TRAIN-only codec coordinates; classifier FIT mean/std for selection, full TRAIN mean/std for final refit',
    'endpoints': ['accuracy','balanced_accuracy','macro_f1','cross_entropy',
                  'raw_prediction_agreement','raw_correct_to_wrong','raw_wrong_to_correct'],
    'interpretation': 'same epsilon stress panel, different rates; no ACF-only causal claim or current-SOTA claim',
    'gpu_lock': '/tmp/nclab_downstream_gpu_20260915.lock',
    'reference': {'paper': 'https://doi.org/10.1109/IJCNN.2017.7966039',
                  'author_code': 'https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline/tree/3ecc8971fa768bd01ada7f3a613688255e6256c2',
                  'reference_code': 'https://github.com/hfawaz/dl-4-tsc/blob/e0233efd886df8c6ca18e6f1b545d15aaf423627/classifiers/resnet.py'},
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def tensor_sha(x):
    x = np.ascontiguousarray(x)
    h = hashlib.sha256(x.dtype.str.encode('ascii'))
    h.update(np.asarray(x.shape,dtype='<u8').tobytes()); h.update(x.tobytes())
    return h.hexdigest()

def write_json(path, obj, exclusive=False):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    data=(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
    if exclusive and path.exists():
        if path.read_bytes()!=data: raise RuntimeError(f'frozen evidence differs: {path}')
        return
    temporary=path.with_suffix(path.suffix+'.tmp'); temporary.write_bytes(data); temporary.replace(path)

def code_identity():
    return {p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}

def load(name):
    freeze=json.loads((OUT/'freeze.json').read_text())
    entry=freeze['datasets'][name]; path=OUT/entry['prepared_file'].replace('\\','/')
    if sha(path)!=entry['prepared_sha256']: raise RuntimeError('prepared input changed')
    return np.load(path), entry, freeze
