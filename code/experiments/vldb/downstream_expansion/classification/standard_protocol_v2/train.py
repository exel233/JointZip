"""Full-budget, training-loss-selected ResNet follow-up; retain v1 unchanged.

Architecture is the explicitly disclosed PyTorch port in ../models.py. The
1500-epoch training configuration and per-query UCR2018 normalization follow
the pinned hfawaz/dl-4-tsc reference; TEST monitoring is deliberately omitted.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from ..common import ROOT, OUT as INPUT_OUT, DATASETS, SEEDS, METHODS, load, sha, write_json
from ..models import ResNetTSC
from ..evaluate import summary, extra_arrays, available

OUT=ROOT/'experiments/results/vldb_ucr_classification_standard_protocol_v2_20260915'
LOCK='/tmp/nclab_downstream_gpu_20260915.lock'
CONFIG={
    'schema':'nclab.ucr.resnet.standard-protocol-followup.v2',
    'status':'follow-up specified after observing weak raw v1 results; not initially preregistered',
    'retention':'report all v1 and v2 datasets/seeds; no replacement or best-of-protocol selection',
    'datasets':DATASETS,'seeds':SEEDS,'methods':METHODS,
    'scope':'six UCR classification datasets, official complete TRAIN and TEST; not the EDBT source matrix',
    'train':'all original official TRAIN; no internal split, refit, augmentation, class weighting or pretraining',
    'test':'never used in optimization, learning-rate scheduling or checkpoint selection; evaluation only after checkpoint persisted',
    'normalization':'per query subtract its own mean and divide by its own population std; zero std replaced by 1; float64 arithmetic then float32 network input',
    'normalization_application':'same deterministic rule independently on raw TRAIN and each raw/reconstructed TEST query; codec protection metrics remain in inherited TRAIN-standardized coordinates',
    'architecture':'explicit PyTorch port of pinned hfawaz 1D Wang2017 ResNet, 64/128/128 channels, 8/5/3 kernels, GAP',
    'runtime_difference':'not execution of the official TensorFlow/Keras library; framework initialization, BN variance arithmetic, optimizer arithmetic and shuffling differ',
    'epochs':1500,'batch_size':'int(min(n_TRAIN/10,64))',
    'optimizer':{'name':'Adam','lr':.001,'betas':[.9,.999],'eps':1e-7,'weight_decay':0.},
    'scheduler':{'monitor':'sample-weighted training cross entropy accumulated during each epoch',
                 'factor':.5,'patience':50,'min_delta':.0001,'min_lr':.0001,'cooldown':0,
                 'rule':'absolute improvement loss < best - min_delta; reduce after wait >= patience'},
    'checkpoint':{'monitor':'same training cross entropy','mode':'min','strict_improvement':True,
                  'save':'end-of-epoch model including BN buffers; no retraining of selected epoch'},
    'early_stopping':False,'gpu_lock':LOCK,
    'reference':{'paper':'https://doi.org/10.1109/IJCNN.2017.7966039',
       'author_original':'https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline/tree/3ecc8971fa768bd01ada7f3a613688255e6256c2',
       'resnet':'https://github.com/hfawaz/dl-4-tsc/blob/e0233efd886df8c6ca18e6f1b545d15aaf423627/classifiers/resnet.py',
       'normalization':'https://github.com/hfawaz/dl-4-tsc/blob/e0233efd886df8c6ca18e6f1b545d15aaf423627/utils/utils.py',
       'callback_defaults':'https://keras.io/api/callbacks/reduce_lr_on_plateau/'},
    'reference_relation':'hfawaz reference uses 1500 epochs, min(N/10,64), Adam, train-loss ReduceLROnPlateau and train-loss best checkpoint; UCR2018 loader normalizes per query. Original Wang author script instead caps batch size at 16; this follow-up follows the named 1D hfawaz reference consistently.',
    'no_current_sota_claim':True,
}

def identity():
    folder=Path(__file__).resolve().parent
    return {str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in
            [folder/'train.py',folder.parent/'models.py',folder.parent/'common.py',folder.parent/'evaluate.py']}

def normalize(x):
    x=np.asarray(x,dtype=np.float64)
    std=x.std(axis=1,keepdims=True);std[std==0]=1.
    return np.ascontiguousarray((x-x.mean(axis=1,keepdims=True))/std,dtype=np.float32)

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def run_epoch(model,x,y,opt,generator,batch):
    model.train();total=0.
    for idx in torch.randperm(len(x),generator=generator).split(batch):
        idx=idx.cuda();opt.zero_grad(set_to_none=True)
        loss=nn.functional.cross_entropy(model(x[idx]),y[idx])
        if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
        loss.backward();opt.step();total+=float(loss.detach())*len(idx)
    return total/len(x)

@torch.no_grad()
def predict(model,x):
    model.eval();return np.concatenate([model(part.cuda()).softmax(-1).cpu().numpy()
        for part in torch.from_numpy(normalize(x)[:,None,:]).split(64)])

def evaluate_saved(name,seed):
    data,entry,freeze=load(name);data=extra_arrays(data,name)
    folder=OUT/'resnet'/name/str(seed)
    ck=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
    assert ck['code_identity']==identity()
    model=ResNetTSC(ck['classes']).cuda();model.load_state_dict(ck['state_dict'])
    prob={m:predict(model,data[m]) for m in available(data)}
    y=data['ytest'];pred={m:p.argmax(1) for m,p in prob.items()}
    rows={m:summary(y,p,prob[m],pred['raw'],ck['classes']) for m,p in pred.items()}
    np.savez(folder/'predictions.npz',y=y,**{m+'_prob':p for m,p in prob.items()})
    write_json(folder/'evaluation.json',{'dataset':name,'seed':seed,'rows':rows,
        'checkpoint_sha256':sha(folder/'model.pt'),'same_checkpoint_for_every_method':True,
        'preprocessing':CONFIG['normalization'],'evaluation_code':identity()})
    del model;torch.cuda.empty_cache();return rows

def train_one(name,seed):
    start=time.perf_counter();data,entry,freeze=load(name)
    folder=OUT/'resnet'/name/str(seed);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'model.pt').exists():raise RuntimeError('refuse to retrain a frozen model')
    seed_all(seed);classes=len(data['class_labels']);batch=int(min(len(data['train'])/10,64));assert batch>=1
    x=torch.from_numpy(normalize(data['train'])[:,None,:]).cuda();y=torch.from_numpy(data['ytrain']).cuda()
    model=ResNetTSC(classes).cuda();params=sum(p.numel() for p in model.parameters())
    opt=torch.optim.Adam(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-7)
    generator=torch.Generator().manual_seed(seed)
    best=float('inf');best_epoch=0;best_state=None;scheduler_best=float('inf');wait=0;history=[]
    for epoch in range(1,CONFIG['epochs']+1):
        lr=float(opt.param_groups[0]['lr']);loss=run_epoch(model,x,y,opt,generator,batch)
        if loss<best:
            best=loss;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if loss<scheduler_best-.0001:scheduler_best=loss;wait=0
        else:
            wait+=1
            if wait>=50 and lr>.0001:
                for group in opt.param_groups:group['lr']=max(lr*.5,.0001)
                wait=0
        history.append({'epoch':epoch,'train_loss':loss,'lr':lr})
        if epoch%100==0:
            write_json(folder/'progress.json',{'epoch':epoch,'epochs':CONFIG['epochs'],'best_epoch':best_epoch,
                'elapsed_seconds':time.perf_counter()-start,'history':history})
            print(json.dumps({'training':name,'seed':seed,'epoch':epoch,'best_epoch':best_epoch,'loss':loss,
                'elapsed_seconds':time.perf_counter()-start}),flush=True)
    checkpoint={'state_dict':best_state,'classes':classes,'seed':seed,'selected_epochs':best_epoch,
                'normalization':'per_query_population_znorm_float64_to_float32','code_identity':identity()}
    torch.save(checkpoint,folder/'model.pt')
    write_json(folder/'training.json',{'dataset':name,'seed':seed,'parameters':params,
        'epochs':CONFIG['epochs'],'batch_size':batch,'selected_epochs':best_epoch,'best_training_loss':best,
        'training_seconds':time.perf_counter()-start,'training_cases':len(x),'history':history,
        'TEST_used_for_training_or_selection':False,'code_identity':identity(),
        'input_freeze_sha256':sha(INPUT_OUT/'freeze.json'),
        'runtime':{'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)}})
    del model,opt,x,y,best_state;torch.cuda.empty_cache()
    rows=evaluate_saved(name,seed)
    print(json.dumps({'complete':name,'seed':seed,'selected_epochs':best_epoch,
        'seconds':time.perf_counter()-start,'raw_accuracy':rows['raw']['accuracy'],'methods':list(rows)}),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=DATASETS);p.add_argument('--seed',type=int,choices=SEEDS)
    p.add_argument('--evaluate-only',action='store_true');p.add_argument('--freeze-only',action='store_true')
    p.add_argument('--after-v1',action='store_true');args=p.parse_args()
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    protocol={'config':CONFIG,'source_identity':identity(),'input_freeze_sha256':sha(INPUT_OUT/'freeze.json'),
        'compression_protocol_sha256':sha(INPUT_OUT/'compression_protocol.json'),
        'v1_model_protocol_sha256':sha(INPUT_OUT/'model_protocol.json'),
        'reference_sha256':{f:sha(ROOT/'tmp/downstream_expansion_20260915'/f)
                            for f in ['hfawaz_resnet.py','hfawaz_utils.py']}}
    write_json(OUT/'protocol.json',protocol,exclusive=True)
    if args.freeze_only:return
    if args.after_v1:
        while not all((INPUT_OUT/'resnet'/name/str(seed)/'evaluation.json').exists()
                      for name in DATASETS for seed in SEEDS):
            time.sleep(10)
    for seed in ([args.seed] if args.seed else SEEDS):
        for name in ([args.dataset] if args.dataset else DATASETS):
            folder=OUT/'resnet'/name/str(seed)
            if not args.evaluate_only and (folder/'evaluation.json').exists():continue
            print(json.dumps({'waiting_gpu':name,'seed':seed,'protocol':'standard_v2'}),flush=True)
            with open(LOCK,'a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                if args.evaluate_only:evaluate_saved(name,seed)
                else:train_one(name,seed)
                fcntl.flock(lock,fcntl.LOCK_UN)
            time.sleep(1)

if __name__=='__main__':main()
