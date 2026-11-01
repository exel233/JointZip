"""TRAIN-VAL model selection, full TRAIN refit, then shared TEST evaluation."""
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
from .common import ROOT, OUT, DATASETS, SEEDS, PROTOCOL, load, sha, write_json
from .models import ResNetTSC
from .evaluate import normalized,summary,extra_arrays,available

def identity():
    folder=Path(__file__).parent
    return {p:sha(folder/p) for p in ['common.py','models.py','train.py','evaluate.py']}

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def tensors(x,y,mean,std):
    return torch.from_numpy(normalized(x,mean,std)[:,None,:]).cuda(),torch.from_numpy(y).cuda()

def run_epoch(model,x,y,optimizer,generator,batch):
    model.train();total=0.
    order=torch.randperm(len(x),generator=generator)
    for indices in order.split(batch):
        indices=indices.cuda();optimizer.zero_grad(set_to_none=True)
        loss=nn.functional.cross_entropy(model(x[indices]),y[indices])
        if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
        loss.backward();optimizer.step();total+=float(loss.detach())*len(indices)
    return total/len(x)

@torch.no_grad()
def val_loss(model,x,y):
    model.eval();return float(nn.functional.cross_entropy(model(x),y))

@torch.no_grad()
def predict(model,x,mean,std):
    model.eval();parts=[]
    for block in torch.from_numpy(normalized(x,mean,std)[:,None,:]).split(64):
        parts.append(model(block.cuda()).softmax(-1).cpu().numpy())
    return np.concatenate(parts)

def evaluate_saved(name,seed):
    data,entry,freeze=load(name);data=extra_arrays(data,name)
    folder=OUT/'resnet'/name/str(seed)
    checkpoint=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
    model=ResNetTSC(checkpoint['classes']).cuda();model.load_state_dict(checkpoint['state_dict'])
    probs={m:predict(model,data[m],checkpoint['mean'],checkpoint['std']) for m in available(data)}
    y=data['ytest'];pred={m:p.argmax(1) for m,p in probs.items()}
    rows={m:summary(y,p,probs[m],pred['raw'],checkpoint['classes']) for m,p in pred.items()}
    np.savez(folder/'predictions.npz',y=y,**{m+'_prob':p for m,p in probs.items()})
    write_json(folder/'evaluation.json',{'dataset':name,'seed':seed,'checkpoint_sha256':sha(folder/'model.pt'),
          'same_checkpoint_for_every_method':True,'rows':rows,'evaluation_code':identity()})
    del model;torch.cuda.empty_cache()
    return rows

def train_one(name,seed):
    started=time.perf_counter();data,entry,freeze=load(name)
    config=PROTOCOL['resnet'];train=data['train'];y=data['ytrain'];fit=data['fit_indices'];val=data['val_indices']
    folder=OUT/'resnet'/name/str(seed);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'model.pt').exists():raise RuntimeError('refuse to retrain an existing frozen model')
    classes=len(data['class_labels']);seed_all(seed)
    mean=float(train[fit].mean(dtype=np.float64));std=float(train[fit].std(dtype=np.float64))
    xfit,yfit=tensors(train[fit],y[fit],mean,std);xval,yval=tensors(train[val],y[val],mean,std)
    model=ResNetTSC(classes).cuda();params=sum(p.numel() for p in model.parameters())
    opt=torch.optim.Adam(model.parameters(),lr=.001,eps=1e-7)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=20,min_lr=1e-5)
    generator=torch.Generator().manual_seed(seed);best=float('inf');best_epoch=0;wait=0;history=[]
    for epoch in range(1,config['max_epochs']+1):
        lr=opt.param_groups[0]['lr'];loss=run_epoch(model,xfit,yfit,opt,generator,16);vl=val_loss(model,xval,yval)
        if not np.isfinite(vl):raise RuntimeError('nonfinite validation loss')
        history.append({'epoch':epoch,'train_loss':loss,'val_loss':vl,'lr':lr})
        if vl<best-config['min_delta']:best=vl;best_epoch=epoch;wait=0
        else:wait+=1
        scheduler.step(vl)
        if epoch%25==0:print(json.dumps({'phase':'selection','dataset':name,'seed':seed,'epoch':epoch,'best_epoch':best_epoch,'val_loss':vl}),flush=True)
        if epoch>=config['min_epochs'] and wait>=config['early_stop_patience']:break
    del model,opt,scheduler,xfit,yfit,xval,yval;torch.cuda.empty_cache()
    write_json(folder/'selection.json',{'history':history,'best_epoch':best_epoch,'best_val_loss':best,'selected_without_TEST':True})
    seed_all(seed);mean=float(train.mean(dtype=np.float64));std=float(train.std(dtype=np.float64))
    xfull,yfull=tensors(train,y,mean,std);model=ResNetTSC(classes).cuda()
    opt=torch.optim.Adam(model.parameters(),lr=.001,eps=1e-7);generator=torch.Generator().manual_seed(seed);refit=[]
    for epoch in range(1,best_epoch+1):
        for group in opt.param_groups:group['lr']=history[epoch-1]['lr']
        loss=run_epoch(model,xfull,yfull,opt,generator,16);refit.append(loss)
        if epoch%50==0:print(json.dumps({'phase':'refit','dataset':name,'seed':seed,'epoch':epoch,'epochs':best_epoch}),flush=True)
    torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'classes':classes,
                'mean':mean,'std':std,'seed':seed,'selected_epochs':best_epoch,'code_identity':identity()},folder/'model.pt')
    elapsed=time.perf_counter()-started
    write_json(folder/'training.json',{'dataset':name,'seed':seed,'parameters':params,'selection_epochs':len(history),
            'selected_epochs':best_epoch,'best_val_loss':best,'refit_loss':refit,'training_seconds':elapsed,
            'full_original_TRAIN_used_for_refit':len(train),'TEST_used_for_training_or_selection':False,
            'input_freeze_sha256':sha(OUT/'freeze.json'),'code_identity':identity(),
            'runtime':{'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)}})
    del model,opt,xfull,yfull;torch.cuda.empty_cache()
    rows=evaluate_saved(name,seed)
    print(json.dumps({'complete':name,'seed':seed,'seconds':elapsed,'selected_epochs':best_epoch,'raw_accuracy':rows['raw']['accuracy'],'methods':list(rows)}),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=DATASETS);parser.add_argument('--seed',type=int,choices=SEEDS);parser.add_argument('--evaluate-only',action='store_true');args=parser.parse_args()
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    model_protocol={'protocol':PROTOCOL,'source_identity':identity(),'input_freeze_sha256':sha(OUT/'freeze.json'),
                    'reference_sha256':sha(ROOT/'tmp/downstream_expansion_20260915/hfawaz_resnet.py')}
    write_json(OUT/'model_protocol.json',model_protocol,exclusive=True)
    for seed in ([args.seed] if args.seed else SEEDS):
        for name in ([args.dataset] if args.dataset else DATASETS):
            folder=OUT/'resnet'/name/str(seed)
            if not args.evaluate_only and (folder/'evaluation.json').exists():continue
            print(json.dumps({'waiting_gpu':name,'seed':seed}),flush=True)
            with open(PROTOCOL['gpu_lock'],'a+') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                print(json.dumps({'gpu_acquired':name,'seed':seed}),flush=True)
                if args.evaluate_only:evaluate_saved(name,seed)
                else:train_one(name,seed)
                fcntl.flock(lock,fcntl.LOCK_UN)
            time.sleep(1)

if __name__=='__main__':main()
