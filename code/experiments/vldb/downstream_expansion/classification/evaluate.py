"""Shared classification metrics and CPU baselines; saved per-query outputs."""
import argparse
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
from .common import OUT, DATASETS, METHODS, load, write_json, sha

def normalized(x,mean,std):return np.ascontiguousarray((x-mean)/std,dtype=np.float32)

def summary(y,pred,prob,raw_pred,classes):
    rc=raw_pred==y; correct=pred==y
    result={'n':len(y),'correct':int(correct.sum()),'accuracy':float(accuracy_score(y,pred)),
            'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'macro_f1':float(f1_score(y,pred,labels=np.arange(classes),average='macro',zero_division=0)),
            'raw_prediction_agreement':float(np.mean(pred==raw_pred)),
            'raw_correct':int(rc.sum()),'raw_correct_to_wrong':int(np.sum(rc&~correct)),
            'raw_wrong_to_correct':int(np.sum(~rc&correct)),
            'per_class_accuracy':{str(k):float(np.mean(correct[y==k])) for k in range(classes)}}
    result['cross_entropy']=float(log_loss(y,prob,labels=np.arange(classes))) if prob is not None else None
    return result

def available(data):return [m for m in METHODS if m in data]

def extra_arrays(data,name):
    value={k:data[k] for k in data.files} if hasattr(data,'files') else dict(data)
    for method in ('pearson','acf'):
        entry=OUT/'compression'/name/f'{method}.json'
        if entry.exists():
            row=json.loads(entry.read_text());path=OUT/row['reconstruction_file']
            assert sha(path)==row['reconstruction_sha256']
            value[method]=np.load(path)
    return value

def unit(x):
    z=x.astype(np.float64)-x.mean(axis=1,keepdims=True,dtype=np.float64)
    norms=np.linalg.norm(z,axis=1,keepdims=True)
    assert np.all(norms>1e-15)
    return z/norms

def baselines():
    for name in DATASETS:
        data,entry,freeze=load(name);data=extra_arrays(data,name)
        train=data['train'];yt=data['ytrain'];y=data['ytest'];classes=len(data['class_labels'])
        fit=data['fit_indices'];val=data['val_indices']
        archive=unit(train)
        pred={m:yt[np.argmin(np.maximum(0.,2.-2.*(unit(data[m])@archive.T)),axis=1)] for m in available(data)}
        nnrows={m:summary(y,p,None,pred['raw'],classes) for m,p in pred.items()}
        for method,historical in entry['artifacts'].items():
            assert nnrows[method]['accuracy']==historical['historical_1nn_accuracy']
        dest=OUT/'baselines'/name;dest.mkdir(parents=True,exist_ok=True)
        np.savez(dest/'one_nn_predictions.npz',y=y,**{m+'_pred':p for m,p in pred.items()})
        write_json(dest/'one_nn.json',{'rows':nnrows,'model':'shape 1NN on complete raw TRAIN','test_selection':False})
        mean=float(train[fit].mean(dtype=np.float64));std=float(train[fit].std(dtype=np.float64))
        candidate=[]
        for c in (.01,.1,1.,10.,100.):
            model=LogisticRegression(C=c,max_iter=3000,solver='lbfgs',random_state=20260915)
            model.fit(normalized(train[fit],mean,std),yt[fit])
            loss=float(log_loss(yt[val],model.predict_proba(normalized(train[val],mean,std)),labels=np.arange(classes)))
            candidate.append({'C':c,'val_log_loss':loss,'iterations':model.n_iter_.tolist()})
        selected=min(candidate,key=lambda r:(r['val_log_loss'],r['C']))['C']
        mean=float(train.mean(dtype=np.float64));std=float(train.std(dtype=np.float64))
        model=LogisticRegression(C=selected,max_iter=3000,solver='lbfgs',random_state=20260915)
        model.fit(normalized(train,mean,std),yt)
        probabilities={m:model.predict_proba(normalized(data[m],mean,std)) for m in available(data)}
        predictions={m:p.argmax(1) for m,p in probabilities.items()}
        rows={m:summary(y,p,probabilities[m],predictions['raw'],classes) for m,p in predictions.items()}
        np.savez(dest/'linear_model.npz',coef=model.coef_,intercept=model.intercept_,classes=model.classes_,mean=mean,std=std,C=selected)
        np.savez(dest/'linear_predictions.npz',y=y,**{m+'_prob':p for m,p in probabilities.items()})
        write_json(dest/'linear.json',{'rows':rows,'C':selected,'selection':candidate,'refit_on_full_TRAIN':True,'test_selection':False})
        print(json.dumps({'baseline_complete':name,'one_nn_raw':nnrows['raw']['accuracy'],'linear_raw':rows['raw']['accuracy'],'methods':list(rows)}),flush=True)

if __name__=='__main__':baselines()
