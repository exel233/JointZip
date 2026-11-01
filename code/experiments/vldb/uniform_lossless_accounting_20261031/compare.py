"""Reframe existing streams and compare actual complete file sizes.

No compression search, correction, scientific encoding, or downstream
inference is invoked. All previous experiments and manuscript files are read-only.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
import statistics
import struct
import sys
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tmp/runtime_uniform_accounting_20261031/deps'))
import numpy as np
from common_stream import frame,parse,decode_lossless,as_original_jointzip,decompress_exact

OUT=ROOT/'experiments/results/vldb_uniform_lossless_accounting_20261031_v1'
OLD=ROOT/'experiments/results/vldb_simple_correction_20260929_v1'
RUNS=[ROOT/'experiments/results/vldb_framework_full_20261015_v1',
      ROOT/'experiments/results/vldb_framework_sz2016_20261016_v1']
BACKENDS=('sz3','zfp','sperr','serf_xor','serf_qt','sz2016')
TASKS=('forecast','ucr','anomaly')

def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(data):return hashlib.sha256(data).hexdigest()
def rows(path):return list(csv.DictReader(path.open(encoding='utf-8-sig',newline='')))
def save(name,data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
def savecsv(name,data):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def write_artifact(relative,data):
    path=OUT/relative;path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():assert path.read_bytes()==data
    else:path.write_bytes(data)
    assert path.read_bytes()==data
    return path.stat().st_size
def geo(values):return math.exp(statistics.fmean(math.log(v) for v in values))
def macro(data,field):
    groups=defaultdict(list)
    for r in data:groups[r['dataset']].append(float(r[field]))
    return geo([geo(v) for v in groups.values()])

def historical_parts(data):
    # Keep all codec records. Replace only the historical common envelopes.
    tsl_true=None
    if data[:4]==b'TSL1':
        _,block,tsl_true=struct.unpack_from('<4sIQ',data)
        assert block==128;data=data[16:]
    magic,version,ncodec,nnorm,nprov,ncollection=struct.unpack_from('<4sIQQQQ',data)
    assert magic==b'SCS2' and version==2
    assert 40+ncodec+nnorm+nprov+ncollection==len(data)
    pos=40
    codec=data[pos:pos+ncodec].decode();pos+=ncodec
    norm=data[pos:pos+nnorm];pos+=nnorm
    provenance=data[pos:pos+nprov];pos+=nprov
    collection=data[pos:]
    assert codec=='zstd_lossless'
    nmagic,nch=struct.unpack_from('<4sI',norm)
    assert nmagic==b'NOR2' and nch==1
    name_len,=struct.unpack_from('<H',norm,8)
    channel=norm[10:10+name_len].decode()
    assert len(norm)==10+name_len+8
    mean,std=struct.unpack_from('<ff',norm,10+name_len)
    return dict(collection=collection,channel=channel,mean=mean,std=std,
                true_count=tsl_true,provenance_sha256=sha(provenance),
                provenance_bytes=nprov,normalization_bytes=nnorm)

def verify_jointzip_inner(payload,meta,saved):
    inner=decompress_exact(payload[4:])
    head=struct.Struct('<4sBBIIIdddI');record=struct.Struct('<BIdII')
    magic,version,codec,w,c,t,eps,tau,rho,lag=head.unpack_from(inner)
    assert magic==b'JZI1' and version==1 and codec in range(1,7)
    assert [w,c,t]==meta['shape'] and lag==16
    assert eps==saved['point']['epsilon'] and rho==saved['point']['rho'] and tau==saved['point']['tau']
    offset=head.size
    for _ in range(w*c):
        kind,count,cap,nb,nc=record.unpack_from(inner,offset);offset+=record.size
        assert kind in (0,1) and nb>0 and nc>=0 and offset+nb+nc<=len(inner)
        if kind==0:
            assert count==0 and nc==0
            decompress_exact(inner[offset:offset+nb],t*4)
        offset+=nb+nc
    assert offset==len(inner)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(schema='jointzip.uniform-complete-accounting.v1',
        created_utc=datetime.now(timezone.utc).isoformat(),
        scope='All 172 original TEST inputs; all 3024 ERA JointZip TEST conditions; Zstd lossless reference.',
        current_runs=[str(p.relative_to(ROOT)) for p in RUNS],
        baseline_run=str(OLD.relative_to(ROOT)),
        encoding='No encoding. Copy complete saved JointZip JZP1 and lossless NCS collections byte-for-byte.',
        common_format='JCF1: 4-byte magic + uint32 JSON length + identical per-input JZF1 JSON + self-identifying codec payload.',
        common_metadata=['version','dtype','shape','true_count','mean','std','channel'],
        jointzip_payload='Unchanged JZP1, including its 4-byte dispatch magic, all records, bases, corrections and whole-stream compression.',
        lossless_payload='4-byte ZLC1 dispatch magic + unchanged complete NCS1/NCS2 collection, including closure mode and lengths, ZSL1 shape and window boundaries.',
        lossless_codec='Saved windowed Zstd level19 with batch closure selected among raw/levels3,9,15,19. Not the internal Zstd3 fallback.',
        provenance='External manifest for both methods, excluded from both stored files. All required decoding metadata counted.',
        channel_policy='Use current input schema channel from JZF1 on both sides. Record historical channel names externally.',
        numerator='4*true_count, unpadded Float32 bytes.',
        aggregation='First geometric mean within each source, then equal sources. Forecast and NASA settings equally represented; UCR stress only.',
        verification='All lossless files independently decoded bit-exact. All JointZip inner payloads and full original-file reconstruction hashes verified unchanged, with existing independent-decode evidence reused.',
        no_changes=['No scientific encoders or downstream tasks rerun','No manuscript or Overleaf changes'])
    proto=OUT/'PROTOCOL.json'
    if not proto.exists():save('PROTOCOL.json',protocol)
    current=[]
    for run in RUNS:
        for r in rows(run/'analysis/compression.csv'):
            if r['split']=='test' and r['arm']=='jointzip':r['_run']=run;current.append(r)
    assert len(current)==3024
    frozen_entries={(e['task'],e['sid']):e for e in read(RUNS[0]/'freeze.json')['entries']}
    groups=defaultdict(list)
    for r in current:groups[r['task'],r['sid']].append(r)
    assert len(groups)==172
    baselines={};audit=[];pairs=[];channels=[]
    for key,conditions in sorted(groups.items()):
        task,sid=key
        sample=conditions[0];s=read(sample['_run']/sample['source_row'])
        original=(sample['_run']/s['artifact']).read_bytes()
        assert sha(original)==s['artifact_sha256']
        magic,n=struct.unpack_from('<4sI',original);assert magic==b'JZF1'
        header=original[8:8+n];meta=json.loads(header)
        assert meta==s['container_metadata']
        old=read(OLD/'compression'/task/sid/'test/zstd__lossless.json')
        old_bytes=(OLD/old['artifact']).read_bytes()
        assert old['status']=='complete' and sha(old_bytes)==old['artifact_sha256']
        assert old['input_sha256']==s['input_sha256'] and old['shape']==meta['shape']
        assert old['raw_bytes']==4*meta['true_count']==s['raw_bytes']
        parts=historical_parts(old_bytes)
        assert parts['mean']==meta['mean'] and parts['std']==meta['std']
        if parts['true_count'] is not None:assert parts['true_count']==meta['true_count']
        if parts['channel']!=meta['channel']:
            channels.append(dict(task=task,sid=sid,old_channel=parts['channel'],common_channel=meta['channel']))
        packed=frame(header,b'ZLC1'+parts['collection'])
        relative=f'artifacts/lossless/{task}__{sid}.jcf'
        size=write_artifact(relative,packed)
        rebuilt,decoded_meta=decode_lossless((OUT/relative).read_bytes())
        frozen=frozen_entries[key]
        input_path=RUNS[0]/frozen['input']
        assert sha(input_path.read_bytes())==old['input_sha256']==frozen['input_sha256']
        with np.load(input_path,allow_pickle=False) as arrays:
            source=np.ascontiguousarray(arrays['source'],dtype='<f4')
        assert decoded_meta==meta and rebuilt.shape==source.shape and rebuilt.tobytes()==source.tobytes()
        old_recon=np.load(OLD/old['reconstruction'])
        assert rebuilt.shape==old_recon.shape and rebuilt.tobytes()==old_recon.astype('<f4',copy=False).tobytes()
        entry=dict(task=task,dataset=s['dataset'],sid=sid,shape=meta['shape'],raw_bytes=s['raw_bytes'],
            artifact=relative,complete_bytes=size,cr=s['raw_bytes']/size,
            common_metadata_bytes=8+len(header),codec_dispatch_bytes=4,
            complete_collection_bytes=len(parts['collection']),
            original_complete_bytes=len(old_bytes),historical_provenance_bytes=parts['provenance_bytes'],
            input_sha256=old['input_sha256'],artifact_sha256=sha(packed),
            source_array_sha256=sha(source.tobytes()),
            retained_collection_sha256=sha(parts['collection']),lossless_bitwise_pass=True)
        baselines[key]=entry
        audit.append(dict(task=task,sid=sid,old_row=str((OLD/'compression'/task/sid/'test/zstd__lossless.json').relative_to(ROOT)),
            old_artifact=old['artifact'],old_artifact_sha256=old['artifact_sha256'],
            old_channel=parts['channel'],common_channel=meta['channel'],
            historical_provenance_sha256=parts['provenance_sha256'],common_metadata_sha256=sha(header)))
        for r in conditions:
            saved=read(r['_run']/r['source_row'])
            data=(r['_run']/saved['artifact']).read_bytes()
            assert saved['status']=='complete' and saved['independent_decode_bitwise'] is True
            assert saved['enabled_pass_numerical'] is True and sha(data)==saved['artifact_sha256']
            mg,nh=struct.unpack_from('<4sI',data)
            assert mg==b'JZF1' and data[8:8+nh]==header
            assert saved['input_sha256']==entry['input_sha256'] and saved['raw_bytes']==entry['raw_bytes']
            payload=data[8+nh:];assert payload.startswith(b'JZP1')
            verify_jointzip_inner(payload,meta,saved)
            new=frame(header,payload)
            relative_j=f"artifacts/jointzip/{task}__{sid}__{r['backend']}__{r['point']}.jcf"
            new_size=write_artifact(relative_j,new)
            assert as_original_jointzip((OUT/relative_j).read_bytes())==data
            assert new_size==saved['artifact_bytes']==int(r['artifact_bytes'])
            ratio=size/new_size
            pairs.append(dict(task=task,dataset=r['dataset'],sid=sid,backend=r['backend'],point=r['point'],
                raw_bytes=saved['raw_bytes'],jointzip_complete_bytes=new_size,lossless_complete_bytes=size,
                jointzip_cr=saved['raw_bytes']/new_size,lossless_cr=saved['raw_bytes']/size,
                cr_ratio=ratio,byte_savings_percent=100*(1-new_size/size),
                outcome='higher' if size>new_size else 'lower' if size<new_size else 'tied',
                common_metadata_bytes=8+len(header),jointzip_payload_bytes=len(payload),
                jointzip_artifact=relative_j,lossless_artifact=entry['artifact'],
                jointzip_artifact_sha256=sha(new),original_jointzip_sha256=sha(data),
                jointzip_payload_sha256=sha(payload),jointzip_original_decode_verified=True))
    assert len(pairs)==3024 and len(baselines)==172
    macros=[];settings=[];sources=[]
    for backend in BACKENDS:
        for task in TASKS:
            group=[p for p in pairs if p['backend']==backend and p['task']==task]
            c=Counter(p['outcome'] for p in group)
            j=macro(group,'jointzip_cr');l=macro(group,'lossless_cr')
            macros.append(dict(backend=backend,task=task,conditions=len(group),sources=len({r['dataset'] for r in group}),
                jointzip_cr=j,lossless_cr=l,cr_ratio=j/l,geomean_byte_savings_percent=100*(1-l/j),
                higher=c['higher'],tied=c['tied'],lower=c['lower']))
            for point in sorted({r['point'] for r in group}):
                g=[r for r in group if r['point']==point];cc=Counter(r['outcome'] for r in g)
                jj=macro(g,'jointzip_cr');ll=macro(g,'lossless_cr')
                settings.append(dict(backend=backend,task=task,point=point,conditions=len(g),jointzip_cr=jj,lossless_cr=ll,
                    cr_ratio=jj/ll,higher=cc['higher'],tied=cc['tied'],lower=cc['lower']))
            for source in sorted({r['dataset'] for r in group}):
                g=[r for r in group if r['dataset']==source];cc=Counter(r['outcome'] for r in g)
                jj=geo(r['jointzip_cr'] for r in g);ll=geo(r['lossless_cr'] for r in g)
                sources.append(dict(backend=backend,task=task,dataset=source,conditions=len(g),jointzip_cr=jj,lossless_cr=ll,
                    cr_ratio=jj/ll,higher=cc['higher'],tied=cc['tied'],lower=cc['lower']))
    savecsv('conditions.csv',pairs);savecsv('macros.csv',macros);savecsv('settings.csv',settings);savecsv('sources.csv',sources)
    save('lossless_files.json',list(baselines.values()));save('source_bindings.json',audit);save('channel_mapping.json',channels)
    counts=Counter(r['outcome'] for r in pairs)
    summary=dict(complete=True,conditions=len(pairs),unique_inputs=len(baselines),
        generated_complete_files=len(pairs)+len(baselines),counts=dict(counts),
        macro_groups=len(macros),macro_higher=sum(r['cr_ratio']>1 for r in macros),
        macro_cr_ratio_range=[min(r['cr_ratio'] for r in macros),max(r['cr_ratio'] for r in macros)],
        macro_geomean_byte_savings_range=[min(r['geomean_byte_savings_percent'] for r in macros),max(r['geomean_byte_savings_percent'] for r in macros)],
        source_groups=len(sources),source_macro_higher=sum(r['cr_ratio']>1 for r in sources),
        channel_name_mappings=len(channels),lossless_bitwise_verified=len(baselines),
        jointzip_preserved_original_file_bitwise=len(pairs),jointzip_existing_independent_decode_evidence_verified=len(pairs),
        jointzip_native_decoders_rerun=False,scientific_encoders_rerun=False,downstream_rerun=False,
        all_required_metadata_counted=True,provenance_external_for_both=True,
        only_wrapper_changed=True,errors=[],
        example=[r for r in pairs if r['sid']=='544_SMAP_id_14_Sensor_tr_1978_1st_4450' and r['backend']=='zfp' and r['point']=='strict'][0])
    save('SUMMARY.json',summary)
    save('COMPLETE.json',dict(summary,completed_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
