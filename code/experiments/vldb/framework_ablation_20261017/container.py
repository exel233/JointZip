"""Complete source-free stream; identical metadata cost for both policies."""
import json
import struct
from . import codec

HEAD=struct.Struct('<4sI')
def frame(payload,entry,split,shape):
    count=int(shape[0]*shape[1]*shape[2])
    if entry['task']=='anomaly': count=int(entry['true_length'])
    meta=dict(version=1,dtype='<f4',shape=list(shape),true_count=count,
              mean=float(entry['mean']),std=float(entry['std']),channel=entry['channel'])
    header=json.dumps(meta,separators=(',',':'),sort_keys=True,allow_nan=False).encode()
    return HEAD.pack(b'JZF1',len(header))+header+payload,meta
def unframe(data):
    magic,size=HEAD.unpack_from(data)
    if magic!=b'JZF1' or size>65536 or HEAD.size+size>=len(data): raise ValueError('Bad JZF1 header')
    meta=json.loads(data[HEAD.size:HEAD.size+size]); payload=data[HEAD.size+size:]
    if meta['version']!=1 or meta['dtype']!='<f4': raise ValueError('Unsupported JZF1 stream')
    return meta,payload
def decode(data,workdir=None):
    meta,payload=unframe(data); y=codec.decode(payload,workdir)
    if list(y.shape)!=meta['shape'] or not 0<meta['true_count']<=y.size: raise ValueError('Bad decoded JZF1 dimensions')
    return y,meta
