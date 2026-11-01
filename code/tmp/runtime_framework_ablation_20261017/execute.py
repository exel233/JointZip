"""Copy pinned runtime once, only after the prerequisite full run is delivered."""
from pathlib import Path
import hashlib
import json
import os
import shutil

ROOT=Path('/opt/jointzip')
NAME='vldb_framework_ablation_20261017_v1'
REAL=Path('/opt/jointzip-home/.local/cth_expansion_runs')/NAME
PARENT=REAL.parent/'vldb_framework_full_20261015_v1'

def read(path):return json.loads(path.read_text())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def setup():
    if read(PARENT/'execution/COMPLETE.json').get('complete') is not True:
        raise RuntimeError('Parent pipeline incomplete')
    if read(PARENT/'audit/closeout/CLOSEOUT.json').get('passed') is not True:
        raise RuntimeError('Parent exact closeout not passed')
    if read(PARENT/'timing/COMPLETE.json').get('total')!=480:
        raise RuntimeError('Parent independent timing not complete')
    target=REAL/'runtime';target.mkdir(parents=True,exist_ok=True)
    receipt=target/'ENVIRONMENT.json'
    if not receipt.exists():
        source=PARENT/'runtime'
        shutil.copytree(source/'lib',target/'lib',dirs_exist_ok=True)
        pairs=[(source/'qpet_correlation_batch',target/'qpet_correlation_batch'),
               (source/'anchors',target/'anchors'),
               (ROOT/'tools/sz2016/bin/sz2016_batch',target/'sz2016_batch'),
               (ROOT/'tools/sz2016/sz2016.config',target/'sz2016.config')]
        evidence=[]
        for original,destination in pairs:
            shutil.copy2(original,destination)
            if digest(original)!=digest(destination):raise RuntimeError('Runtime copy differs')
            evidence.append(dict(source=str(original),copy=str(destination),sha256=digest(destination)))
        env=dict(JOINTZIP_PILOT_NATIVE_LIBS=str(target/'lib'),
            JOINTZIP_PILOT_REPAIR=str(target/'qpet_correlation_batch'),
            JOINTZIP_PILOT_ANCHORS=str(target/'anchors'),
            JOINTZIP_SZ2016_BINARY=str(target/'sz2016_batch'),
            JOINTZIP_SZ2016_CONFIG=str(target/'sz2016.config'))
        (target/'COPY_RECEIPT.json').write_text(json.dumps(evidence,indent=2))
        temporary=receipt.with_suffix('.tmp');temporary.write_text(json.dumps(env,indent=2));temporary.replace(receipt)
    os.environ.update(read(receipt))
    os.environ.update(MODERN_CODECS_PREFIX=str(PARENT/'runtime/modern-codecs'), JOINTZIP_FULL_BRIDGE=str(PARENT/'runtime/fixed_backend'), JOINTZIP_FULL_BRIDGE_MANIFEST=str(PARENT/'runtime/manifest.json'))
    os.environ['JOINTZIP_ABLATION_RUN_NAME']=NAME

if __name__=='__main__':
    setup()
    from experiments.vldb.framework_ablation_20261017.pipeline import main
    main()
