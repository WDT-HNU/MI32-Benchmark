import argparse
import hashlib
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from model_adapters.uni_ntfm import UniNTFMAdapter
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from torch.utils.data import DataLoader, Dataset, Sampler


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


REGIONS = {
    "frontal": ["Fp1","Fp2","F3","F4","F7","F8","Fz","F1","F2","F5","F6","AF3","AF4","AF7","AF8","AFz","FT7","FT8","FC1","FC2","FC3","FC4","FC5","FC6"],
    "central": ["C3","C4","Cz","C1","C2","C5","C6","CP1","CP2","CP3","CP4","CP5","CP6","CPz"],
    "parietal": ["P3","P4","P7","P8","Pz","P1","P2","P5","P6","PO3","PO4","PO7","PO8","POz","CP1","CP2","CP3","CP4","CP5","CP6","CPz","TP7","TP8","TP9"],
    "temporal": ["T3","T4","T5","T6","T7","T8","FT7","FT8","TP7","TP8","TP9","TP10"],
    "occipital": ["O1","O2","Oz","PO3","PO4","PO7","PO8","POz","CB1","CB2"],
}

UNINTFM_OFFICIAL_COMMIT = "c0ce0152f94366e59b31b3fb2c108ce909bcd95c"
UNINTFM_BENCHMARK_TRACK = "protocol_benchmark"


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class MI3Trials(Dataset):
    def __init__(self, root, subject_ids, max_trials_per_subject=0, preload=False, adapter=None):
        if adapter is None:
            raise ValueError("MI3Trials requires the project UniNTFMAdapter (fitted)")
        self.adapter = adapter
        self.mapping = np.asarray(adapter.mapping, dtype=np.int64)
        self.root = Path(root)
        self.subjects = pd.read_csv(self.root / "subjects.csv").set_index("subject_id")
        self.items = []
        self.by_subject = []
        self.cache = OrderedDict()
        self.preload = preload
        for sid in sorted(map(int, subject_ids)):
            n = int(self.subjects.loc[sid, "n_trials"])
            if max_trials_per_subject and n > max_trials_per_subject:
                # Preserve the corpus 1:1:2 label ratio in every tuning subject.
                X, labels, mask = self._read_subject(sid)
                q = max_trials_per_subject // 4
                quotas = [q, q, max_trials_per_subject - 2 * q]
                parts = []
                for label, quota in enumerate(quotas):
                    candidates = np.flatnonzero(labels == label)
                    parts.append(candidates[np.linspace(0, len(candidates) - 1, quota, dtype=int)])
                chosen = np.concatenate(parts)
            else:
                chosen = np.arange(n)
            group = []
            for trial in chosen:
                group.append(len(self.items)); self.items.append((sid, int(trial)))
            self.by_subject.append(group)
        if preload:
            for sid in sorted(map(int, subject_ids)):
                if sid not in self.cache:
                    self.cache[sid] = self._read_subject(sid)

    def __len__(self): return len(self.items)

    def _read_subject(self, sid):
        name = self.subjects.loc[sid, "subject_name"]
        with np.load(self.root / "data" / f"{name}.npz", allow_pickle=False) as z:
            return z["X"].copy(), z["y"].copy(), z["channel_mask"].astype(bool).copy()

    def _load(self, sid):
        if sid in self.cache:
            return self.cache[sid]
        value = self._read_subject(sid)
        if not self.preload:
            self.cache.clear()
        self.cache[sid] = value
        return value

    def __getitem__(self, index):
        sid, tid = self.items[index]
        X, y, mask = self._load(sid)
        out = self.adapter.transform(X[tid : tid + 1])[0]
        padding = self.adapter.padding_mask(1)[0]
        # Keep the frozen MI-32 signal contract. Provenance is recorded, but it
        # must not turn interpolated channels into padding.
        return torch.from_numpy(out), torch.from_numpy(padding), int(y[tid]), sid


class SubjectBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle):
        self.groups=dataset.by_subject; self.batch_size=batch_size; self.shuffle=shuffle
    def __iter__(self):
        groups=list(range(len(self.groups)))
        if self.shuffle: random.shuffle(groups)
        for g in groups:
            ids=list(self.groups[g])
            if self.shuffle: random.shuffle(ids)
            for i in range(0,len(ids),self.batch_size): yield ids[i:i+self.batch_size]
    def __len__(self): return sum(math.ceil(len(g)/self.batch_size) for g in self.groups)


class UniClassifier(nn.Module):
    def __init__(self, model_module, compact=False):
        super().__init__()
        cfg=model_module.ModelConfig(); cfg.sequence_length=750
        if compact:
            cfg.embed_dim=192; cfg.num_heads=6; cfg.depth=4; cfg.use_moe=False; cfg.num_experts=4
        self.cfg=cfg
        self.backbone=model_module.DualDomainTransformerMEM(cfg)
        self.head=nn.Sequential(nn.LayerNorm(cfg.embed_dim),nn.Linear(cfg.embed_dim,3))
    def forward(self,x,padding):
        b,r,e,t=x.shape
        tf,ff,rf=self.backbone.region_projection(x,padding)
        h=self.backbone.fusion_module(tf,ff)
        region_ids=torch.arange(r,device=x.device).repeat_interleave(e).unsqueeze(0).expand(b,-1)
        electrode_ids=torch.arange(e,device=x.device).repeat(r).unsqueeze(0).expand(b,-1)
        pos=torch.arange(r*e,device=x.device).unsqueeze(0).expand(b,-1)
        h=h+self.backbone.region_embedding(region_ids)+self.backbone.intra_region_pos_embedding(electrode_ids)+self.backbone.positional_embedding(pos)
        h=self.backbone.dropout(h)
        key=padding.view(b,r*e)
        aux=0.0
        for layer in self.backbone.layers:
            h,a=layer(h,key_padding_mask=key)
            if a is not None: aux=aux+a
        h=self.backbone.norm(h)
        valid=(~key).float().unsqueeze(-1)
        pooled=(h*valid).sum(1)/valid.sum(1).clamp_min(1)
        return self.head(pooled),aux


@torch.no_grad()
def evaluate(model,loader,device):
    model.eval(); ys=[]; ps=[]; subjects=[]; loss_sum=0; n=0
    ce=nn.CrossEntropyLoss()
    for x,m,y,sid in loader:
        x,m,y=x.to(device),m.to(device),y.to(device)
        with torch.autocast("cuda",dtype=torch.bfloat16): logits,_=model(x,m)
        loss_sum += ce(logits.float(),y).item()*len(y); n+=len(y)
        ys.extend(y.cpu().tolist()); ps.extend(logits.argmax(1).cpu().tolist()); subjects.extend(sid.tolist())
    recalls=recall_score(ys,ps,labels=[0,1,2],average=None,zero_division=0)
    ya=np.asarray(ys); pa=np.asarray(ps); upper=ya!=2; predicted_upper=pa!=2
    detected=upper & predicted_upper
    return {"loss":loss_sum/max(n,1),"accuracy":accuracy_score(ys,ps),"macro_f1":f1_score(ys,ps,average="macro",zero_division=0),"balanced_accuracy":balanced_accuracy_score(ys,ps),"recall_0":recalls[0],"recall_1":recalls[1],"recall_2":recalls[2],"upper_vs_non_accuracy":float(np.mean(upper==predicted_upper)),"upper_recall":float(np.mean(predicted_upper[upper])),"non_upper_recall":float(np.mean(~predicted_upper[~upper])),"upper_lr_end_to_end_accuracy":float(np.mean(pa[upper]==ya[upper])),"upper_lr_conditional_accuracy":float(np.mean(pa[detected]==ya[detected])) if detected.any() else float('nan'),"upper_lr_conditional_n":int(detected.sum()),"n_trials":n,"n_subjects":len(set(subjects)),"confusion_matrix":confusion_matrix(ys,ps,labels=[0,1,2]).tolist()}


def _git():
    """Resolve git from PATH on Linux, macOS, or Windows."""
    exe = shutil.which("git")
    if exe:
        return exe
    raise RuntimeError("git executable not found; Uni-NTFM repository identity cannot be verified")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',required=True)
    ap.add_argument('--repo',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--fold',type=int,default=0)
    ap.add_argument('--epochs',type=int,default=5)
    ap.add_argument('--batch-size',type=int,default=4)
    ap.add_argument('--lr',type=float,default=2e-4)
    ap.add_argument('--seed',type=int,default=20260813)
    ap.add_argument('--max-trials-per-subject',type=int,default=0)
    ap.add_argument('--compact',action='store_true')
    ap.add_argument('--preload',action='store_true')
    ap.add_argument('--balanced-loss',action='store_true',help='Use inverse-frequency class weights computed only from training subjects.')
    ap.add_argument('--final-test',action='store_true',help='Explicitly evaluate the held-out test set after hyperparameters are frozen.')
    ap.add_argument('--preflight',action='store_true',help='Run one real forward/backward batch and exit.')
    ap.add_argument(
        '--allow-custom-adapter',
        action='store_true',
        help='Explicitly run the non-official supervised adapter; this is not a paper-reproduction run.',
    )
    args=ap.parse_args()
    if not args.allow_custom_adapter:
        raise RuntimeError(
            'Uni-NTFM official release has no downstream classification head or checkpoint. '
            'Use --allow-custom-adapter only for an explicitly non-paper custom adapter run.'
        )
    seed_all(args.seed); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,args.repo); import model as official
    try:
        official_commit = subprocess.check_output(
            [_git(), '-C', args.repo, 'rev-parse', 'HEAD'],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError('Uni-NTFM runtime repository must retain its git identity') from error
    if official_commit != UNINTFM_OFFICIAL_COMMIT:
        raise RuntimeError(
            f'Uni-NTFM commit mismatch: expected={UNINTFM_OFFICIAL_COMMIT}, actual={official_commit}'
        )
    split=pd.read_csv(Path(args.data)/'splits.csv'); subjects=pd.read_csv(Path(args.data)/'subjects.csv')
    dataset_manifest_sha256=sha256_file(Path(args.data)/'SHA256SUMS')
    test=set(split.loc[split.fold==args.fold,'subject_id'].astype(int)); val_fold=1 if args.fold==0 else 0; val=set(split.loc[split.fold==val_fold,'subject_id'].astype(int))-test; all_ids=set(subjects.subject_id.astype(int)); train=all_ids-test-val
    if not train or not val or not test: raise RuntimeError('empty split')
    split_hash = hashlib.sha256(
        ",".join(map(str, sorted(train))).encode("utf-8")
    ).hexdigest()
    cfg = json.loads((Path(args.data) / "config.json").read_text(encoding="utf-8"))
    adapter = UniNTFMAdapter(cfg["channel_order"])

    def _raw_stream(ids):
        for sid in sorted(ids):
            name = subjects.loc[sid, "subject_name"]
            with np.load(Path(args.data) / "data" / f"{name}.npz", allow_pickle=False) as z:
                yield sid, z["X"], z["y"]

    adapter.fit(_raw_stream(train), split_hash)
    adapter.save_manifest(out)
    split_sets=[('train',train),('val',val)]
    if args.final_test:
        split_sets.append(('test',test))
    datasets={k:MI3Trials(args.data,v,args.max_trials_per_subject,args.preload and k!='test',adapter=adapter) for k,v in split_sets}
    loaders={k:DataLoader(d,batch_sampler=SubjectBatchSampler(d,args.batch_size,k=='train'),num_workers=0,pin_memory=True) for k,d in datasets.items()}
    device=torch.device('cuda'); net=UniClassifier(official,args.compact).to(device)
    if args.preflight:
        net.train(); torch.cuda.reset_peak_memory_stats(); started=time.time()
        x,m,y,_=next(iter(loaders['train'])); x,m,y=x.to(device),m.to(device),y.to(device)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits,aux=net(x,m); preflight_loss=nn.CrossEntropyLoss()(logits,y)+(1e-3*aux if torch.is_tensor(aux) else 0)
        preflight_loss.backward(); torch.cuda.synchronize()
        print(json.dumps({'preflight':'ok','model':'Uni-NTFM','variant':'compact' if args.compact else 'full','logits_shape':list(logits.shape),'parameters':sum(p.numel() for p in net.parameters()),'seconds':time.time()-started,'peak_gpu_memory_mib':torch.cuda.max_memory_allocated()/2**20,'test_dataset_instantiated':False,'benchmark_track':UNINTFM_BENCHMARK_TRACK,'dataset_manifest_sha256':dataset_manifest_sha256}),flush=True)
        return
    opt=torch.optim.AdamW(net.parameters(),lr=args.lr,weight_decay=1e-4)
    class_weights=None
    if args.balanced_loss:
        counts=np.zeros(3,dtype=np.int64)
        for sid in sorted(train):
            name=datasets['train'].subjects.loc[sid,'subject_name']
            with np.load(Path(args.data)/'data'/f'{name}.npz',allow_pickle=False) as z:
                counts += np.bincount(z['y'],minlength=3)[:3]
        class_weights=torch.tensor(counts.sum()/(3.0*counts),dtype=torch.float32,device=device)
        print(json.dumps({'training_class_counts':counts.tolist(),'class_weights':class_weights.cpu().tolist()}),flush=True)
    ce=nn.CrossEntropyLoss(weight=class_weights); scaler=torch.amp.GradScaler('cuda'); best=-1; history=[]
    for epoch in range(1,args.epochs+1):
        net.train(); total=0;n=0;t0=time.time()
        for x,m,y,_ in loaders['train']:
            x,m,y=x.to(device,non_blocking=True),m.to(device,non_blocking=True),y.to(device,non_blocking=True); opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16): logits,aux=net(x,m); loss=ce(logits,y)+(1e-3*aux if torch.is_tensor(aux) else 0)
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); scaler.step(opt); scaler.update(); total+=loss.item()*len(y);n+=len(y)
        val_metrics=evaluate(net,loaders['val'],device); row={'epoch':epoch,'train_loss':total/n,'seconds':time.time()-t0,**{f'val_{k}':v for k,v in val_metrics.items() if k!='confusion_matrix'}};history.append(row);print(json.dumps(row),flush=True)
        if val_metrics['macro_f1']>best: best=val_metrics['macro_f1'];torch.save({'model':net.state_dict(),'args':vars(args),'epoch':epoch},out/'best.pt')
    checkpoint=torch.load(out/'best.pt',map_location=device,weights_only=False);net.load_state_dict(checkpoint['model'])
    rows=[]
    for name in (['val','test'] if args.final_test else ['val']):
        met=evaluate(net,loaders[name],device); rows.append({'model':'Uni-NTFM','implementation':'protocol-preserving supervised adapter (official backbone; frozen downstream protocol)','variant':'compact' if args.compact else 'full','dataset_version':json.loads((Path(args.data)/'config.json').read_text(encoding='utf-8')).get('version','unknown'),'dataset_manifest_sha256':dataset_manifest_sha256,'model_adapter_kind':'model_input_adapter','input_channels':32,'fold':args.fold,'split':name,'seed':args.seed,'best_epoch':checkpoint['epoch'],'benchmark_track':UNINTFM_BENCHMARK_TRACK,**{k:v for k,v in met.items() if k!='confusion_matrix'},'confusion_matrix':json.dumps(met['confusion_matrix'])})
    pd.DataFrame(rows).to_csv(out/'results.csv',index=False);pd.DataFrame(history).to_csv(out/'history.csv',index=False)
    (out/'run_config.json').write_text(json.dumps({
        'args':vars(args),
        'train_subjects':sorted(train),
        'val_subjects':sorted(val),
        'test_subjects':sorted(test),
        'test_was_not_used_for_selection':True,
        'dataset_manifest_sha256':dataset_manifest_sha256,
        'model_adapter_manifest': adapter.manifest(),
        'benchmark_track': UNINTFM_BENCHMARK_TRACK,
        'benchmark_contract': {
            'official_repo_commit': official_commit,
            'downstream_head': 'LayerNorm(embed_dim) -> Linear(embed_dim, 3)',
            'input_shape': '[B, 5, 24, 750]',
            'channel_mapping': 'official five-region token grid',
            'per_trial_zscore': False,
            'mask_semantics': 'provenance only; no signal zeroing',
            'input_unit_scale': 'V->uV x1e6 (adapter fix 2026-08-27; validated: V collapses, uV escapes)',
            'status': 'protocol_benchmark',
        },
        'paper_reproduction': {
            'status': 'not_claimed',
            'official_repo_commit': official_commit,
            'official_repo_expected_commit': UNINTFM_OFFICIAL_COMMIT,
            'downstream_head': 'LayerNorm(embed_dim) -> Linear(embed_dim, 3)',
            'input_samples_hz': 750,
            'official_model_sequence_length': 1600,
            'compact_config': {
                'embed_dim': net.cfg.embed_dim,
                'num_heads': net.cfg.num_heads,
                'depth': net.cfg.depth,
                'use_moe': net.cfg.use_moe,
            },
        },
    },indent=2),encoding='utf8');print(pd.DataFrame(rows).to_csv(index=False),flush=True)

if __name__=='__main__': main()
