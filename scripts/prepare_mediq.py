#!/usr/bin/env python
"""
prepare_mediq.py -- build the MediQ-derived NA/NAP transfer pairs.

Original construction code used for the paper's transfer evaluation; only the
input/output paths were turned into arguments. It reproduces the files used in
the paper byte-for-byte from the source file below (verified by SHA-256):

  source : stellalisy/mediQ  data/all_dev_good.jsonl  (1,272 records)
           sha256 3bfc7090d060dd8d11e4237344ed78846707faab433a84d078191627ad3c9526
  output : mediq_na_nap.jsonl     sha256 2b01ec574a1557b8...
           mediq_na_nap_qc.jsonl  sha256 713aa6a0c7212ac8...

Selection rule (automatic, no LLM or human labels): for each item, remove the
single context sentence with the largest weighted lexical overlap with the
question stem and the gold option (gold-option tokens weighted x2); if no
sentence overlaps, remove the last context sentence.

The stop-word list is kept exactly as used for the paper; editing it changes
the constructed pairs.

The source file is not redistributed here. Download it yourself after reviewing
the MediQ license/terms, e.g.
  curl -L https://raw.githubusercontent.com/stellalisy/mediQ/main/data/all_dev_good.jsonl -o data/mediq/all_dev_good.jsonl

Usage:
  python scripts/prepare_mediq.py --source data/mediq/all_dev_good.jsonl --out_dir data/mediq
"""
import argparse as _ap
_p=_ap.ArgumentParser(); _p.add_argument("--source", required=True); _p.add_argument("--out_dir", default="data/mediq")
_ARGS=_p.parse_args()
import os as _os
if not _os.path.exists(_ARGS.source):
    raise SystemExit(f"MediQ source not found: {_ARGS.source} (see the download note in this file's docstring)")
import json, re, os, statistics as st
from collections import Counter

recs=[json.loads(l) for l in open(_ARGS.source)]
def wellformed(r):
    opts=r.get("options") or {}; ctx=r.get("context") or []
    return (isinstance(opts,dict) and len(opts)>=2 and r.get("answer_idx") in opts
            and isinstance(ctx,list) and len(ctx)>=2
            and isinstance(r.get("question"),str) and r["question"].strip())
good=[r for r in recs if wellformed(r)]

STOP=set("a an the of to in on at for with and or is are was were be been being has have had "
         "this that these those which who whom whose what when where why how as by from into "
         "his her their its it he she they them patient year old years男 shows show given most likely "
         "following which of the".split())
def toks(s):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP and len(w)>2]

def auto_decisive(rec, topk=1):
    """Remove the context sentence(s) most lexically tied to (question + gold option)."""
    gold_text = rec["options"][rec["answer_idx"]]
    target = Counter(toks(rec["question"]) + toks(gold_text)*2)   # weight gold option
    scores=[]
    for i,s in enumerate(rec["context"]):
        st_=toks(s)
        overlap=sum(target[w] for w in set(st_))
        scores.append((overlap,i))
    scores.sort(reverse=True)
    chosen=[i for sc,i in scores[:topk] if sc>0]
    if not chosen:                       # zero overlap anywhere -> remove last (still automatic)
        chosen=[len(rec["context"])-1]
    return sorted(chosen)

def assemble(ctx_sents, stem):
    vign=" ".join(x.strip() for x in ctx_sents if x.strip())
    return (vign+"\n\n"+stem.strip()) if vign else stem.strip()

items=[]; qc=[]
for rec in good:
    idx=auto_decisive(rec, topk=1)
    keep=[s for i,s in enumerate(rec["context"]) if i not in idx]
    sid=rec["id"]
    q_na=assemble(rec["context"], rec["question"])
    q_nap=assemble(keep, rec["question"])
    for v,q in [("NA",q_na),("NAP",q_nap)]:
        items.append({"sample_id":sid,"variant":v,"question":q,
                      "choices":rec["options"],"answer":rec["answer_idx"]})
    qc.append({"sample_id":sid,"n_ctx":len(rec["context"]),"n_removed":len(idx),
               "removed_idx":idx,"mode":"auto_overlap","nap_empty":len(keep)==0})

os.makedirs(_ARGS.out_dir,exist_ok=True)
with open(_ARGS.out_dir+"/mediq_na_nap.jsonl","w") as f:
    for it in items: f.write(json.dumps(it)+"\n")
with open(_ARGS.out_dir+"/mediq_na_nap_qc.jsonl","w") as f:
    for q in qc: f.write(json.dumps(q)+"\n")

rem=[q["n_removed"] for q in qc]; nctx=[q["n_ctx"] for q in qc]
n_empty=sum(1 for q in qc if q["nap_empty"])
n_zero_overlap=sum(1 for q,r in zip(qc,good) if q["removed_idx"]==[len(r["context"])-1] and True)  # approx
report={"source":"stellalisy/mediQ all_dev_good.jsonl","n_source_items":len(recs),
 "n_wellformed":len(good),"n_pairs":len(good),"n_rows":len(items),
 "ablation_mode":"automatic lexical overlap (no LLM/human sentence labels)",
 "removed_per_item":1,"nap_empty":n_empty,
 "ctx_per_item_mean":round(st.mean(nctx),2),"ctx_per_item_median":st.median(nctx),
 "removed_fraction_mean":round(st.mean([r/c for r,c in zip(rem,nctx)]),3)}
json.dump(report, open(_ARGS.out_dir+"/mediq_conversion_report.json","w"), indent=2)
print(json.dumps(report,indent=2))
