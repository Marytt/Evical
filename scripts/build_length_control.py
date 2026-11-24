#!/usr/bin/env python
"""
build_matched_deletion_control.py
=================================
Construct an approximately length-matched, NON-SELECTED sentence-deletion control
for the EVICAL / MedAbstain rebuttal experiment. DATA CONSTRUCTION ONLY — no model,
no GPU, no probability/hidden-state/relevance signal is used anywhere.

Three matched conditions per retained item:
  A. control_na.json            : exact original NA item (full context)
  B. control_selected_nap.json  : exact original MedAbstain NAP (selected sentence removed)
  C. control_matched_nap.json   : NA with ONE length-matched NON-selected narrative
                                  sentence removed (deterministic, earliest-index tie-break)

Determinism: fixed seed 42; the control sentence is chosen purely by
  (a) not the selected sentence, (b) not the final question / no '?',
  (c) >=5 tokens, (d) verbatim in NA, (e) length ratio in [0.8, 1.2],
  minimizing |L_candidate - L_selected|, ties broken by earliest sentence index.

Token rule (single, deterministic): whitespace tokenization, len(sentence.split()).
"""
import re, os, json, csv, argparse, random, statistics, sys
import urllib.request

SEED = 42
BASE_URL = "https://raw.githubusercontent.com/sravanthi6m/MedAbstain/main/data/"
TEST_NA  = "medqa_1_test_noabst.json"
TEST_NAP = "perturbed_medqa_1_test_noabst.json"
OUTDIR   = "control_dataset"

# ----------------------------------------------------------------------------- IO
def load_json_url_or_path(name, local):
    if local and os.path.exists(os.path.join(local, name)):
        with open(os.path.join(local, name)) as f:
            return json.load(f)
    with urllib.request.urlopen(BASE_URL + name, timeout=60) as r:
        return json.load(r)

# ------------------------------------------------------------------- segmentation
ABBR = ["e.g","i.e","etc","vs","Dr","Mr","Mrs","Ms","Prof","St","Fig","No",
        "approx","a.m","p.m","b.i.d","t.i.d","q.i.d","q.d","p.r.n","Sr","Jr","Inc","Ltd"]

def protect(text):
    """Replace non-boundary periods with a same-length sentinel so they do not split."""
    w = list(text)
    for m in re.finditer(r"\d\.\d", text):            # decimals: 37.5
        w[m.start()+1] = "\x00"
    for m in re.finditer(r"(?<![A-Za-z])[A-Z]\.", text):  # single-letter initials: J.
        w[m.end()-1] = "\x00"
    for m in re.finditer(r"\b(?:[A-Za-z]\.){2,}", text):  # dotted abbrevs: e.g. i.e. a.m.
        for k in range(m.start(), m.end()):
            if text[k] == ".":
                w[k] = "\x00"
    low = text.lower()
    for ab in ABBR:                                    # trailing-period abbrevs: Dr. vs. etc.
        for m in re.finditer(r"\b" + re.escape(ab.lower()) + r"\.", low):
            w[m.end()-1] = "\x00"
    return "".join(w)

def segment(text):
    """Return list of (start,end) char spans; text[start:end] is a sentence with no
    leading/trailing separator whitespace. Coverage is complete and deterministic."""
    w = protect(text)
    n = len(text)
    bounds = []
    for m in re.finditer(r"[.!?]+[\"')\]]*", w):
        end = m.end()
        rest = w[end:]
        if rest == "" or re.match(r"\s*$", rest) or re.match(r"\s+[\"'(\[]?[A-Z0-9]", rest):
            bounds.append(end)
    spans, start = [], 0
    for b in bounds:
        s = start
        while s < b and text[s].isspace():
            s += 1
        if s < b:
            spans.append((s, b))
        start = b
    s = start
    while s < n and text[s].isspace():
        s += 1
    if s < n:
        e = n
        while e > s and text[e-1].isspace():
            e -= 1
        spans.append((s, e))
    return spans

def sent_texts(text, spans):
    return [text[s:e] for s, e in spans]

def norm(x):
    return re.sub(r"\s+", " ", x).strip()

def tok(s):
    return len(s.split())

def delete_sentence(text, spans, i):
    """Exact deletion of sentence i (with one adjacent separator), minimal ws cleanup."""
    if i < len(spans) - 1:
        new = text[:spans[i][0]] + text[spans[i+1][0]:]
    else:
        new = text[:spans[i-1][1]] + text[spans[i][1]:] if i > 0 else text[spans[i][1]:]
    return re.sub(r"[ \t]{2,}", " ", new).strip()

def delete_range(text, spans, i, j):
    lo = spans[i][0]
    hi = spans[j+1][0] if j < len(spans)-1 else len(text)
    new = text[:lo] + text[hi:]
    return re.sub(r"[ \t]{2,}", " ", new).strip()

def is_contiguous_removal(NA, NAP):
    a, b = norm(NA), norm(NAP)
    if len(b) >= len(a):
        return False
    p = 0
    while p < len(b) and a[p] == b[p]:
        p += 1
    sfx = 0
    while sfx < len(b)-p and a[len(a)-1-sfx] == b[len(b)-1-sfx]:
        sfx += 1
    return p + sfx >= len(b) - 2   # allow a couple of chars slack at the junction

def classify(NA, NAP, spans):
    """Return (status, selected_index). status in
    {clean, multi_sentence_deletion, non_exact_deletion, rewrite_detected}."""
    nNAP = norm(NAP)
    single = [i for i in range(len(spans)) if norm(delete_sentence(NA, spans, i)) == nNAP]
    if single:
        return "clean", min(single)
    for i in range(len(spans)):
        for j in range(i+1, len(spans)):
            if norm(delete_range(NA, spans, i, j)) == nNAP:
                return "multi_sentence_deletion", None
    if is_contiguous_removal(NA, NAP):
        return "non_exact_deletion", None
    return "rewrite_detected", None

QUESTION_PREFIXES = ("which of the following", "what is the most likely", "what is the next best",
    "which is the most appropriate", "which of the following is", "what is the diagnosis",
    "which finding", "what should", "how should", "which treatment")

def is_question_sentence(s, is_last):
    if "?" in s:
        return True
    sl = s.strip().lower()
    if is_last and any(sl.startswith(p) for p in QUESTION_PREFIXES):
        return True
    return False

# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", default=None, help="dir with local NA/NAP json (else fetch)")
    ap.add_argument("--outdir", default=OUTDIR)
    a = ap.parse_args()
    random.seed(SEED)
    os.makedirs(a.outdir, exist_ok=True)

    na_list  = load_json_url_or_path(TEST_NA,  a.local)
    nap_list = load_json_url_or_path(TEST_NAP, a.local)
    for lst, nm in [(na_list, "NA"), (nap_list, "NAP")]:
        assert isinstance(lst, list) and lst, f"{nm} empty/not a list"
        for it in lst[:5]:
            assert {"id", "question", "choices", "answer"} <= set(it.keys()), \
                f"{nm} missing required fields; got {list(it.keys())}"
    nap_by_id = {x["id"]: x for x in nap_list}

    retained, excluded = [], []
    N_original = 0
    for na in na_list:
        nap = nap_by_id.get(na["id"])
        if nap is None:
            excluded.append((na["id"], "alignment_failure")); continue
        N_original += 1
        if na["answer"] != nap["answer"] or na["choices"] != nap["choices"]:
            excluded.append((na["id"], "alignment_failure")); continue

        spans = segment(na["question"])
        if len(spans) < 3:                       # need >=1 narrative + question + a candidate
            excluded.append((na["id"], "no_valid_candidate")); continue
        status, sel_i = classify(na["question"], nap["question"], spans)
        if status != "clean":
            excluded.append((na["id"], status)); continue

        texts = sent_texts(na["question"], spans)
        sel_sent = texts[sel_i]
        sel_len = tok(sel_sent)
        last_idx = len(spans) - 1

        # candidate control sentences
        cands = []
        for i, s in enumerate(texts):
            if i == sel_i:                       continue
            if is_question_sentence(s, i == last_idx): continue
            if "?" in s:                         continue
            if tok(s) < 5:                       continue
            if s not in na["question"]:          continue      # verbatim
            cands.append((i, s, tok(s)))
        if not cands:
            excluded.append((na["id"], "no_valid_candidate")); continue

        # strict +-20% length window; no fallback
        elig = [(i, s, L) for (i, s, L) in cands if 0.8 <= L/sel_len <= 1.2]
        if not elig:
            excluded.append((na["id"], "no_candidate_within_20pct")); continue

        # min |L - sel_len|, tie-break earliest index
        elig.sort(key=lambda t: (abs(t[2]-sel_len), t[0]))
        ctrl_i, ctrl_sent, ctrl_len = elig[0]

        matched_q = delete_sentence(na["question"], spans, ctrl_i)
        retained.append(dict(
            id=na["id"], na=na, nap=nap,
            sel_sent=sel_sent, sel_i=sel_i, sel_len=sel_len,
            ctrl_sent=ctrl_sent, ctrl_i=ctrl_i, ctrl_len=ctrl_len,
            matched_q=matched_q, ratio=ctrl_len/sel_len))

    # ---- build the three condition items (exact order = retained order) ----
    ctrl_na  = [r["na"] for r in retained]
    ctrl_nap = [r["nap"] for r in retained]
    ctrl_mat = [dict(id=r["na"]["id"], source=r["na"].get("source", ""),
                     question=r["matched_q"], choices=r["na"]["choices"],
                     answer=r["na"]["answer"]) for r in retained]

    # =========================== SANITY CHECKS (hard) ===========================
    fails = 0
    ids = [r["id"] for r in retained]
    assert len(ids) == len(set(ids)), "23: duplicate IDs"                       # 23
    for k, r in enumerate(retained):
        na_q  = r["na"]["question"]; nap_q = r["nap"]["question"]; mat_q = r["matched_q"]
        A, B, C = ctrl_na[k], ctrl_nap[k], ctrl_mat[k]
        c = []
        c.append(A["id"] == B["id"] == C["id"] == r["id"])                       # 1,2
        c.append(A["choices"] == B["choices"] == C["choices"])                   # 3,21,24
        c.append(A["answer"]  == B["answer"]  == C["answer"])                    # 4,24
        c.append(norm(na_q) != norm(nap_q))                                      # 5
        c.append(norm(na_q) != norm(mat_q))                                      # 6
        c.append(norm(nap_q) != norm(mat_q))                                     # 7
        c.append(r["sel_sent"] != r["ctrl_sent"])                                # 8
        c.append(r["sel_sent"] in na_q)                                          # 9
        c.append(r["ctrl_sent"] in na_q)                                         # 10
        c.append(norm(r["sel_sent"]) not in norm(nap_q))                         # 11
        c.append(norm(r["ctrl_sent"]) in norm(nap_q))                            # 12
        c.append(norm(r["ctrl_sent"]) not in norm(mat_q))                        # 13
        c.append(norm(r["sel_sent"]) in norm(mat_q))                             # 14
        # 15: NAP == NA minus exactly one sentence (the selected one)
        sp = segment(na_q)
        c.append(norm(delete_sentence(na_q, sp, r["sel_i"])) == norm(nap_q))     # 15
        c.append(norm(delete_sentence(na_q, sp, r["ctrl_i"])) == norm(mat_q))    # 16
        c.append("?" not in r["ctrl_sent"])                                      # 17
        c.append(r["ctrl_i"] != len(sp) - 1)                                     # 18
        c.append(tok(r["ctrl_sent"]) >= 5)                                       # 19
        c.append(0.8 <= r["ctrl_len"]/r["sel_len"] <= 1.2)                       # 20
        c.append(bool(mat_q.strip()) and bool(nap_q.strip()))                    # 22
        if not all(c):
            fails += 1
            bad = [i+1 for i, ok in enumerate(c) if not ok]
            print(f"SANITY FAIL id={r['id']} checks={bad}", file=sys.stderr)
    assert fails == 0, f"{fails} retained items failed sanity checks"

    # =========================== WRITE OUTPUTS ===========================
    P = lambda f: os.path.join(a.outdir, f)
    json.dump(ctrl_na,  open(P("control_na.json"), "w"))
    json.dump(ctrl_nap, open(P("control_selected_nap.json"), "w"))
    json.dump(ctrl_mat, open(P("control_matched_nap.json"), "w"))

    meta_cols = ["id","selected_sentence","control_sentence","selected_tokens","control_tokens",
        "length_ratio","selected_sentence_index","control_sentence_index",
        "clean_single_sentence_deletion","selection_rule","exclusion_reason",
        "na_question","selected_nap_question","matched_control_question"]
    with open(P("control_metadata.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(meta_cols)
        for r in retained:
            w.writerow([r["id"], r["sel_sent"], r["ctrl_sent"], r["sel_len"], r["ctrl_len"],
                f"{r['ratio']:.4f}", r["sel_i"], r["ctrl_i"], True,
                "closest_length_nonselected_sentence_within_20pct", "",
                r["na"]["question"], r["nap"]["question"], r["matched_q"]])
    with open(P("control_excluded_items.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id","reason"]); w.writerows(excluded)

    # ---- statistics ----
    sel_t = [r["sel_len"] for r in retained]; ctl_t = [r["ctrl_len"] for r in retained]
    rat   = [r["ratio"] for r in retained]
    from collections import Counter
    exc_counts = Counter(reason for _, reason in excluded)
    n_clean = len(retained) + exc_counts.get("no_valid_candidate", 0) + exc_counts.get("no_candidate_within_20pct", 0)
    def stats(x): return dict(mean=statistics.mean(x), median=statistics.median(x),
                              std=(statistics.pstdev(x) if len(x)>1 else 0.0),
                              min=min(x), max=max(x))
    audit = dict(seed=SEED, token_rule="whitespace_split_len",
        original_matched_pairs=N_original, retained=len(retained), excluded=len(excluded),
        exclusion_counts=dict(exc_counts),
        selected_token_stats=stats(sel_t), control_token_stats=stats(ctl_t),
        length_ratio_stats=stats(rat),
        n_control_with_question_mark=sum("?" in r["ctrl_sent"] for r in retained),
        n_final_question_selected=sum(r["ctrl_i"]==len(segment(r["na"]["question"]))-1 for r in retained),
        n_answer_choices_modified=sum(ctrl_mat[k]["choices"]!=ctrl_na[k]["choices"] for k in range(len(retained))),
        n_sanity_failures=fails)
    json.dump(audit, open(P("control_audit.json"), "w"), indent=2)

    # ---- summary txt ----
    L = []
    L.append("MATCHED-DELETION CONTROL — DATASET SUMMARY")
    L.append("(approximately length-matched non-selected sentence deletion control)")
    L.append("")
    L.append(f"Original matched pairs:                 {N_original}")
    L.append(f"Clean single-sentence deletions:        {n_clean}")
    L.append(f"Retained with valid length-matched ctrl:{len(retained)}")
    L.append(f"Excluded:                               {len(excluded)}")
    L.append("")
    L.append("Exclusion counts:")
    for reason in ["non_exact_deletion","multi_sentence_deletion","rewrite_detected",
                   "no_valid_candidate","no_candidate_within_20pct","alignment_failure","other"]:
        L.append(f"  {reason}: {exc_counts.get(reason,0)}")
    L.append("")
    for nm, x in [("Selected sentence", sel_t), ("Control sentence", ctl_t)]:
        st = stats(x)
        L.append(f"{nm} tokens:  mean={st['mean']:.2f} median={st['median']} std={st['std']:.2f}")
    st = stats(rat)
    L.append(f"Length ratio:  mean={st['mean']:.3f} median={st['median']:.3f} "
             f"min={st['min']:.3f} max={st['max']:.3f}")
    L.append("")
    L.append(f"control sentences containing '?':   {audit['n_control_with_question_mark']} (must be 0)")
    L.append(f"final-question sentences selected:  {audit['n_final_question_selected']} (must be 0)")
    L.append(f"answer choices modified:            {audit['n_answer_choices_modified']} (must be 0)")
    L.append(f"sanity-check failures:              {audit['n_sanity_failures']} (must be 0)")
    L.append("")
    L.append("10 randomly sampled retained examples (seed=42):")
    for r in random.Random(SEED).sample(retained, min(10, len(retained))):
        L.append(f"  id={r['id']}")
        L.append(f"    SELECTED ({r['sel_len']} tok): {r['sel_sent']}")
        L.append(f"    CONTROL  ({r['ctrl_len']} tok, ratio={r['ratio']:.2f}): {r['ctrl_sent']}")
    open(P("control_dataset_summary.txt"), "w").write("\n".join(L))

    # ---- console report ----
    print("\n".join(L))
    print("\nOUTPUT PATHS:")
    for f in ["control_na.json","control_selected_nap.json","control_matched_nap.json",
              "control_metadata.csv","control_excluded_items.csv",
              "control_dataset_summary.txt","control_audit.json"]:
        print("  ", os.path.abspath(P(f)))

if __name__ == "__main__":
    main()
