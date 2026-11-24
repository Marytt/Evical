"""
src/features/extractor.py
Extracts hidden states at ALL candidate layers in ONE forward pass.

Correctness properties:
  * Cache identity covers EVERYTHING that can change a feature: model, layers,
    prompt_mode, conf_scale, dataset version + filenames, max_length. A T9 run
    can never reuse a T100 cache.
  * Extraction is RESUMABLE. Progress is written every `save_every` items with a
    data fingerprint; a killed instance resumes instead of restarting.
  * Tokenization is truncated to a fixed max_length for every model, so no
    architecture silently sees a different effective prompt.
  * The HF cache directory is passed explicitly to tokenizer and model.
"""

import os
import json
import hashlib
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple

# transformers is imported lazily inside extract_and_cache() so that
# cache-only runs (and the smoke test) do not require it.

LETTERS = ["A", "B", "C", "D", "E", "F"]


# --- Prompt helpers ---------------------------------------------------------

def build_msg(q: str, choices: dict, conf_scale: str) -> str:
    opts = "\n".join(f"{k}. {v}" for k, v in choices.items())
    conf_instr = ("a single digit 0-9, where 9 is most confident"
                  if conf_scale == "T9"
                  else "a number 0-100, where 100 is most confident")
    return (f"The following is a multiple-choice medical question with "
            f"{len(choices)} options. Only one option is correct.\n\n"
            f"{q}\n\n{opts}\n\n"
            f"Respond in exactly this format:\n"
            f"Answer: [letter]\nConfidence: [{conf_instr}]")


def build_prompt(tok, q: str, choices: dict,
                 conf_scale: str, prompt_mode: str) -> str:
    msg = build_msg(q, choices, conf_scale)
    if prompt_mode == "chat":
        return tok.apply_chat_template(
            [{"role": "user", "content": msg}],
            tokenize=False, add_generation_prompt=True)
    return msg + "\n"


def build_letter_ids(tok, choices: dict, device) -> torch.Tensor:
    return torch.tensor(
        [tok.encode(" " + L, add_special_tokens=False)[-1]
         for L in LETTERS[:len(choices)]],
        dtype=torch.long, device=device)


def build_conf_token_ids(tok, conf_scale: str, device):
    numbers = [str(i) for i in range(10 if conf_scale == "T9" else 101)]
    ids = torch.tensor(
        [tok.encode(n, add_special_tokens=False)[-1] for n in numbers],
        dtype=torch.long, device=device)
    n_unique = len(set(ids.tolist()))
    assert n_unique == len(numbers), (
        f"Token collision in conf_scale={conf_scale}: "
        f"{n_unique}/{len(numbers)} unique ids. Use a coarser scale.")
    scores = torch.arange(0, len(numbers), dtype=torch.float32, device=device) \
             / (len(numbers) - 1)
    return ids, scores


# --- Cache identity (FIX 1) -------------------------------------------------

def cache_identity(model_cfg: dict, pipeline_cfg: dict) -> Dict:
    """
    Every field that can change an extracted feature. Anything omitted here is
    a silent-cache-reuse bug waiting to happen.
    """
    d = pipeline_cfg["data"]
    e = pipeline_cfg.get("extraction", {})
    return {
        "model_name":   model_cfg["model_name"],
        "prompt_mode":  model_cfg["prompt_mode"],
        "conf_scale":   model_cfg["conf_scale"],
        "layer_fracs":  sorted(pipeline_cfg["pipeline"]["layer_fracs"]),
        "max_length":    e.get("max_length", 2048),
        "answer_prefix": e.get("answer_prefix", "Answer:"),
        "dataset_name":    pipeline_cfg.get("dataset", "medabstain"),
        "dataset_version": (pipeline_cfg.get("datasets", {})
                            .get(pipeline_cfg.get("dataset", "medabstain"), {})
                            .get("version") or d.get("version", "unversioned")),
        "dataset_files": {
            "train_file": d["train_file"], "pool_file": d["pool_file"],
            "test_na":    d["test_na"],    "test_nap":  d["test_nap"],
        },
        "extractor_schema": "v3",   # bump when the feature format changes
    }


def cache_key(model_cfg: dict, pipeline_cfg: dict) -> str:
    ident = cache_identity(model_cfg, pipeline_cfg)
    blob  = json.dumps(ident, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def data_fingerprint(paired_train, na_test, nap_test) -> str:
    """Guards resume: refuses to continue a partial cache built on other data."""
    ids = ([f"{a['id']}|{b['id']}" for a, b in paired_train]
           + [str(x["id"]) for x in na_test]
           + [str(x["id"]) for x in nap_test])
    return hashlib.sha256("_".join(ids).encode()).hexdigest()[:16]


# --- Extraction -------------------------------------------------------------

@torch.no_grad()
def _extract_item(item, model, tok, layer_idxs, conf_scale,
                  prompt_mode, device, max_length, answer_prefix="Answer:"):
    """
    READ POSITION (critical):
    The prompt ends by instructing the output FORMAT, so the model's next token
    is "Answer", not an option letter. Reading letter logits there measures
    nothing about the answer. Appending `answer_prefix` moves the read position
    to where the letter is actually decided. Verified on Qwen2.5-7B/MedQA:
    accuracy 0.40 -> 0.57 (published range ~0.55-0.60).
    The hidden state for the confidence head is read at the same position.
    """
    prompt   = build_prompt(tok, item["question"], item["choices"],
                            conf_scale, prompt_mode)
    read_text = prompt + answer_prefix
    # FIX 3: fixed truncation for every model
    inputs = tok(read_text, return_tensors="pt",
                 truncation=True, max_length=max_length).to(device)
    n_tok  = int(inputs["input_ids"].shape[1])
    out    = model(**inputs, output_hidden_states=True)

    # hidden states at all candidate layers -- ONE forward pass
    h_all = {frac: out.hidden_states[li][0, -1, :].float().cpu()
             for frac, li in layer_idxs.items()}

    lids      = build_letter_ids(tok, item["choices"], device)
    opt_logits = out.logits[0, -1, lids].float().cpu()  # raw option-letter logits
    probs     = F.softmax(opt_logits, dim=0)
    pred_idx  = probs.argmax().item()
    pred      = LETTERS[pred_idx]
    gold_idx  = (LETTERS.index(item["answer"])
                 if item["answer"] in LETTERS else 0)
    gold_prob = float(probs[min(gold_idx, len(probs) - 1)].cpu())
    return h_all, pred, gold_prob, n_tok, opt_logits.tolist()


def _partial_path(cache_path: str) -> str:
    return cache_path + ".partial"


def _load_partial(cache_path: str, fingerprint: str, ident: Dict):
    """Return (train, test_na, test_nap) already done, or empty lists."""
    pp = _partial_path(cache_path)
    if not os.path.exists(pp):
        return [], [], []
    try:
        p = torch.load(pp, map_location="cpu")
    except Exception as ex:
        print(f"[extractor] partial cache unreadable ({ex}); restarting")
        return [], [], []
    if p.get("fingerprint") != fingerprint or p.get("identity") != ident:
        print("[extractor] partial cache is for different data/settings; restarting")
        return [], [], []
    tr, na, nap = p.get("train", []), p.get("test_na", []), p.get("test_nap", [])
    print(f"[extractor] RESUMING from partial: train={len(tr)} "
          f"test_na={len(na)} test_nap={len(nap)}")
    return tr, na, nap


def _save_partial(cache_path, train, test_na, test_nap,
                  fingerprint, ident, meta):
    pp  = _partial_path(cache_path)
    tmp = pp + ".tmp"
    torch.save({"train": train, "test_na": test_na, "test_nap": test_nap,
                "fingerprint": fingerprint, "identity": ident, **meta}, tmp)
    os.replace(tmp, pp)   # atomic


def extract_and_cache(paired_train, na_test, nap_test,
                      model_cfg: dict, pipeline_cfg: dict,
                      cache_path: str):
    """Loads model, extracts all features (resumable), saves to cache_path."""
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)

    model_name  = model_cfg["model_name"]
    prompt_mode = model_cfg["prompt_mode"]
    conf_scale  = model_cfg["conf_scale"]
    layer_fracs = pipeline_cfg["pipeline"]["layer_fracs"]
    ex_cfg      = pipeline_cfg.get("extraction", {})
    max_length  = ex_cfg.get("max_length", 2048)
    answer_prefix = ex_cfg.get("answer_prefix", "Answer:")
    save_every  = ex_cfg.get("save_every", 200)
    hf_cache    = ex_cfg.get("hf_cache_dir") or None

    ident = cache_identity(model_cfg, pipeline_cfg)
    fp    = data_fingerprint(paired_train, na_test, nap_test)

    # FIX 2: resume whatever is already done
    train_data, test_na_data, test_nap_data = _load_partial(cache_path, fp, ident)

    n_done = len(train_data) + len(test_na_data) + len(test_nap_data)
    n_todo = ((len(paired_train) - len(train_data))
              + (len(na_test)  - len(test_na_data))
              + (len(nap_test) - len(test_nap_data)))
    if n_todo <= 0:
        print("[extractor] partial cache already complete; finalizing")

    print(f"[extractor] Loading model: {model_name} (hf_cache_dir={hf_cache})")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    # FIX 3: explicit cache_dir for both tokenizer and model
    tok   = AutoTokenizer.from_pretrained(model_name, cache_dir=hf_cache)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, device_map="auto",
        output_hidden_states=True, cache_dir=hf_cache,
        # attention backend from the per-model config (null = library default)
        attn_implementation=model_cfg.get("attn_implementation"))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    device     = next(model.parameters()).device
    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    layer_idxs = {f: round(f * num_layers) for f in layer_fracs}

    # fail loudly on confidence-scale collision before doing any work
    build_conf_token_ids(tok, conf_scale, device)

    print(f"[extractor] num_layers={num_layers} hidden_dim={hidden_dim} "
          f"layer_idxs={layer_idxs} max_length={max_length}")
    print(f"[extractor] read position = end of prompt + {answer_prefix!r}")
    print(f"[extractor] resume state: {n_done} done, {max(n_todo,0)} to go")

    meta = {"num_layers": num_layers, "hidden_dim": hidden_dim,
            "layer_idxs": layer_idxs, "model_name": model_name,
            "conf_scale": conf_scale, "max_length": max_length,
            "answer_prefix": answer_prefix}

    def extract(item):
        return _extract_item(item, model, tok, layer_idxs, conf_scale,
                             prompt_mode, device, max_length, answer_prefix)

    trunc_hits = 0

    # -- train pairs (resume from len(train_data)) --
    for i in range(len(train_data), len(paired_train)):
        na_item, nap_item = paired_train[i]
        h_na,  pred_na,  gp_na,  n1, _ = extract(na_item)
        h_nap, pred_nap, gp_nap, n2, _ = extract(nap_item)
        trunc_hits += int(n1 >= max_length) + int(n2 >= max_length)
        train_data.append({
            "h_na": h_na, "h_nap": h_nap,
            "pred_na": pred_na, "pred_nap": pred_nap,
            "y_na":  float(pred_na  == na_item["answer"]),
            "y_nap": float(pred_nap == nap_item["answer"]),
            "gp_na": gp_na, "gp_nap": gp_nap,
            "gold_gap": max(0.0, gp_na - gp_nap),
            "id_na": na_item["id"], "id_nap": nap_item["id"],
        })
        if (i + 1) % save_every == 0:
            print(f"[extractor] train {i+1}/{len(paired_train)} (checkpointed)")
            _save_partial(cache_path, train_data, test_na_data,
                          test_nap_data, fp, ident, meta)

    # -- test sets (resume similarly) --
    for name, src, dst in [("test_na", na_test,  test_na_data),
                           ("test_nap", nap_test, test_nap_data)]:
        for i in range(len(dst), len(src)):
            item = src[i]
            h, pred, gp, n, ol = extract(item)
            trunc_hits += int(n >= max_length)
            dst.append({"h": h, "pred": pred, "gold": item["answer"],
                        "correct": float(pred == item["answer"]),
                        "gold_prob": gp, "id": item["id"],
                        "opt_logits": ol})
            if (i + 1) % save_every == 0:
                print(f"[extractor] {name} {i+1}/{len(src)} (checkpointed)")
                _save_partial(cache_path, train_data, test_na_data,
                              test_nap_data, fp, ident, meta)

    if trunc_hits:
        print(f"[extractor] WARNING: {trunc_hits} prompts hit "
              f"max_length={max_length} and were truncated")

    # -- finalize atomically, then drop the partial --
    tmp = cache_path + ".tmp"
    torch.save({"train": train_data, "test_na": test_na_data,
                "test_nap": test_nap_data, "identity": ident,
                "fingerprint": fp, "truncated_prompts": trunc_hits,
                **meta}, tmp)
    os.replace(tmp, cache_path)
    if os.path.exists(_partial_path(cache_path)):
        os.remove(_partial_path(cache_path))
    print(f"[extractor] Cache saved: {cache_path}")

    del model, tok
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return train_data, test_na_data, test_nap_data, hidden_dim


def load_or_extract(paired_train, na_test, nap_test,
                    model_cfg: dict, pipeline_cfg: dict,
                    cache_dir: str):
    """Loads a complete cache if identity matches, else extracts (resumably)."""
    key        = cache_key(model_cfg, pipeline_cfg)
    cache_path = os.path.join(cache_dir, f"{key}.pt")
    ident      = cache_identity(model_cfg, pipeline_cfg)

    # record the identity next to the cache for human inspection
    json.dump(ident, open(os.path.join(cache_dir, f"{key}.identity.json"), "w"),
              indent=2)

    if os.path.exists(cache_path):
        c = torch.load(cache_path, map_location="cpu")
        if c.get("identity") == ident:
            print(f"[extractor] Loaded cache: {cache_path}")
            print(f"[extractor] train={len(c['train'])} "
                  f"test_na={len(c['test_na'])} test_nap={len(c['test_nap'])}")
            return (c["train"], c["test_na"], c["test_nap"],
                    c["hidden_dim"], cache_path)
        print("[extractor] cache identity mismatch -- re-extracting")

    train_data, test_na_data, test_nap_data, hidden_dim = \
        extract_and_cache(paired_train, na_test, nap_test,
                          model_cfg, pipeline_cfg, cache_path)
    return train_data, test_na_data, test_nap_data, hidden_dim, cache_path
