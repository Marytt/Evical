#!/usr/bin/env python
"""
control_infer.py — apply ALREADY-TRAINED frozen EVICAL heads to the three control
conditions (NA / selected-NAP / matched-control-NAP). Inference only, no retraining.

Faithfully mirrors the pipeline extractor: 4-bit NF4 + bf16 compute, eager attn for
Gemma, hidden state read at the checkpoint's layer_index at the LAST position after
appending 'Answer:'; head_conf = sigmoid-MLP(hidden). gold_prob = softmax over the
option-letter logits at that position.

Output (per model): out/control_preds_<MODEL>.csv with columns
  id, variant(NA|selected|matched), pred, gold, correct, head_conf, gold_prob
"""
import os, json, csv, argparse, glob, re
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

LETTERS = ["A","B","C","D","E","F"]
MODELS = {
  "Qwen2.5-7B-Instruct":        dict(model_name="Qwen/Qwen2.5-7B-Instruct",        conf_scale="T9"),
  "Meta-Llama-3-8B-Instruct":   dict(model_name="meta-llama/Meta-Llama-3-8B-Instruct", conf_scale="T100"),
  "Ministral-8B-Instruct-2410": dict(model_name="mistralai/Ministral-8B-Instruct-2410", conf_scale="T9"),
  "Gemma-2-9B-Instruct":        dict(model_name="google/gemma-2-9b-it",            conf_scale="T9"),
  "Phi-4-Mini-Instruct":        dict(model_name="microsoft/Phi-4-mini-instruct",   conf_scale="T9"),
  "Llama-3.1-8B-Instruct":      dict(model_name="meta-llama/Llama-3.1-8B-Instruct", conf_scale="T100"),
  "Mistral-7B-v0.3":            dict(model_name="mistralai/Mistral-7B-Instruct-v0.3", conf_scale="T9"),
}
MAX_LEN = 2048
ANSWER_PREFIX = "Answer:"

class ConfidenceHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 1), nn.Sigmoid())
    def forward(self, h):
        return self.net(h.float()).squeeze(-1)

def build_msg(q, choices, conf_scale):
    opts = "\n".join(f"{k}. {v}" for k, v in choices.items())
    conf_instr = ("a single digit 0-9, where 9 is most confident" if conf_scale == "T9"
                  else "a number 0-100, where 100 is most confident")
    return (f"The following is a multiple-choice medical question with {len(choices)} "
            f"options. Only one option is correct.\n\n{q}\n\n{opts}\n\n"
            f"Respond in exactly this format:\nAnswer: [letter]\nConfidence: [{conf_instr}]")

def build_prompt(tok, q, choices, conf_scale):
    return tok.apply_chat_template([{"role":"user","content":build_msg(q,choices,conf_scale)}],
                                   tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def score_item(item, model, tok, head, layer_index, device, conf_scale):
    read_text = build_prompt(tok, item["question"], item["choices"], conf_scale) + ANSWER_PREFIX
    inp = tok(read_text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(device)
    out = model(**inp, output_hidden_states=True)
    h = out.hidden_states[layer_index][0, -1, :].float()
    conf = float(head(h).cpu())
    lids = torch.tensor([tok.encode(" "+L, add_special_tokens=False)[-1]
                         for L in LETTERS[:len(item["choices"])]], device=device)
    probs = F.softmax(out.logits[0, -1, lids], dim=0)
    pred = LETTERS[int(probs.argmax())]
    gidx = LETTERS.index(item["answer"]) if item["answer"] in LETTERS else 0
    gold_prob = float(probs[min(gidx, len(probs)-1)].cpu())
    return pred, gold_prob, conf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--heads_dir", required=True, help="dir with <MODEL>__graded/.../final_head.pt")
    ap.add_argument("--data_dir", required=True, help="dir with the three control_*.json files")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only first N items/condition")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    mname = MODELS[a.model]["model_name"]
    conf_scale = MODELS[a.model]["conf_scale"]

    # locate trained head checkpoint for this model
    hits = glob.glob(os.path.join(a.heads_dir, f"**/{a.model}__graded/**/final_head.pt"), recursive=True)
    assert hits, f"no final_head.pt for {a.model} under {a.heads_dir}"
    ck = torch.load(hits[0], map_location="cpu", weights_only=False)
    layer_index = ck["layer_index"]
    in_dim = ck["state_dict"]["net.0.weight"].shape[1]
    harch = ck.get("head_arch", {}) or {}
    head = ConfidenceHead(in_dim, harch.get("hidden_dim", 256), harch.get("dropout", 0.1))
    head.load_state_dict(ck["state_dict"]); head.eval()
    print(f"[{a.model}] head layer_index={layer_index} input_dim={in_dim} val_loss={ck.get('val_loss')}")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    tok = AutoTokenizer.from_pretrained(mname)
    model = AutoModelForCausalLM.from_pretrained(
        mname, quantization_config=bnb, device_map="auto", output_hidden_states=True,
        attn_implementation=("eager" if "gemma" in mname.lower() else None))
    model.eval()
    device = next(model.parameters()).device
    head = head.to(device)
    assert model.config.num_hidden_layers >= layer_index, "layer_index out of range"

    conditions = [("full", "control_na.json"),
                  ("selected", "control_selected_nap.json"),
                  ("matched", "control_matched_nap.json")]
    rows = []
    for variant, fn in conditions:
        items = json.load(open(os.path.join(a.data_dir, fn)))
        if a.limit: items = items[:a.limit]
        for k, it in enumerate(items):
            pred, gp, conf = score_item(it, model, tok, head, layer_index, device, conf_scale)
            rows.append(dict(id=it["id"], variant=variant, pred=pred, gold=it["answer"],
                             correct=int(pred == it["answer"]), head_conf=conf, gold_prob=gp))
        print(f"[{a.model}] {variant}: scored {len(items)} items")
    out = os.path.join(a.outdir, f"control_preds_{a.model}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id","variant","pred","gold","correct","head_conf","gold_prob"])
        w.writeheader(); w.writerows(rows)
    print(f"[{a.model}] wrote {out} ({len(rows)} rows)")

if __name__ == "__main__":
    main()
