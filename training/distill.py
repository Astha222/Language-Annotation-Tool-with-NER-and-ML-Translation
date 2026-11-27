"""Distil an English NER teacher into a small, fast student (evaluation plan, WS2.5).

    python training/distill.py --teacher Jean-Baptiste/roberta-large-ner-english --student distilroberta-base \
        --train evaluation/data/train_mix.jsonl --dev evaluation/data/dev_mix.jsonl --out out/student-a --epochs 4 --seed 0

What it does
  1. Teacher labelling: the teacher reads every training sentence in three versions (original, lower-case, upper-case) and its
     per-word TYPE scores (O / PER / ORG / LOC / MISC) are stored. Case variants are how the student learns to cope with
     tweets typed in lower case. The teacher may use plain types (roberta-large) or B-/I- labels; B-/I- scores are added up.
  2. Student training: for every sentence one of the three versions is picked at random each epoch (--augment). The loss is
         alpha * CE(gold BIO labels)  +  (1 - alpha) * T^2 * KL(teacher types || student types)
     The student has 9 BIO labels; its type probability is the sum of its B- and I- probabilities, softened with temperature T.
     Boundaries (B/I) are learned only from the human labels, because the teacher does not predict them.
  3. Every half epoch the student is scored on the dev set (strict entity F1 on PER/ORG/LOC); the best checkpoint is kept.
Test sets are never read here. alpha = 1 skips the teacher (plain fine-tuning, used as an ablation).
Needs: torch, transformers, numpy (no seqeval). Runs on a GPU (fast) or a CPU (slow).
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

LABELS = ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]     # same order as evaluation/label_maps.py
TYPES = ["O", "PER", "ORG", "LOC", "MISC"]
STUDENT_GROUPS = [[0], [3, 4], [5, 6], [7, 8], [1, 2]]                                       # label ids of each type, in TYPES order
KINDS = ("orig", "lower", "upper")
KEEP = {"PER", "ORG", "LOC"}


# ── data ─────────────────────────────────────────────────────────────────
def load_rows(path, sources=None, limit=None):
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    if sources and sources != ["all"]:
        rows = [r for r in rows if r["source"] in sources]
    return rows[:limit] if limit else rows


def variant(tokens, kind):
    return tokens if kind == "orig" else [t.lower() for t in tokens] if kind == "lower" else [t.upper() for t in tokens]


def load_tokenizer(name):
    return AutoTokenizer.from_pretrained(name, add_prefix_space=True)       # roberta-style tokenizers need it for pre-split words


def encode(tok, token_lists, max_len):
    """Sub-word ids per sentence, and for each word the position of its FIRST sub-word (-1 if it was cut off)."""
    enc = tok(token_lists, is_split_into_words=True, truncation=True, max_length=max_len, padding=False)
    first = []
    for i, words in enumerate(token_lists):
        pos, prev = [-1] * len(words), None
        for j, w in enumerate(enc.word_ids(i)):
            if w is not None and w != prev:
                pos[w] = j
            prev = w
        first.append(pos)
    return enc["input_ids"], first


def pad(tok, id_lists):
    batch = tok.pad({"input_ids": id_lists}, return_tensors="pt")
    return batch["input_ids"], batch["attention_mask"]


# ── teacher ──────────────────────────────────────────────────────────────
def type_groups(id2label):
    """Teacher label ids for each of TYPES (B-PER and I-PER, or just PER)."""
    groups = [[] for _ in TYPES]
    for i, name in id2label.items():
        t = "O" if name == "O" else name.split("-", 1)[-1]
        if t not in TYPES:
            raise ValueError(f"teacher label {name!r} is not one of {TYPES}")
        groups[TYPES.index(t)].append(int(i))
    if any(not g for g in groups):
        raise ValueError(f"teacher is missing a type: {id2label}")
    return groups


@torch.inference_mode()
def label_with_teacher(name, rows, device, max_len, batch_size=64):
    """{kind: [array(n_words, 5) of teacher type logits, NaN for words cut off]} for the three case variants."""
    tok = load_tokenizer(name)
    model = AutoModelForTokenClassification.from_pretrained(name).to(device).eval()
    if device == "cuda":
        model.half()
    groups = type_groups(model.config.id2label)
    out = {k: [None] * len(rows) for k in KINDS}
    for kind in KINDS:
        lists = [variant(r["tokens"], kind) for r in rows]
        ids, first = encode(tok, lists, max_len)
        order = sorted(range(len(rows)), key=lambda i: len(ids[i]))
        t0 = time.time()
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            x, mask = pad(tok, [ids[i] for i in idx])
            logits = model(input_ids=x.to(device), attention_mask=mask.to(device)).logits.float().cpu()
            typ = torch.stack([logits[..., g[0]] if len(g) == 1 else torch.logsumexp(logits[..., g], -1) for g in groups], -1)
            for b, i in enumerate(idx):
                arr = np.full((len(lists[i]), len(TYPES)), np.nan, dtype=np.float32)
                for w, p in enumerate(first[i]):
                    if p >= 0:
                        arr[w] = typ[b, p].numpy()
                out[kind][i] = arr
            if (s // batch_size) % 50 == 0:
                print(f"  teacher {kind}: {min(s + batch_size, len(order))}/{len(order)} ({time.time() - t0:.0f}s)", flush=True)
    del model
    return out


# ── loss and scoring ─────────────────────────────────────────────────────
def distill_loss(logits, gold, teacher, tmask, alpha, temp):
    """alpha * CE(gold) + (1 - alpha) * T^2 * KL(teacher types || student types)."""
    ce = (F.cross_entropy(logits.reshape(-1, logits.size(-1)), gold.reshape(-1), ignore_index=-100)
          if (gold != -100).any() else logits.sum() * 0.0)           # a batch with no labelled word must not give NaN
    if alpha >= 1.0 or tmask.sum() == 0:
        return ce
    z = logits.float() / temp
    log_all = torch.logsumexp(z, -1, keepdim=True)
    student_lp = torch.stack([torch.logsumexp(z[..., g], -1) for g in STUDENT_GROUPS], -1) - log_all       # (B, L, 5)
    pt = F.softmax(teacher.float() / temp, -1)
    kl = (pt * (torch.log(pt.clamp_min(1e-9)) - student_lp)).sum(-1)
    kl = (kl * tmask).sum() / tmask.sum() * temp * temp
    return alpha * ce + (1 - alpha) * kl


def spans(tags):
    """Strict entity spans (type, first word, last word) from BIO label ids; PER / ORG / LOC only."""
    out, cur = set(), None
    for i, t in enumerate(tags):
        lab = LABELS[t]
        if lab == "O":
            if cur:
                out.add(tuple(cur))
            cur = None
        elif lab.startswith("B-") or cur is None or cur[0] != lab[2:]:
            if cur:
                out.add(tuple(cur))
            cur = [lab[2:], i, i]
        else:
            cur[2] = i
    if cur:
        out.add(tuple(cur))
    return {s for s in out if s[0] in KEEP}


def f1_from_counts(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


@torch.inference_mode()
def evaluate(model, tok, rows, device, max_len, batch_size=64):
    """Strict entity F1 of the student on a dev set, overall and per source."""
    model.eval()
    ids, first = encode(tok, [r["tokens"] for r in rows], max_len)
    counts = {}
    order = sorted(range(len(rows)), key=lambda i: len(ids[i]))
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        x, mask = pad(tok, [ids[i] for i in idx])
        pred = model(input_ids=x.to(device), attention_mask=mask.to(device)).logits.argmax(-1).cpu()
        for b, i in enumerate(idx):
            words = [pred[b, p].item() if p >= 0 else 0 for p in first[i]]
            gold, mine = spans(rows[i]["tags"]), spans(words)
            c = counts.setdefault(rows[i]["source"], [0, 0, 0])
            c[0] += len(gold & mine)
            c[1] += len(mine - gold)
            c[2] += len(gold - mine)
    per = {k: f1_from_counts(*v) for k, v in counts.items()}
    return per, float(np.mean(list(per.values()))) if per else 0.0


# ── training ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", default="Jean-Baptiste/roberta-large-ner-english")
    ap.add_argument("--student", default="distilroberta-base")
    ap.add_argument("--train", default="evaluation/data/train_mix.jsonl")
    ap.add_argument("--dev", default="evaluation/data/dev_mix.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sources", default="all", help="comma-separated sources to train on, e.g. conll2003 (zero-shot variant) or all")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--alpha", type=float, default=0.5, help="weight of the gold-label loss; 1.0 = no teacher")
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--no-augment", action="store_true", help="always use the original casing")
    ap.add_argument("--limit", type=int, help="use only the first N training sentences (smoke tests)")
    ap.add_argument("--dev-limit", type=int)
    ap.add_argument("--evals-per-epoch", type=int, default=2)
    ap.add_argument("--fp32", action="store_true", help="disable mixed precision on GPU")
    ap.add_argument("--teacher-cache", help="file to save/load the teacher's scores, so seeds and students reuse them")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""), flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.train, args.sources.split(","), args.limit)
    dev = load_rows(args.dev, None, args.dev_limit)
    print(f"training sentences: {len(rows)} ({args.sources}); dev sentences: {len(dev)}", flush=True)

    teacher = None
    if args.alpha < 1.0:
        cache = Path(args.teacher_cache) if args.teacher_cache else None
        if cache and cache.exists():
            teacher = torch.load(cache, weights_only=False)
            if len(teacher["orig"]) != len(rows):
                raise SystemExit(f"{cache} was made for {len(teacher['orig'])} sentences, not {len(rows)}; delete it")
            print(f"teacher scores loaded from {cache}", flush=True)
        else:
            t0 = time.time()
            teacher = label_with_teacher(args.teacher, rows, device, args.max_len)
            print(f"teacher labelling done in {time.time() - t0:.0f}s", flush=True)
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                torch.save(teacher, cache)

    tok = load_tokenizer(args.student)
    model = AutoModelForTokenClassification.from_pretrained(
        args.student, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)}).to(device)
    enc = {k: encode(tok, [variant(r["tokens"], k) for r in rows], args.max_len) for k in KINDS}

    steps_per_epoch = (len(rows) + args.batch - 1) // args.batch
    total = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    amp = device == "cuda" and not args.fp32
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    eval_every = max(1, steps_per_epoch // args.evals_per_epoch)
    log, best, step = [], -1.0, 0
    rng = random.Random(args.seed)
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        order = list(range(len(rows)))
        rng.shuffle(order)
        for s in range(0, len(order), args.batch):
            idx = order[s:s + args.batch]
            kinds = [("orig" if args.no_augment else rng.choices(KINDS, weights=[0.65, 0.30, 0.05])[0]) for _ in idx]
            x, mask = pad(tok, [enc[k][0][i] for i, k in zip(idx, kinds)])
            gold = torch.full(x.shape, -100, dtype=torch.long)
            tvec = torch.zeros(*x.shape, len(TYPES))
            tmask = torch.zeros(x.shape)
            for b, (i, k) in enumerate(zip(idx, kinds)):
                for w, p in enumerate(enc[k][1][i]):
                    if p < 0:
                        continue
                    gold[b, p] = rows[i]["tags"][w]
                    if teacher is not None and not np.isnan(teacher[k][i][w]).any():
                        tvec[b, p] = torch.from_numpy(teacher[k][i][w])
                        tmask[b, p] = 1.0
            with torch.autocast(device_type="cuda", enabled=amp):
                logits = model(input_ids=x.to(device), attention_mask=mask.to(device)).logits
            loss = distill_loss(logits.float(), gold.to(device), tvec.to(device), tmask.to(device), args.alpha, args.temp)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 20 == 0:
                print(f"epoch {epoch + 1} step {step}/{total} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
            if step % eval_every == 0 or step == total:
                per, mean = evaluate(model, tok, dev, device, args.max_len)
                log.append({"step": step, "epoch": step / steps_per_epoch, "loss": loss.item(), "dev_f1": per, "dev_mean": mean})
                print(f"  dev F1 {mean:.4f} {({k: round(v, 3) for k, v in per.items()})}" + ("  <- best" if mean > best else ""), flush=True)
                if mean > best:
                    best = mean
                    model.save_pretrained(out / "best")
                    tok.save_pretrained(out / "best")
                model.train()

    meta = {"args": vars(args), "best_dev_mean_f1": best, "train_sentences": len(rows), "seconds": time.time() - t0, "log": log}
    (out / "training_log.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"done: best dev mean F1 {best:.4f}; model in {out / 'best'}")


if __name__ == "__main__":
    main()
