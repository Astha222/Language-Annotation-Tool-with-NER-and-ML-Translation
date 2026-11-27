# Model card: English NER student A (distilled DistilRoBERTa)

Status: **experimental, not the app default.** Files are not published or bundled; they live on the developer's machine (`ws2_results/A_distilroberta_distill/best`, 316 MB;
ONNX fp32 326 MB, ONNX INT8 82 MB).

| | |
|---|---|
| Task | English named-entity recognition: PER, ORG, LOC (and MISC internally) |
| Architecture | `distilroberta-base` (82 M parameters, 6 layers; Apache-2.0), token classification head with 9 BIO labels |
| Teacher | `Jean-Baptiste/roberta-large-ner-english` (355 M parameters; MIT) |
| Training data | train splits of CoNLL-2003 (12,589 sentences after cleaning), BTC (6,312), WNUT-2017 (2,296), FIN (978); sentences that also occur in any reported test set removed (326 CoNLL, 10 BTC, 11 FIN) |
| Method | distillation (`training/distill.py`): 0.5 x cross-entropy on human BIO labels + 0.5 x T^2 x KL to the teacher's per-word type probabilities (T = 2); case augmentation (original 65%, lower-case 30%, upper-case 5%, teacher scores the augmented text); 4 epochs, batch 32, lr 5e-5, AdamW, seed 0, one Colab T4 session (all four experiments took 22 minutes) |
| Scores (strict F1, full test sets) | WNUT-2017 0.558 [0.526, 0.587]; BTC 0.630 [0.617, 0.644]; FIN 0.296 [0.223, 0.366] |
| For comparison | bert-base-NER 0.493 / 0.401 / 0.175; teacher 0.554 / 0.581 / 0.222 |
| Speed (CPU, 4 threads, ~2,000 characters) | PyTorch p95 956 ms; ONNX fp32 p95 670 ms (today's default bert-base: 2,003 ms) |

## Correction (2026-11-18) - read this first
The caveat below that "41.7% of BTC and 73.6% of FIN test entities also occur in the training data" counted a bare `@` and the contract words "Borrower" and "Lender" as seen names. Excluding such annotation artefacts, the seen share of real name entities is 14.8% (BTC)
and 28.7% (FIN, only 115 names). On names only (artefacts removed from gold and predictions), student A scores WNUT-2017 0.565, BTC 0.759, FIN 0.378; the same-data model trained without a teacher (B) scores 0.595 / 0.787 / 0.476, i.e. A is **not** the best use of this training data.
See `evaluation/RESULTS.md`, section "E1".

## English-EWT result (2026-11-18) - read with the correction above
On UNER English-EWT (2,077 sentences of emails, reviews and web text that no model was trained on), student A scores 0.714 [0.682, 0.740]: not reliably better than bert-base-NER
(0.703; difference +0.011 [-0.015, +0.034]) and below student B (0.771) and the teacher (0.793). See `evaluation/RESULTS.md`, section "E5".

## Read before use
- **In-domain caveat.** The model saw tweets, WNUT-style text and finance contracts in training. 41.7% of BTC and 73.6% of FIN test entities also occur in the training data, so those two scores are partly
  memorisation. The fairest test is WNUT-2017 (2.3% overlap): +0.065 [+0.033, +0.096] over bert-base-NER, level with the teacher.
- **A zero-shot version exists** (experiment C, trained on CoNLL news only): 0.523 / 0.565 / 0.206.
- **Not measured:** emails, reports, legal or medical text, non-English text, or any text unlike the three domains above. The in-house gold set planned in the improvement plan (section 8.3) has not been built.
- Entity types: PER, ORG, LOC only are evaluated. Dates and other types are not predicted.
- Strict matching: a prediction counts only with the exact type and boundaries.

## Licences and provenance to settle before publishing the weights
- distilroberta-base: Apache-2.0. Teacher: MIT.
- CoNLL-2003: annotations are free to use, the underlying Reuters text is copyrighted (the usual accepted practice for CoNLL-trained models).
- WNUT-2017 and the Broad Twitter Corpus: CC-BY-4.0 (attribution needed). FIN: MIT on the tner dataset card, original licence unverified.
- Hosting: a decision for the project owner and mentor.

## Reproduce
`python training/build_mix.py` and `--dev`, then the Colab notebook `WS2_Colab/WS2_Distillation_Colab.ipynb` (experiment A), then
`python evaluation/run_all.py --models "A=<folder>/best" --datasets wnut2017,btc,fin --full`. Overlap check: `python evaluation/overlap_check.py`.
