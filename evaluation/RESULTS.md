# Measured results

Recorded 2026-10-30 with the scripts in this folder. Read the caveats at the bottom before quoting any number.

## Translation, English -> X (WMT24++, 40 segments per language, chrF++; higher is better)

| language | offline (local models) | Azure AI Translator | gap (Azure - local) |
|---|---|---|---|
| Hindi (hi) | 35.2 | 36.0 | 0.8 (within noise) |
| French (fr) | 53.7 | 58.6 | 4.9 |
| German (de) | 53.9 | 61.6 | 7.7 |
| Spanish (es) | 62.3 | 65.5 | 3.2 |
| Russian (ru) | 35.8 | 47.6 | 11.8 |
| Chinese (zh) | 17.8 | 32.0 | 14.2 |
| Arabic (ar) | 32.1 | 38.6 | 6.5 |
| Japanese (ja) | 4.4 -> **17.9** (WS1, new model) | 38.3 | 33.9 -> 20.4 |

* No segment failed for any provider (0/40 everywhere).
* **Offline Japanese was unusable (4.4) because the model was wrong** (`opus-mt-en-jap` is not Japanese and was only evaluated on
  Bible verses). After WS1 it uses `opus-tatoeba-en-ja`: 17.9 chrF++ (see the WS1 section). Still weak; use Azure for Japanese.
  Offline Chinese is weak (many names and slang are mangled).
* Offline Hindi (IndicTrans2) is on par with Azure on this sample; the other offline languages (Marian) trail Azure.
* Offline speed measured on a laptop CPU: roughly 22-37 characters per second for zh/ar/ja.
* Raw tables: Appendix A.2 (offline zh/ar/ja). The other rows come from `translation_run.log`
  (an out-of-disk error stopped the run before it wrote its table, and the offline zh/ar/ja rows in that log are invalid
  for the same reason; they were re-run).

## Entity recognition, English (precision / recall / F1, exact match on PER, ORG, LOC)

F1 per test set (500 sentences; FIN has 305):

| test set | bert-base-NER (app default) | bert-large-NER | roberta-large-ner-english |
|---|---|---|---|
| WikiANN (Wikipedia) | 0.557 | 0.594 | 0.610 |
| WNUT-2017 (noisy social media), first 500 sentences | 0.592 (full set: **0.493**) | 0.617 | 0.657 |
| BTC (Twitter) | 0.396 | 0.423 | 0.582 |
| FIN (financial agreements) | 0.175 | 0.217 | 0.222 |
| CoNLL-2003 (news) - in-domain, do not quote | 0.787 | 0.875 | 0.975 |

* All three models find fewer than half of the names in informal text (BTC recall 0.29 to 0.48) and under a quarter in
  financial text (FIN recall 0.18 to 0.21).
* The confidence-threshold sweep (Appendix A.1) did not help on BTC and gained at most about 0.02 F1 on FIN, so no
  threshold is applied by default.
* Not yet measured: NER for Hindi and the other non-English languages.

### Full-split results from the WS0 harness (2026-11-16, `run_all.py`, 95% bootstrap intervals)

The table above uses the first 500 sentences of each set. The harness scores every sentence of each test split and
shows that the first 500 WNUT-2017 sentences were easier than the whole set, so the WNUT-2017 numbers above (all three
models) are too high; only bert-base-NER has been re-measured on the full set so far.

| test set | model | sentences | strict F1 [95% CI] | partial F1 | PER / ORG / LOC F1 |
|---|---|---|---|---|---|
| WNUT-2017 | bert-base-NER | 1,287 | 0.493 [0.459, 0.525] | 0.535 | 0.614 / 0.305 / 0.531 |
| BTC | bert-base-NER | 2,000 | 0.401 [0.382, 0.422] | 0.430 | 0.342 / 0.380 / 0.610 |
| FIN | bert-base-NER | 305 | 0.175 [0.127, 0.225] | 0.270 | 0.186 / 0.127 / 0.327 |

BTC, first 500 sentences: roberta-large-ner-english is better than bert-base-NER by +0.186 F1, 95% CI [+0.156, +0.219],
permutation p = 0.0002.

Error types of bert-base-NER on the full sets (share of gold entities): BTC 29% correct, 61% missed, 8% wrong type,
2% wrong boundary; WNUT-2017 49% / 37% / 10% / 4%; FIN 21% / 26% / **40%** / 13%. On financial text the main problem is
the wrong type (and 190 spurious predictions), not only missed entities.

### Speed, current NER path (PyTorch fp32, 4 threads, AMD Ryzen 3 5300U, 7.9 GB RAM, 30 timed calls, 1 repeat)

| model | input | batch | p50 | p95 | chars/s | peak RAM |
|---|---|---|---|---|---|---|
| bert-base-NER (0.43 GB, cold start 15 s) | ~200 chars | 1 | 230 ms | 378 ms | 1,076 | 0.88 GB |
| bert-base-NER | ~2,000 chars | 1 | 1,437 ms | 1,858 ms | 1,364 | 0.91 GB |
| roberta-large-ner-english (1.42 GB, cold start 17.5 s) | ~200 chars | 1 | 906 ms | 1,643 ms | 262 | 1.75 GB |
| roberta-large-ner-english | ~2,000 chars | 1 | 4,776 ms | 6,371 ms | 405 | 1.46 GB |

roberta-large is about 3.3 to 3.5 times slower than bert-base on medium text. Batch size 32 made no difference for
texts of at most three chunks. Plot: `evaluation/pareto.png`.

## WS1 (quick wins), 2026-11-16, same laptop (AMD Ryzen 3 5300U, 4 cores / 8 threads, 7.9 GB RAM, CPU only)

### Translation, same 40 WMT24++ segments as above (chrF++ / BLEU; +-2 is noise at this size)

| language | before | after | change made |
|---|---|---|---|
| Japanese | 4.4 / 2.6 | **17.9 / 19.9** | `opus-mt-en-jap` -> `opus-tatoeba-en-ja` |
| Japanese (opt-in, CC-BY-SA) | | 16.4 / 20.6 | `staka/fugumt-en-ja`; 1 of 40 outputs suspect against 5 of 40 for Tatoeba |
| Chinese | 17.8 / 22.5 | 16.4 / 20.5 | target token `>>cmn_Hans<<` added: no measurable change (within noise) |
| Arabic | 32.1 / 7.9 | 30.4 / 8.3 | target token `>>ara<<` added: no measurable change (within noise) |

`facebook/m2m100_418M` was not evaluated: the download alone filled 3.6 GB of the model cache. The plan's target of 25 chrF++ for offline
Japanese was not reached.

### English NER, bert-base-NER, ONNX Runtime against PyTorch (full test sets, strict F1 [95% CI])

| test set | PyTorch fp32 | ONNX fp32 | ONNX INT8 |
|---|---|---|---|
| WNUT-2017 (1,287) | 0.493 [0.459, 0.525] | 0.493 (identical predictions) | 0.494 (no reliable difference) |
| BTC (2,000) | 0.401 [0.382, 0.422] | 0.401 (identical predictions) | **0.389, -0.012 [-0.018, -0.006]** |
| FIN (305) | 0.175 [0.127, 0.225] | 0.175 (identical predictions) | 0.175 (no reliable difference) |

INT8 fails the plan's gate (no more than 0.01 F1 lost on any set) on BTC. ONNX fp32 changes nothing.

| engine (batch 1, 4 threads, 30 timed calls, 1 repeat) | ~200 chars p50 | ~2,000 chars p50 | ~2,000 chars p95 | peak RAM | cold start |
|---|---|---|---|---|---|
| PyTorch fp32 | 230 ms | 1,437-1,561 ms | 1,858-2,003 ms | 0.90 GB | 15.3 s |
| ONNX fp32 | 124 ms | 861 ms | 1,274 ms | 0.99 GB | 15.2 s |
| ONNX INT8 | 124 ms | 854 ms | 1,206 ms | 0.67 GB | 10.1 s |

ONNX files: fp32 431 MB, INT8 108 MB. On this CPU INT8 is no faster than fp32 in the full pipeline (it lacks INT8 instructions); it only
saves memory and costs accuracy. PyTorch's own dynamic INT8 was 1.32x faster and lost entities, and was not used.

### Other WS1 measurements
* First request after startup: **17.36 s -> 0.09 s** with `PRELOAD_MODELS=ner_en` (the 15.2 s moves to startup).
* CPU threads, medium text p50 (3 alternating rounds): 4 threads 1,668-1,784 ms, 8 threads 1,383-1,420 ms. But NER and translation
  together (a document job): 8 threads 26.4 s against 4 threads 20.6 s (translation alone was 34% slower with 8). Default stays at the
  physical core count.
* NER chunking on long documents (documents made of 15 consecutive test sentences, F1): 800-char chunks FIN 0.178, WNUT 0.542,
  BTC 0.307; 64-token overlapping windows 0.161, 0.510, 0.279 (worse and up to 25% slower); 300-char chunks 0.185, 0.556, 0.303 (noise).
  Defaults unchanged. The documents are unrelated sentences joined together, which is not realistic.
* Batching (20,000-character document): PyTorch batch 1 15.3 s, batch 8 17.8 s; ONNX fp32 batch 1 9.9 s, batch 8 14.6 s. Sorting chunks by
  length did not help. Default batch size is now 1.
* Result cache: the same 20,000-character document again 16.0 s -> 57 ms; the same 14 sentences translated again 28.6 s -> 3 ms.

## WS2 (English NER: distilled students), 2026-11-16

Strict entity F1 [95% bootstrap CI] on the **full** test sets. Students were trained on a Colab T4 (`training/distill.py`, 22,175 training sentences, 4 epochs, seed 0; the
four experiments took 22 minutes in total) and scored on the laptop with `evaluation/run_all.py`.

| model | WNUT-2017 (1,287) | BTC (2,000) | FIN (305) |
|---|---|---|---|
| bert-base-NER (current default, zero-shot) | 0.493 [0.459, 0.525] | 0.401 [0.382, 0.422] | 0.175 [0.127, 0.225] |
| distilbert-NER (zero-shot) | 0.427 [0.396, 0.459] | 0.337 [0.318, 0.358] | 0.182 [0.134, 0.234] |
| roberta-large-ner-english = the teacher (zero-shot) | 0.554 [0.522, 0.584] | 0.581 [0.566, 0.596] | 0.222 [0.139, 0.295] |
| **A** distilroberta, distilled from the teacher, trained on CoNLL + WNUT + BTC + FIN (in-domain) | **0.558** [0.526, 0.587] | 0.630 [0.617, 0.644] | 0.296 [0.223, 0.366] |
| **B** distilroberta, same data, **no teacher** (ablation, in-domain) | 0.490 [0.461, 0.518] | 0.808 [0.793, 0.822] | 0.759 [0.686, 0.828] |
| **C** distilroberta, distilled, trained on **CoNLL news only** (zero-shot control) | 0.523 [0.493, 0.554] | 0.565 [0.550, 0.580] | 0.206 [0.138, 0.272] |
| **D** deberta-v3-small, distilled, in-domain | 0.549 [0.521, 0.580] | not run | 0.342 [0.268, 0.413] |

Differences against bert-base-NER (paired bootstrap): A: WNUT +0.065 [+0.033, +0.096], BTC +0.230 [+0.212, +0.244], FIN +0.121 [+0.062, +0.178].
C: WNUT +0.031 [0.000, +0.062], BTC +0.164 [+0.146, +0.180], FIN +0.030 [-0.024, +0.080] (not reliable).
Against the teacher: A is level on WNUT (+0.004, n.s.), better on BTC (+0.049) and FIN (+0.074); C is worse on WNUT (-0.031) and BTC (-0.017), level on FIN.

**How to read it.** A, B and D were trained on the training splits of WNUT, BTC and FIN, so those sets are in-domain for them. The share of test entities that also occur as
entities in the training data (`evaluation/overlap_check.py`) is **2.3% for WNUT-2017, 41.7% for BTC and 73.6% for FIN** (91% of FIN person-type names): scores on BTC and FIN,
especially B's 0.81 and 0.76, largely reflect memorised names and phrasing and say little about general quality. **WNUT-2017 (almost all new names) and the zero-shot model C
are the fair tests.** On WNUT, the teacher-distilled students (A 0.558, D 0.549) match the teacher (0.554), while the student trained without the teacher (B 0.490) is no better than
bert-base (0.493): the teacher is what transfers the ability to recognise unseen names, although B scores highest on the dev set, which resembles its training data.

Speed (PyTorch fp32 unless stated; batch 1, 4 threads, ~2,000 characters, 30 calls, 1 repeat; no result cache):

| model | p50 | p95 | chars/s | peak RAM | cold start |
|---|---|---|---|---|---|
| bert-base-NER, PyTorch (today's default) | 1,561 ms | 2,003 ms | 1,292 | 0.90 GB | 11.2 s |
| bert-base-NER, ONNX fp32 | 861 ms | 1,274 ms | 2,177 | 0.99 GB | 15.2 s |
| student A (distilroberta), PyTorch | 765 ms | 956 ms | 2,610 | 0.74 GB | 17.0 s |
| **student A, ONNX fp32** (identical predictions to PyTorch on WNUT and FIN) | **496 ms** | **670 ms** | **3,904** | 0.88 GB | 11.6 s |
| roberta-large (teacher), PyTorch | 4,776 ms | 6,371 ms | 405 | 1.46 GB | 17.5 s |

The student plus ONNX is 3.0x faster than today's default at p95 and 9.5x faster than the teacher. Pareto plot on WNUT-2017: `evaluation/pareto_wnut.png`
(student A with ONNX is the only point on the frontier). Student files: 316 MB (PyTorch), 326 MB (ONNX fp32), 82 MB (ONNX INT8, not evaluated for accuracy).

Plan gate (WS2.7: F1 >= bert-base on every test set with a CI above 0 on BTC and WNUT, and p95 latency <= today's): **A passes** both parts. C (the zero-shot student) is
above bert-base on every set but its WNUT interval starts at exactly 0.000 and FIN is not reliable. Not measured: deberta (D) on BTC and its latency; three seeds; the
news-only student without a teacher (which would separate the effect of the teacher from case augmentation in the zero-shot setting); anything outside tweets, WNUT-style
text and finance text. Dev-set means chosen during training (not test scores): A 0.678, B 0.815, C 0.526, D 0.747.

## WS3 (Hindi and other-language NER), 2026-11-17

Strict entity F1 on PER / ORG / LOC [95% bootstrap CI], run with `evaluation/run_all.py`; test data frozen in `MANIFEST.json`. Hardware as above (CPU only).

### Hindi: three independent test sets

| test set | sentences / entities | previous model `Davlan/xlm-roberta-base-wikiann-ner` | `neuronsbyisshu/indicbert-hi-ner-naamapadam` (CC0, new default) | difference |
|---|---|---|---|---|
| Naamapadam Hindi **gold** test (ai4bharat, CC0) | 864 / 1,929 | 0.650 [0.626, 0.675] | **0.827** [0.807, 0.847] | **+0.177** [+0.155, +0.198] |
| HiNER-collapsed test, every 7th sentence (CC-BY-SA-4.0) | 3,094 / 7,840 | 0.527 [0.516, 0.537] | **0.755** [0.746, 0.763] | **+0.228** [+0.218, +0.238] |
| MultiCoNER v2 Hindi dev (short, lower-cased, low context; CC-BY-4.0) | 514 / 417 | 0.421 [0.381, 0.466] | **0.601** [0.555, 0.640] | **+0.179** [+0.134, +0.218] |

Per type (PER / ORG / LOC): Naamapadam old 0.765 / 0.478 / 0.681, new 0.891 / 0.731 / 0.830; HiNER old 0.713 / 0.346 / 0.528, new 0.804 / 0.644 / 0.761; MultiCoNER hi old 0.63 / 0.26 / 0.44, new 0.66 / 0.49 / 0.63.
Organizations are the weakest type for both models. The old model's wrong-boundary errors were 24% of HiNER entities against 6% for the new one; its strict-to-partial F1 gap is 0.215 on HiNER.
Naamapadam is the new model's home dataset (it was fine-tuned on the silver Naamapadam training split; the test split is human-annotated), so HiNER and MultiCoNER are the independent confirmation. The model scored 0.827 on Naamapadam, which equals the figure on its model card.
Speed (CPU, same sentences): Naamapadam 133 s against 223 s, MultiCoNER hi 51 s against 59 s.
Not measured: `ai4bharat/IndicNER` (gated: needs a Hugging Face login and acceptance of its terms; the plan's first choice, MIT, published Hindi F1 82.9), the two `cfilt/HiNER-collapsed-*` models (no licence declared; 0.9 and 2.2 GB), fine-tuning (not needed at this level).

### Other languages: the app's multilingual model on MultiCoNER v2 dev (first measured numbers)

| language | sentences / entities | strict F1 [95% CI] | partial F1 | PER / ORG / LOC |
|---|---|---|---|---|
| German | 512 / 544 | 0.341 [0.301, 0.378] | 0.502 | 0.51 / 0.21 / 0.30 |
| Spanish | 854 / 837 | 0.310 [0.280, 0.340] | 0.451 | 0.45 / 0.18 / 0.30 |
| French | 857 / 887 | 0.264 [0.238, 0.292] | 0.395 | 0.39 / 0.12 / 0.31 |
| Chinese (after the fix below) | 506 / 483 | 0.281 [0.241, 0.317] | 0.477 | 0.37 / 0.17 / 0.33 |

MultiCoNER v2 is deliberately hard (all text lower-cased and without punctuation, short sentences with little context, 33 fine-grained types mapped to PER / ORG / LOC), so these numbers are a stress test and **underestimate performance on ordinary text**; they are the only multilingual numbers we have. No suitable test set was found for Russian, Arabic or Japanese (MultiCoNER v2 does not cover them); no alternative model was evaluated for any of these languages (the plan's `universalner/uner_all` is 4.4 GB; GLiNER-based candidates were excluded on the owner's instruction), so the unlicensed Davlan model remains in use for them.

### A bug found and fixed: Chinese (and Japanese) entities covered whole clauses
On MultiCoNER Chinese the previous code scored **strict F1 0.000** (partial 0.165). The model located names correctly, but the entity then ran to the end of the clause: Chinese has no spaces, so the tokenizer treats the
whole sentence as one "word" and aggregation `first` gives it one label, and `_snap_to_token` then grew every entity to the nearest space or punctuation mark. Using aggregation `simple` for zh and ja raised strict F1 to 0.020 (partial 0.494); additionally switching
off the snapping for those languages gives **0.281** (partial 0.477). Japanese uses the same unspaced script and receives the same fix, but no Japanese NER test set was available at the time (since 2026-11-18: stockmark and UNER Japanese-PUD, see below), so the effect there is unmeasured.

### Hinglish (Hindi in English letters)
Five realistic sentences ("kal Rahul Mumbai ja raha hai", "Priya ne Delhi mein Infosys join kar liya hai", ...): `langdetect` returned Indonesian (3), Swahili (1) and English at 29% confidence (1), **never Hindi**; the same sentences in Devanagari were detected as Hindi with probability 1.00.
Entities were partly found (Delhi, Infosys, Amazon, Flipkart, Modi, Gujarat, Sachin Tendulkar) but "Rahul Mumbai" was merged into one person and "Priya" was missed. This is a sample of five, not a test set. The plan's fix (transliterate to Devanagari with IndicXlit, then use the Hindi model) was not built because
its install risks the environment; a hand-labelled Hinglish test set does not exist.

## Indian languages: open-licence NER candidates (2026-11-17)

Strict entity F1 on PER / ORG / LOC [95% CI], Naamapadam human-annotated test sets (CC0), first 300 sentences per language (Hindi: all 864), run with `evaluation/run_all.py`.
"Old" = `Davlan/xlm-roberta-base-wikiann-ner` (no declared licence, the model used before WS3). Differences from a paired bootstrap on the same sentences.

| language | old (no licence) | `ai4bharat/IndicNER` (MIT) | `indicbert-hi-ner-naamapadam` (CC0, trained on Hindi only) | best open-licence option measured |
|---|---|---|---|---|
| Hindi | 0.650 | 0.704 [0.682, 0.726] | **0.827** [0.807, 0.847] | indicbert-hi |
| Marathi | 0.617 | 0.505 (-0.112 [-0.169, -0.051] vs old) | **0.792** (+0.175 [+0.129, +0.222] vs old; +0.287 [+0.244, +0.336] vs IndicNER) | indicbert-hi (zero-shot transfer from Hindi) |
| Bengali | 0.559 | **0.716** (+0.158 [+0.096, +0.218]) | not run | IndicNER |
| Gujarati | 0.652 | **0.736** (+0.084 [+0.039, +0.132]) | not run | IndicNER |
| Tamil | 0.517 | **0.617** (+0.101 [+0.039, +0.166]) | not run | IndicNER |
| Telugu | 0.629 | 0.623 (-0.006 [-0.066, +0.057], no difference) | not run | IndicNER (same accuracy, open licence) |
| Kannada, Malayalam, Punjabi, Odia, Assamese | 0.635, 0.687, 0.480, 0.140, 0.196 | **not yet measured** | not run | pending |

IndicNER per type on Hindi: PER 0.801, ORG 0.598, LOC 0.667 (strict 0.704, partial 0.846). Its errors are genuine boundary disagreements with the gold labels (it adds words such as "जेल" or "राज्य" to names, or drops a first name), not an offset fault. Its authors report about 0.83 on Hindi,
probably with a different scoring method; the score here is under the project's strict entity match.
indicbert-hi on Marathi: PER 0.89, ORG 0.56, LOC 0.83.
Download note: the first two attempts to fetch IndicNER's 667 MB weights stalled at near 0 MB/s (the Hugging Face "xet" transfer, then a temporary slow route); the third, with `HF_HUB_DISABLE_XET=1`, took 100 s. The laptop was heavily loaded (about 0.4-0.9 GB RAM free, CPU at 100%), so IndicNER ran about ten times slower than the other models (8-9 minutes per 300 sentences);
timings in these runs are not meaningful.

## Indian languages on full test sets, three entity models, on a GPU (Colab T4, 2026-11-17)

Strict entity F1 on PER / ORG / LOC, **all** sentences of each human-annotated Naamapadam test set (CC0) plus two independent Hindi sets; run with the same `evaluation/run_all.py` through `colab_eval_bundle.zip`
(the old model's scores on Colab equal the laptop's to three decimals: Hindi 0.650, HiNER 0.527, MultiCoNER 0.421). Differences are paired bootstrap, 1,000 resamples.

| test set | sentences | **indicbert-hi** (CC0) | IndicNER (MIT) | old model (no licence) | indicbert minus IndicNER [95% CI] | indicbert per type PER / ORG / LOC |
|---|---|---|---|---|---|---|
| Hindi, Naamapadam gold | 864 | **0.827** | 0.704 | 0.650 | +0.124 [+0.103, +0.144] | 0.89 / 0.73 / 0.83 |
| Hindi, HiNER (independent) | 3,094 | **0.755** | 0.597 | 0.527 | +0.158 [+0.149, +0.167] | 0.80 / 0.64 / 0.76 |
| Hindi, MultiCoNER (independent, hard) | 514 | **0.601** | 0.466 | 0.421 | +0.134 [+0.087, +0.180] | 0.66 / 0.49 / 0.63 |
| Marathi | 1,060 | **0.831** | 0.544 | 0.627 | +0.287 [+0.262, +0.311] | 0.90 / 0.67 / 0.86 |
| Bengali | 607 | **0.808** | 0.709 | 0.526 | +0.099 [+0.070, +0.128] | 0.84 / 0.73 / 0.80 |
| Tamil | 758 | **0.750** | 0.608 | 0.501 | +0.142 [+0.113, +0.173] | 0.84 / 0.60 / 0.76 |
| Telugu | 847 | **0.854** | 0.676 | 0.653 | +0.178 [+0.153, +0.206] | 0.90 / 0.75 / 0.85 |
| Gujarati | 1,076 | **0.825** | 0.728 | 0.620 | +0.098 [+0.078, +0.119] | 0.91 / 0.69 / 0.83 |
| Kannada | 1,003 | **0.826** | 0.712 | 0.594 | +0.114 [+0.090, +0.136] | 0.90 / 0.72 / 0.80 |
| Malayalam | 965 | **0.847** | 0.721 | 0.647 | +0.125 [+0.104, +0.146] | 0.91 / 0.65 / 0.88 |
| Punjabi | 993 | **0.732** | 0.587 | 0.461 | +0.145 [+0.120, +0.170] | 0.79 / 0.57 / 0.79 |
| Odia | 994 | 0.167 | **0.271** | 0.127 | -0.103 [-0.146, -0.054] | 0.12 / 0.12 / 0.25 |
| Assamese | 51 | 0.467 | **0.500** | 0.196 | -0.033 [-0.242, +0.141] (no reliable difference) | 0.67 / 0.37 / 0.50 |

**Findings.**
1. `neuronsbyisshu/indicbert-hi-ner-naamapadam` (CC0; IndicBERTv2 fine-tuned on **Hindi only**, 100k silver sentences) is the best open-licence model on Hindi and on **eight other languages** (Marathi, Bengali, Tamil, Telugu, Gujarati, Kannada, Malayalam, Punjabi),
   ahead of IndicNER by +0.10 to +0.29 with every interval clear of zero, and ahead of the unlicensed old model on all of them. The model card reports the same zero-shot numbers for the four languages it lists (Bengali 0.808, Tamil 0.751, Telugu 0.853); this run reproduces them
   (0.808, 0.750, 0.854) and extends the picture to Marathi, Gujarati, Kannada, Malayalam and Punjabi. The transfer works because IndicBERTv2 was pretrained on 24 Indian languages.
2. **Odia is not usable by any of the three models** (best 0.271, IndicNER). **Assamese** is uncertain (51 test sentences; IndicNER 0.500 with a range of 0.29-0.73).
3. **ORG is the weakest type everywhere** (0.57-0.75 for indicbert); PER and LOC are 0.76-0.91.
4. **Caution about what the scores mean.** For languages other than Hindi the only test set is Naamapadam's gold test, which comes from the same data family and labelling guidelines as the model's silver training data. For Hindi, where independent sets exist, the score falls from 0.827 (Naamapadam)
   to 0.755 (HiNER) and 0.601 (MultiCoNER): expect the other languages to lose a similar amount on text from other sources. No independent test set exists for them yet. Silver training labels also limit the model (ORG boundaries are its weakest class, per its card).
5. Offline translation into these languages was not measured here (Azure translated a test sentence into all of them).

## E1: seen / unseen entities, annotation artefacts, and names-only scores (2026-11-18)

Computed from the saved prediction files with `evaluation/seen_unseen.py` (7 unit tests, including a hand-counted 5-sentence fixture). A test entity is *seen* when an entity of the same type and the same lower-cased text occurs in the training data of the model
(bert-base: CoNLL-2003 train; teacher: CoNLL train and test; students A and B: the four-dataset mix; student C: CoNLL only; indicbert-hi: the Naamapadam Hindi training split, all of it, which over-counts relative to the 100k subsample the model saw).

**Annotation artefacts (a correction to earlier analyses).** Two kinds of gold "entities" are labelling conventions, not names: a bare symbol such as `@` (Twitter set: 1,380 of 4,376 gold entities, about 31%) and the contract role words "Borrower" and "Lender" (finance set: 196 of 311 gold entities, 63%).
They are frequent in the training data and rarely tagged by models that did not learn the convention. They are set aside from the seen/unseen split and reported separately below. Excluding them, the share of gold *name* entities seen in training is **WNUT-2017 2.3%, BTC 14.8% (2,995 names), FIN 28.7% (only 115 names)**,
not the 41.7% and 73.6% reported in the WS2 sections of this file and of `IMPROVEMENT_LOG.md` (those figures counted the artefacts as "seen names").

### Seen / unseen split, artefacts set aside (full test sets)

| model | test set | sentences | F1 all | F1 unseen subset [95% CI] | unseen sentences | gold entities seen in training | recall seen | recall unseen |
|---|---|---|---|---|---|---|---|---|
| bert-base-NER (default) | wnut2017 | 1287 | 0.493 | 0.454 [0.417, 0.492] | 1208 | 11% | 0.807 | 0.449 |
| bert-base-NER (default) | btc | 2000 | 0.401 | 0.327 [0.309, 0.346] | 1741 | 10% | 0.866 | 0.380 |
| bert-base-NER (default) | fin | 305 | 0.175 | 0.139 [0.091, 0.190] | 288 | 20% | 0.609 | 0.391 |
| roberta-large (teacher; trained on CoNLL train and test) | wnut2017 | 1287 | 0.554 | 0.517 [0.485, 0.551] | 1199 | 12% | 0.887 | 0.679 |
| roberta-large (teacher; trained on CoNLL train and test) | btc | 2000 | 0.581 | 0.538 [0.522, 0.554] | 1723 | 11% | 0.897 | 0.676 |
| roberta-large (teacher; trained on CoNLL train and test) | fin | 305 | 0.222 | 0.176 [0.078, 0.271] | 288 | 20% | 0.565 | 0.457 |
| student A (CoNLL + WNUT + BTC + FIN) | wnut2017 | 1287 | 0.558 | 0.503 [0.466, 0.537] | 1161 | 17% | 0.857 | 0.664 |
| student A (CoNLL + WNUT + BTC + FIN) | btc | 2000 | 0.630 | 0.585 [0.568, 0.601] | 1566 | 18% | 0.826 | 0.741 |
| student A (CoNLL + WNUT + BTC + FIN) | fin | 305 | 0.296 | 0.222 [0.127, 0.308] | 278 | 37% | 0.500 | 0.479 |
| student B (same data as A, NO teacher) | wnut2017 | 1287 | 0.490 | 0.426 [0.394, 0.458] | 1161 | 17% | 0.871 | 0.657 |
| student B (same data as A, NO teacher) | btc | 2000 | 0.808 | 0.799 [0.779, 0.816] | 1566 | 18% | 0.833 | 0.770 |
| student B (same data as A, NO teacher) | fin | 305 | 0.759 | 0.787 [0.691, 0.864] | 278 | 37% | 0.714 | 0.534 |
| student C (CoNLL only) | wnut2017 | 1287 | 0.523 | 0.485 [0.452, 0.519] | 1208 | 11% | 0.955 | 0.659 |
| student C (CoNLL only) | btc | 2000 | 0.565 | 0.525 [0.509, 0.540] | 1741 | 10% | 0.886 | 0.689 |
| student C (CoNLL only) | fin | 305 | 0.206 | 0.158 [0.080, 0.241] | 288 | 20% | 0.609 | 0.446 |

### Same sentences for every model (sentences with no seen gold name in ANY model's training data), strict F1 [95% CI]

| test set | common unseen sentences | bert-base-NER (default) | roberta-large (teacher; trained on CoNLL train and test) | student A (CoNLL + WNUT + BTC + FIN) | student B (same data as A, NO teacher) | student C (CoNLL only) |
|---|---|---|---|---|---|---|
| wnut2017 | 1154 of 1287 | 0.441 [0.399, 0.482] | 0.499 [0.462, 0.532] | 0.504 [0.467, 0.537] | 0.426 [0.392, 0.459] | 0.463 [0.427, 0.497] |
| btc | 1555 of 2000 | 0.306 [0.286, 0.327] | 0.529 [0.512, 0.544] | 0.584 [0.567, 0.597] | 0.799 [0.781, 0.817] | 0.516 [0.500, 0.533] |
| fin | 278 of 305 | 0.107 [0.056, 0.153] | 0.181 [0.070, 0.282] | 0.222 [0.127, 0.308] | 0.787 [0.691, 0.864] | 0.143 [0.058, 0.228] |

### Names only: artefact entities removed from BOTH the gold labels and the predictions, strict F1 [95% CI]

| model | test set | F1 all (standard) | F1 names only [95% CI] | change |
|---|---|---|---|---|
| bert-base-NER (default) | wnut2017 | 0.493 | 0.493 [0.459, 0.525] | +0.000 |
| bert-base-NER (default) | btc | 0.401 | 0.511 [0.490, 0.531] | +0.110 |
| bert-base-NER (default) | fin | 0.175 | 0.244 [0.182, 0.299] | +0.069 |
| roberta-large (teacher) | wnut2017 | 0.554 | 0.554 [0.522, 0.584] | +0.000 |
| roberta-large (teacher) | btc | 0.581 | 0.719 [0.702, 0.735] | +0.138 |
| roberta-large (teacher) | fin | 0.222 | 0.369 [0.270, 0.466] | +0.147 |
| student A (in-domain, with teacher) | wnut2017 | 0.558 | 0.565 [0.533, 0.594] | +0.008 |
| student A (in-domain, with teacher) | btc | 0.630 | 0.759 [0.744, 0.774] | +0.129 |
| student A (in-domain, with teacher) | fin | 0.296 | 0.378 [0.282, 0.471] | +0.082 |
| student B (in-domain, NO teacher) | wnut2017 | 0.490 | 0.595 [0.564, 0.625] | +0.105 |
| student B (in-domain, NO teacher) | btc | 0.808 | 0.787 [0.772, 0.803] | -0.021 |
| student B (in-domain, NO teacher) | fin | 0.759 | 0.476 [0.368, 0.571] | -0.283 |
| student C (CoNLL only) | wnut2017 | 0.523 | 0.524 [0.494, 0.554] | +0.001 |
| student C (CoNLL only) | btc | 0.565 | 0.691 [0.673, 0.709] | +0.126 |
| student C (CoNLL only) | fin | 0.206 | 0.342 [0.252, 0.430] | +0.136 |

**What this changes.**
1. **Student B's headline scores were mostly labelling conventions.** On finance its F1 falls from 0.759 to 0.476 (-0.283) once "Borrower" and "Lender" are set aside; on Twitter 0.808 to 0.787. In-domain training teaches a dataset's conventions (tag `@`, tag "Borrower"), which earns large credit on that dataset and costs points elsewhere:
   B's low WNUT score (0.490) came from predicting `@` as an entity, which WNUT does not label (names only: 0.595).
2. **The WS2 statement "the teacher is what generalises" is not supported.** On names only, B (no teacher) is the best model on all three sets (WNUT 0.595, BTC 0.787, FIN 0.476), ahead of student A (0.565, 0.759, 0.378), the teacher (0.554, 0.719, 0.369), student C (0.524, 0.691, 0.342) and bert-base (0.493, 0.511, 0.244).
   Distillation did not beat plain fine-tuning on the same data in this setting; A and the teacher are close. What remains true: every model trained on CoNLL with case augmentation (student C, the teacher) beats bert-base-NER clearly on Twitter and finance names.
3. **Memorisation is a smaller factor than reported for English**: 10-20% of gold names are seen by the zero-shot models and 18% (BTC) / 37% (FIN) by the in-domain students; the unseen-subset F1 is about 0.04-0.07 below the all-sentence F1 for the zero-shot models.
   On the common unseen sentences student B remains far ahead on BTC and FIN (0.799, 0.787) but those sentences still contain artefact entities; the names-only table is the cleaner comparison.
4. The WS2 gate decision for student A stands (it beats bert-base on every set), but A is not the best use of the same data: a plain fine-tune (B) is better on names. Open question for a later step: re-run B with 3 seeds and case augmentation before choosing a student.

### Indian languages: seen / unseen (Naamapadam Hindi training split as "seen"; 100% of that split, so conservative)

| run | sentences | F1 all | F1 unseen subset [95% CI] | unseen sentences | gold entities seen | recall seen | recall unseen |
|---|---|---|---|---|---|---|---|
| Hindi, Naamapadam gold | 864 | 0.827 | 0.563 [0.461, 0.658] | 191 | 78% | 0.889 | 0.768 |
| Hindi, HiNER | 3,094 | 0.755 | 0.480 [0.424, 0.531] | 900 | 78% | 0.689 | 0.688 |
| Marathi (shares the Devanagari script) | 1,060 | 0.831 | 0.751 [0.708, 0.792] | 615 | 46% | 0.888 | 0.793 |
| Bengali | 607 | 0.808 | 0.809 [0.779, 0.836] | 603 | 0% | 0.750 | 0.813 |
| Tamil | 758 | 0.750 | 0.750 [0.722, 0.776] | 755 | 0% | 1.000 | 0.790 |
| Telugu | 847 | 0.854 | 0.855 [0.833, 0.877] | 844 | 0% | 1.000 | 0.879 |
| Gujarati | 1,076 | 0.825 | 0.823 [0.802, 0.842] | 1,057 | 2% | 0.893 | 0.855 |
| Kannada | 1,003 | 0.826 | 0.825 [0.802, 0.847] | 998 | 0% | 0.800 | 0.852 |
| Malayalam | 965 | 0.847 | 0.847 [0.825, 0.867] | 965 | 0% | - | 0.871 |
| Punjabi | 993 | 0.732 | 0.731 [0.704, 0.755] | 989 | 0% | 0.750 | 0.740 |

**Reading it.** For Hindi, 78% of the test entities occur in the training data and the score on sentences with only unseen entities is far lower (0.56 on Naamapadam, 0.48 on HiNER): the headline Hindi figures are helped substantially by seen names.
For the seven languages in other scripts, **there is essentially no overlap with the Hindi training entities (0-2%)**, so their scores (0.73-0.85) are scores on unseen names: the cross-lingual transfer is real and not memorisation. Marathi shares many Devanagari names with Hindi (46% seen), and still scores 0.751 on its unseen subset.
What this does not remove: similarity of text domain and labelling guidelines between the Naamapadam training and test sets, and the silver (machine-projected) training labels.

### GLiNER (Apache-2.0, zero-shot, threshold 0.5 untuned; Colab T4; `ws3b_runs`): only `urchade/gliner_multi-v2.1` produced results

| test set | sentences | GLiNER multi | best other open model on the same sentences | old model (no licence) |
|---|---|---|---|---|
| Indian languages (10 Naamapadam sets) | 51-1,076 | 0.031-0.263 (finds 2-21% of entities; best on 51-sentence Assamese, 0.263) | indicbert-hi 0.73-0.85 on nine of them (Odia: IndicNER 0.271; Assamese: IndicNER 0.500) | 0.127-0.653 |
| Hindi HiNER / MultiCoNER | 3,094 / 514 | 0.103 / 0.028 | indicbert-hi 0.755 / 0.601 | 0.527 / 0.421 |
| German (MultiCoNER) | 512 | **0.552** | - | 0.341 |
| Spanish (MultiCoNER) | 854 | **0.647** | - | 0.310 |
| French (MultiCoNER) | 857 | **0.608** | - | 0.264 |
| Chinese (MultiCoNER) | 506 | 0.023 | - | 0.281 (after the zh fix) |

GLiNER-multi is unusable for Indian languages and Chinese (no spaces between words: its word splitter treats a sentence as one word) but is **far ahead of the unlicensed model on German, Spanish and French** (+0.21 to +0.34) and carries an open licence: a candidate replacement for those three languages.
Caveats: MultiCoNER v2 is lower-cased and punctuation-free; GLiNER's training data may overlap with MultiCoNER (not checked); using it in the app would add the `gliner` package and a second model type; the threshold was not tuned. `knowledgator/gliner-x-base` and `-small` did not run in the first attempt (they need `gliner[stanza]` and sentencepiece); see the next section.

### GLiNER-X (knowledgator/gliner-x-base and -small; Apache-2.0; zero-shot; threshold 0.5 untuned; Colab T4; `ws3c_runs`, 2026-11-18)

Strict F1 on full test sets, from the saved prediction files (`ws3c_runs`, `ws3_runs`, `ws3b_runs`), sentence counts as in the tables above. "-" = did not run.
Training data of GLiNER-X per its model card: `knowledgator/gliner-multilingual-synthetic` and `lang-uk/UberText-NER-Silver` (synthetic / Ukrainian), so it did not train on Naamapadam, HiNER or MultiCoNER (not otherwise checked).

| test set | indicbert-hi (Hindi-trained) | IndicNER | GLiNER multi | GLiNER-X base | GLiNER-X small |
|---|---|---|---|---|---|
| Hindi Naamapadam | **0.827** | 0.704 | 0.138 | 0.757 | 0.714 |
| Hindi HiNER | **0.755** | 0.597 | 0.103 | 0.689 | 0.705 |
| Hindi MultiCoNER (lower-case) | **0.601** | 0.466 | 0.028 | 0.479 | 0.427 |
| Bengali | **0.808** | 0.709 | 0.092 | 0.657 | 0.597 |
| Gujarati | **0.825** | 0.728 | 0.071 | 0.756 | 0.694 |
| Kannada | **0.826** | 0.712 | 0.045 | 0.760 | 0.692 |
| Malayalam | **0.847** | 0.721 | 0.097 | 0.793 | 0.741 |
| Telugu | **0.854** | 0.676 | 0.083 | 0.760 | 0.740 |
| Punjabi | **0.732** | 0.587 | 0.046 | 0.595 | 0.568 |
| Marathi | **0.831** | 0.544 | 0.064 | 0.631 | 0.582 |
| Tamil | **0.750** | 0.608 | 0.031 | 0.658 | 0.623 |
| Odia | 0.167 | **0.271** | 0.123 | 0.133 | 0.128 |
| Assamese (51 sentences) | 0.467 | **0.500** | 0.263 | 0.333 | 0.377 |
| German (MultiCoNER) | - | - | 0.552 | **0.592** | 0.513 |
| Spanish | - | - | 0.647 | **0.667** | 0.600 |
| French | - | - | 0.608 | **0.653** | 0.620 |
| Chinese | - | - | 0.023 | **0.266** | 0.173 |

Reading it:
* GLiNER-X is a real step up from GLiNER multi for Indian languages: 0.57-0.79 on seven languages, with no Indian-language training at all. It is still 0.05-0.15 below indicbert-hi (Hindi-trained) on every language it ran on, so **indicbert-hi stays the default for Hindi and the other eight**. GLiNER-X is a second open (Apache-2.0) option that needs no Indian-language training data. It does not solve Odia (0.13) or Assamese (0.33-0.38, 51 sentences): no open model tried is usable for Odia (best 0.271, IndicNER).
* Marathi and Tamil first crashed for both GLiNER-X sizes (`TypeError`). Cause, found with a Colab diagnostic cell: for 113 of 1,060 Marathi and 57 of 758 Tamil sentences GLiNER-X returned an entity with `start`/`end` = `None`, and the adapter's `int(None)` failed. The adapter now recovers the span by finding the entity's own text in the sentence (drops it if not found). Rerun (`ws3d_runs`, notebook WS3d): Marathi 0.631 / 0.582 and Tamil 0.658 / 0.623 (base / small). Recovered vs dropped entity spans per run: Marathi 129 / 4 (base), 119 / 2 (small); Tamil 190 / 6 and 180 / 5; German 191 / 6; French 192 / 6. About 3% of the spans WITH MISSING OFFSETS were dropped, which is 0.14-0.93% of all predicted entities in those runs (at most +0.001 to +0.007 F1 even if every dropped entity had been correct; checked 2026-11-18 13:05); the recovery is an approximation (repeated texts are assigned in order), so these four runs are slightly less exact than the others. The same fix applied to the first runs would have changed nothing there, since they completed without the error.
* With the full set, GLiNER-X base is below indicbert-hi on **every** Indian language (paired bootstrap, same sentences: Marathi -0.199 [-0.228, -0.173]; Tamil -0.092 [-0.124, -0.062]).
* German/French/Spanish: GLiNER-X **base** (0.592 / 0.653 / 0.667) is ahead of GLiNER multi (0.552 / 0.608 / 0.647): German +0.040 [+0.007, +0.073], French +0.045 [+0.021, +0.069], Spanish +0.021 [-0.005, +0.048] (not reliable). GLiNER-X small is not better than multi. So the best open-licence candidate for those three languages is GLiNER-X base (about 494 M parameters), with the caveats of lower-cased MultiCoNER text and an untuned threshold; Chinese improved from 0.023 to 0.266 but is still below the (unlicensed) old model's 0.281.
* Same caveats as above: threshold untuned, MultiCoNER lower-cased. Bootstrap intervals were computed only for the comparisons quoted above (GLiNER-X base against GLiNER multi, and against indicbert-hi on Marathi and Tamil); the rest of the table has none.

### Hinglish: Hindi typed in English letters (2026-11-18)

**Test set** (`evaluation/hinglish/hinglish_sentences.txt`, frozen in `MANIFEST.json` as `hinglish`): 125 sentences, 182 entities (PER/ORG/LOC), 104 with capitalised names, 10 typed in lower case like chat messages, 11 without any entity.
Written by the assistant, not by native speakers: labels and spelling need review by a Hindi speaker; the sentences are cleaner and more entity-dense than real chat text, so real-world scores will be lower.
*Correction:* the first version of the file had no entity markup on the ten lower-case sentences (they were scored as having no entities); that was fixed (164 -> 182 entities) after the first Colab scoring and all numbers below are the rescored ones (`evaluation/rescore_hinglish.py`; the predictions do not depend on the labels).

Strict F1 on the English-letter text (95% bootstrap interval; Colab T4, `ws3e_runs`):

| model | P | R | strict F1 | partial F1 | PER | ORG | LOC | capitalised / lower-case sentences | false entities in the 11 no-entity sentences |
|---|---|---|---|---|---|---|---|---|---|
| **indicbert-hi (Hindi model, CC0), raw text** | 0.949 | 0.929 | **0.939 [0.907, 0.966]** | 0.944 | 0.94 | 0.91 | 0.96 | 0.963 / 0.759 | 3 |
| indicbert-hi after Azure transliteration to Devanagari | 0.901 | 0.901 | 0.901 [0.862, 0.939] | 0.934 | 0.94 | 0.84 | 0.91 | 0.902 / 0.944 | 2 |
| GLiNER-X base (Apache-2.0), raw text | 0.863 | 0.830 | 0.846 [0.792, 0.901] | 0.908 | 0.80 | 0.77 | 0.94 | 0.848 / 0.875 | 2 |
| GLiNER-X base after transliteration | 0.557 | 0.566 | 0.561 [0.493, 0.628] | 0.796 | 0.47 | 0.45 | 0.73 | 0.578 / 0.500 | 6 |
| Davlan multilingual (what the app did before, no licence) | 0.771 | 0.665 | 0.714 [0.644, 0.784] | 0.743 | 0.58 | 0.67 | 0.84 | 0.772 / 0.296 | 9 |
| bert-base-NER (English model) | 0.780 | 0.566 | 0.656 [0.578, 0.731] | 0.707 | 0.61 | 0.60 | 0.74 | 0.698 / 0.000 | 1 |

Paired bootstrap (same sentences), indicbert-hi on raw text minus: Davlan **+0.225 [+0.158, +0.287]**; English model **+0.283 [+0.209, +0.358]**; GLiNER-X raw +0.093 [+0.039, +0.148]; indicbert-hi after transliteration +0.038 [-0.007, +0.086] (not reliable).
GLiNER multi was in the notebook but produced no file (not investigated; it is unusable for Indian languages in any case).

Reading it:
* The Hindi model can read English-letter Hindi directly (IndicBERTv2 saw romanised text in pretraining); converting to Devanagari first does not help reliably (+0.038 in favour of raw, interval includes 0) and would make NER depend on Azure. **No transliteration step is used.**
* Weak spots: names typed without capital letters (0.759 over only 10 sentences; the old route 0.296), honorifics absorbed into a name ("Pankaj bhaiya"), brand names tagged with the wrong type (Maruti PER, Karnataka ORG), and common words tagged as names in 3 of 11 no-entity sentences ("bhai", "mausam").
* **Translation finding (Azure, tested live 2026-11-18):** with source language `hi`, Azure returns English-letter Hindi **unchanged, with no error**; with `hi-Latn` or auto-detection it translates ("Rahul is going to Mumbai tomorrow for a meeting.").

**Detector** (`backend/hinglish.py`; script `evaluation/eval_hinglish_detect.py`): at least 85% Latin letters, at least 2 and at least 20% of words from a list of unambiguous Hindi function words (hai, ka, mein, nahi, kya, ne ...).
First list: 100/125 (80.0%) of the Hinglish test sentences recognised, 39/40 on a fresh sentence list written beforehand; 0 false positives in 7,010 other sentences (English tweets BTC and WNUT, English Wikipedia, German, Spanish, French).
After about 40 words were added by looking at the 25 misses: 122/125 (97.6%, tuned on the same set, so optimistic), 40/40 on the fresh list, still 0/7,010 false positives. langdetect called none of the 125 sentences Hindi.
Limits: a short message with one Hindi word is missed; English text with a few Hindi words is not Hindi by this rule (by design); Roman-script Marathi, Bengali and the other languages are not handled.

### Russian, Arabic, Japanese (and ordinary-text German and Chinese) (2026-11-18; Colab T4, `ws3f_runs`)

Test sets (`evaluation/foreign_data.py`, frozen in `MANIFEST.json`): Universal NER v1 PUD for Russian, German and Chinese (1,000 parallel news/Wikipedia sentences each; Chinese PUD is in Traditional characters), Universal NER Chinese GSDSIMP (500, Simplified), ANERcorp CAMeL Lab test split for Arabic (924 sentences), stockmark Japanese Wikipedia NER (5,343 sentences, one token per character; facilities counted as LOC). Strict F1 with 95% bootstrap interval.

| test set | current model (Davlan WikiANN, no licence) | **hrl_base** (Davlan/xlm-roberta-base-ner-hrl, AFL-3.0) | uner_all (MIT, XLM-R large) | camel_ar (Apache-2.0, Arabic only) | GLiNER-X base (Apache-2.0) |
|---|---|---|---|---|---|
| Russian (UNER PUD) | 0.488 [0.456, 0.519] | **0.740** [0.709, 0.770] | 0.736 [0.705, 0.767] | - | 0.659 [0.625, 0.687] |
| Arabic (ANERcorp) | 0.493 [0.473, 0.513] | 0.812 [0.791, 0.830] (*) | 0.626 [0.603, 0.649] | **0.828** [0.808, 0.847] | 0.607 [0.583, 0.630] |
| Japanese (stockmark) | 0.346 [0.336, 0.356] | 0.637 [0.627, 0.647] | 0.400 [0.390, 0.411] | - | **0.670** [0.661, 0.679] |
| German (UNER PUD) | 0.514 [0.488, 0.539] | **0.821** [0.798, 0.844] | 0.816 [0.793, 0.839] | - | 0.740 [0.714, 0.763] |
| Chinese, Traditional (UNER PUD) | 0.455 [0.429, 0.483] | **0.775** [0.751, 0.799] | 0.769 [0.749, 0.790] | - | 0.033 [0.022, 0.046] |
| Chinese, Simplified (UNER GSDSIMP) | 0.434 [0.387, 0.479] | 0.709 [0.671, 0.747] | **0.779** [0.747, 0.811] (in-domain) | - | 0.627 [0.586, 0.664] |

(*) hrl_base was trained on ANERcorp; its card does not say which split, so this score may include sentences it saw in training. camel_ar was trained on the CAMeL train split, so for it the test is held out.
GLiNER multi was in the notebook but produced no output files (as in the Hinglish run; not investigated).

Paired differences (same sentences): **hrl_base minus the current model: Russian +0.253 [+0.227, +0.279], Arabic +0.318 [+0.296, +0.340], Japanese +0.291 [+0.279, +0.303], German +0.307 [+0.281, +0.335], Chinese Traditional +0.320 [+0.290, +0.348], Chinese Simplified +0.276 [+0.233, +0.321].**
camel_ar minus hrl_base on Arabic +0.016 [+0.001, +0.031]; GLiNER-X minus hrl_base on Japanese +0.033 [+0.022, +0.044]; hrl_base minus uner_all on Russian +0.004 [-0.011, +0.018] and German +0.005 [-0.016, +0.024] (no difference); uner_all minus hrl_base on Simplified Chinese +0.070 [+0.033, +0.108] (uner_all was trained on that treebank's train split).

hrl_base on the lower-case MultiCoNER dev sets (same sentences as the earlier tables): German 0.435 [0.383, 0.483], Spanish 0.362 [0.322, 0.404], French 0.459 [0.423, 0.494], Chinese 0.526 [0.484, 0.572]; minus the current model: +0.094 [+0.037, +0.147], +0.053 [+0.009, +0.094], +0.194 [+0.158, +0.232], +0.245 [+0.200, +0.294]. On lower-case text GLiNER-X base and GLiNER multi are better for German, Spanish and French (0.55-0.67), worse for Chinese.

Reading it:
* The current multilingual model (no licence) is weak on ordinary text in every one of these languages: 0.35-0.51. Its precision is the problem (0.38-0.46): it tags many spans that are not names.
* **`Davlan/xlm-roberta-base-ner-hrl` is better on every set measured, by +0.25 to +0.32 on ordinary text and +0.05 to +0.25 on lower-case text, with an open licence (AFL-3.0), the same size (XLM-R base) and the same label set.** It was not trained on Russian or Japanese and still reaches 0.74 and 0.64 there. It is a drop-in replacement candidate (setting `NER_MODEL_MULTI`), which would also remove the release blocker; switching is the owner's and mentor's decision (item 3).
* Best per language among open models: Russian hrl_base or uner_all (equal); Arabic camel_ar (slightly ahead of hrl_base, and its score is held-out); Japanese GLiNER-X base (slightly ahead of hrl_base, at almost twice the size and with a new dependency); German and Traditional Chinese hrl_base or uner_all; Simplified Chinese uner_all (in-domain). One model for all, hrl_base, is within 0.02-0.07 of the best everywhere.
* Organisations are the weakest type for every model (hrl_base ORG F1 0.45 Russian, 0.62 German, 0.64 Chinese, 0.65 Japanese, 0.69 Arabic).
* Not measured: hrl_base on Odia and Assamese (which also use `NER_MODEL_MULTI` today), and on Indian languages generally.

### Hindi NER latency (E3, 2026-11-18; this laptop: 4 physical cores, 7.3 GB RAM of which only about 0.6 GB was free, CPU, PyTorch fp32, 4 threads, result cache off)

`python evaluation/bench_latency.py --lang hi ...`; texts built from the frozen Naamapadam + HiNER Hindi test sentences; the app's real path (`run_ner`: chunking, model, clean-up) is timed.
Protocol actually run (reduced from E3 because one long call takes 13-17 s here): batch 1: short and medium 30 calls x 3 repeats, long 6 calls x 3 repeats; batch 32: 15 calls x 3 repeats (long 5 x 3); 5-10 warm-up calls; medians of the repeats.
Files: `evaluation/runs/latency__indicbert-hi-ner-naamapadam__b1__hi.json` and `__b32__hi.json`.

| model | batch | short (~200 chars) p50 / p95 | medium (~2,000) p50 / p95 | long (~20,000) p50 / p95 | chars/s (medium) | cold start | on disk |
|---|---|---|---|---|---|---|---|
| indicbert-hi (Hindi, the app's model) | 1 | 402 / 617 ms | 1,971 / 2,463 ms | 14.3 / 17.0 s | 1,011 | 26 s | 1.12 GB |
| indicbert-hi | 32 (documents) | 263 / 363 ms | 1,418 / 1,582 ms | 12.8 / 16.2 s | 1,473 | 20 s | |
| bert-base-NER (English default), same session, for comparison | 1 | 283 / 444 ms | 1,566 / 2,094 ms | - | 1,251 | 19 s | 0.43 GB |

Reading it: on this laptop Hindi NER takes about 2 seconds for a 2,000-character text, about 25% slower than English (1.6 s) on comparable text, and a 20,000-character document takes 13-14 s (p50) at the document batch size.
The model is 2.6 times larger on disk (1.12 GB, because of its 250,000-word vocabulary), so it also loads more slowly. Peak memory of the process was 0.9-1.0 GB.
Caveats: the machine was short of free memory (the operating system was paging), so these numbers are pessimistic and noisier than on a machine with free RAM; a first trial run under the same conditions gave 3.4 s for medium texts. Peak memory readings for long texts are lowered by paging. No ONNX export of the Hindi model was tried.

### Live end-to-end check with the Indian languages (item 6, 2026-11-18)

`python evaluation/e2e_live.py` against a running backend (uvicorn, port 8010) with a throw-away SQLite file and a separate MongoDB database (deleted afterwards), no model preloading, Azure Translator configured.
**38 of 38 checks passed**: `/api/languages` (11 languages listed, Hindi "beta"); sign-up and login; for one sentence each in Hindi, Marathi, Bengali, Tamil, Telugu, Gujarati, Kannada, Malayalam, Punjabi and Hinglish: language detection correct (Hinglish reported as `hi`, `romanized: true`), entity recognition through the Hindi model, translation to English through Azure; saving, listing and exporting a Hindi annotation; a Hindi `.txt` document job (upload, auto-detected as Hindi, NER + translation: 21 entities, 7 unique, no warnings).
Every expected name was found except Bengaluru in the Kannada sentence. Entity spans in the agglutinative languages include case endings ("पुण्यात" = "in Pune", "சென்னையில்" = "in Chennai", "കൊച്ചിയിൽ" = "in Kochi"); the model was trained to mark the whole word, so this is expected, but a user sees "in Pune" rather than "Pune".
The first Hindi request took 24.7 s because the model is loaded on first use (no preloading); later requests took 0.15-0.5 s per sentence. The web front end itself was not driven in a browser in this check (its TypeScript compiles; the API it calls is what was tested).
These ten sentences were written for the check; they are a smoke test, not an accuracy measurement.

### Indian languages: best permissive model per language, with IndicNER scored fairly (2026-11-18)

**Correction to every earlier IndicNER number in this file.** IndicNER (MIT) is multilingual BERT *uncased*: its tokenizer strips Indian vowel signs and viramas before the model sees the text ("प्रकाश" -> "परकाश", "में" -> "म"). The model was trained that way, but the character offsets it then reports no longer cover whole words, so the app-style path (`run_all.py`: Hugging Face pipeline + entity clean-up) cut or extended its entities (Naamapadam Hindi: strict 0.704, partial 0.846, 281 boundary errors).
Scored **word by word** like its paper (first sub-word label per word; `evaluation/word_level_eval.py`), IndicNER reproduces its published numbers (Hindi 0.828 vs 0.829 published, Bengali 0.798 vs 0.807, Marathi 0.805 vs 0.811, Tamil 0.734 vs 0.741, Telugu 0.814 vs 0.822). indicbert-hi scores the same both ways (0.827 / 0.827 Hindi, 0.808 / 0.808 Bengali), so the app path is fair to it. **To use IndicNER in the app, word-level decoding would have to be added.**

Strict F1 [95% CI], full test sets. IndicNER = word-level; the other models through the app path (GLiNER through the adapter).

| test set | sentences | in-domain for | indicbert-hi (CC0) | IndicNER (MIT) | GLiNER-X base | GLiNER-X small | GLiNER multi | IndicNER minus indicbert-hi (paired) |
|---|---|---|---|---|---|---|---|---|
| Hindi, Naamapadam | 864 | both (Naamapadam train) | **0.827** [0.807, 0.847] | **0.828** [0.809, 0.847] | 0.757 | 0.714 | 0.138 | +0.001 [-0.012, +0.014] tie |
| Hindi, HiNER (out-of-domain) | 3,094 | neither | **0.755** [0.746, 0.763] | 0.739 [0.729, 0.747] | 0.689 | 0.705 | 0.103 | -0.016 [-0.023, -0.010] |
| Hindi, MultiCoNER (out-of-domain, lower-case) | 514 | neither | **0.601** [0.555, 0.640] | 0.578 [0.533, 0.618] | 0.479 | 0.427 | 0.028 | -0.023 [-0.057, +0.012] tie |
| Marathi | 1,060 | IndicNER (indicbert-hi: same corpus, other language) | **0.831** [0.808, 0.851] | 0.805 [0.782, 0.827] | 0.631 | 0.582 | 0.064 | -0.026 [-0.042, -0.010] |
| Bengali | 607 | IndicNER | **0.808** [0.776, 0.835] | 0.798 [0.765, 0.829] | 0.657 | 0.597 | 0.092 | -0.010 [-0.034, +0.012] tie |
| Tamil | 758 | IndicNER | **0.750** [0.724, 0.778] | 0.734 [0.708, 0.757] | 0.658 | 0.623 | 0.031 | -0.017 [-0.041, +0.006] tie |
| Telugu | 847 | IndicNER | **0.854** [0.828, 0.876] | 0.814 [0.788, 0.837] | 0.760 | 0.740 | 0.083 | -0.039 [-0.058, -0.020] |
| Gujarati | 1,076 | IndicNER | **0.825** [0.804, 0.846] | 0.796 [0.775, 0.816] | 0.756 | 0.694 | 0.071 | -0.030 [-0.048, -0.013] |
| Kannada | 1,003 | IndicNER | **0.826** [0.803, 0.847] | 0.801 [0.778, 0.822] | 0.760 | 0.692 | 0.045 | -0.025 [-0.040, -0.008] |
| Malayalam | 965 | IndicNER | **0.847** [0.825, 0.867] | 0.807 [0.784, 0.830] | 0.793 | 0.741 | 0.097 | -0.039 [-0.056, -0.021] |
| Punjabi | 993 | IndicNER | **0.732** [0.706, 0.756] | 0.688 [0.663, 0.715] | 0.595 | 0.568 | 0.046 | -0.044 [-0.063, -0.026] |
| Odia (**silver** test) | 994 | IndicNER | 0.167 [0.139, 0.196] | 0.271 [0.204, 0.330] | 0.133 | 0.128 | 0.123 | +0.103 [+0.054, +0.146] |
| Assamese (**silver**, 51 sentences) | 51 | IndicNER | 0.467 [0.300, 0.609] | 0.625 [0.417, 0.800] | 0.333 | 0.377 | 0.263 | +0.158 [+0.019, +0.316] |

"In-domain": IndicNER was trained on the Naamapadam train split of all 11 languages, indicbert-hi on the Naamapadam Hindi train split only; the Naamapadam test sets come from the same projected-news corpus, so they are in-domain for IndicNER in every language and for indicbert-hi in Hindi (for the other languages indicbert-hi saw the same kind of text, but not the language). Only HiNER and MultiCoNER are out-of-domain, and only for Hindi.
The Naamapadam Odia and Assamese test sets are **silver** (machine-projected, not checked by people; Naamapadam paper, Table 3), so they are not used to choose a model.

**Best permissive model per language:** indicbert-hi for all nine gold-tested languages: better in seven (Hindi out-of-domain HiNER, Marathi, Telugu, Gujarati, Kannada, Malayalam, Punjabi; +0.016 to +0.044) and tied in Bengali, Tamil and the two other Hindi sets. So "one model for all" and "best per language" give the same answer today, and IndicNER would additionally need word-level decoding in the app.
Odia and Assamese: undecided until they are scored on human-labelled sets (IL-NER Odia, GUIT-AsTourNE; research plan R2). GLiNER-X is 0.05-0.20 below indicbert-hi everywhere; GLiNER multi is unusable for these scripts.

### E5: UNER English-EWT, the fair English test (2026-11-18; laptop CPU)

UNER English Web Treebank test: 2,077 sentences of emails, reviews, newsgroups and question answers, 1,088 entities (PER 449, ORG 322, LOC 317); CC-BY-SA-4.0; frozen in `MANIFEST.json` as `uner_en_ewt`. **No English model here was trained on it.** It has no "@" or "Borrower"-style labelling artefacts, so names-only scores equal the standard ones.

| model | P | R | strict F1 [95% CI] | minus bert-base (paired) | PER | ORG | LOC | seen in training | F1 unseen subset [95% CI] | recall seen / unseen |
|---|---|---|---|---|---|---|---|---|---|---|
| bert-base-NER (default) | 0.674 | 0.734 | 0.703 [0.674, 0.729] | - | 0.85 | 0.47 | 0.76 | 22% | 0.640 [0.605, 0.672] | 0.847 / 0.702 |
| roberta-large (teacher, 355M) | 0.734 | 0.863 | **0.793** [0.771, 0.816] | +0.090 [+0.068, +0.116] | 0.88 | 0.65 | 0.85 | 24% | 0.737 [0.704, 0.769] | 0.921 / 0.844 |
| student A (distilled, CoNLL + WNUT + BTC + FIN) | 0.656 | 0.782 | 0.714 [0.682, 0.740] | +0.011 [-0.015, +0.034] | 0.85 | 0.50 | 0.76 | 35% | 0.615 [0.574, 0.652] | 0.888 / 0.726 |
| student B (same data as A, NO teacher) | 0.734 | 0.812 | **0.771** [0.744, 0.797] | +0.068 [+0.045, +0.094] | 0.87 | 0.61 | 0.82 | 35% | 0.674 [0.636, 0.712] | 0.904 / 0.764 |
| student C (distilled, CoNLL only) | 0.657 | 0.810 | 0.726 [0.696, 0.751] | +0.023 [+0.002, +0.045] | 0.81 | 0.57 | 0.77 | 22% | 0.668 [0.632, 0.700] | 0.888 / 0.787 |

Same sentences for every model (the 1,775 sentences with no gold entity seen in ANY model's training data): bert-base 0.590 [0.550, 0.627], teacher 0.700 [0.662, 0.733], A 0.614 [0.573, 0.652], **B 0.676** [0.634, 0.711], C 0.630 [0.587, 0.667].
Other paired differences: B minus teacher -0.022 [-0.045, -0.001]; A minus teacher -0.079 [-0.104, -0.056]; B minus C +0.046 [+0.025, +0.065]; A minus C -0.012 [-0.032, +0.006].

Reading it (one seed per student, so the ordering between students is indicative):
* **On text none of the models was trained on, student A (distilled) is not reliably better than the default** (+0.011, interval includes 0) and no better than student C. Its earlier gains on BTC and FIN came from in-domain training data.
* **Student B (plain fine-tuning on the same data, no teacher) is the best small model: +0.068 over the default, +0.058 over A, within 0.022 of the 355M teacher**, at the same size and speed as A. This agrees with the names-only E1 result (B best on WNUT, BTC and FIN) and contradicts the earlier reading that "the teacher is what generalises".
* Organisations remain the weakest type (ORG F1 0.47-0.65).
* Consequence for the English default (mentor decision 1): B, not A, is the candidate. B was trained with FIN, whose licence is unverified, so a **B-noFIN** variant (and A-noFIN for comparison), each with 3 seeds, is needed before either can ship. Neither exists yet.

### WS2b: licence-safe English students without FIN, 3 seeds each (2026-11-19; Colab T4, `ws2b_runs`)

`A_noFIN_distill` (with teacher) and `B_noFIN_plain` (no teacher) trained on CoNLL + WNUT + BTC (no FIN), seeds 0/1/2, best checkpoint chosen on WNUT + BTC validation only (no FIN anywhere). Strict F1; "names" = names-only (the "@" and "Borrower"/"Lender" artefacts removed, E1).

| model | WNUT-2017 (names) | BTC (names) | FIN (names) | **English-EWT** | EWT unseen subset |
|---|---|---|---|---|---|
| A_noFIN s0 / s1 / s2 | 0.515 / 0.581 / 0.548 (0.571 / 0.589 / 0.583) | 0.746 / 0.652 / 0.678 (0.764 / 0.782 / 0.748) | 0.206 / 0.201 / 0.195 | 0.729 / 0.761 / 0.735 | 0.625 / 0.655 / 0.633 |
| **B_noFIN s0 / s1 / s2** | 0.469 / 0.486 / 0.493 (0.573 / 0.597 / 0.599) | 0.805 / 0.818 / 0.814 (0.786 / 0.800 / 0.790) | 0.191 / 0.189 / 0.180 | **0.755 / 0.772 / 0.760** | 0.649 / 0.673 / 0.645 |
| A_noFIN mean +- sd | 0.548 +- 0.033 (0.581 +- 0.009) | 0.692 +- 0.048 (0.764 +- 0.017) | 0.201 +- 0.006 | 0.742 +- 0.017 | |
| **B_noFIN mean +- sd** | 0.483 +- 0.012 (**0.590** +- 0.015) | **0.812** +- 0.007 (**0.792** +- 0.007) | 0.186 +- 0.006 | **0.762 +- 0.009** | |
| for reference (1 seed, with FIN): bert-base / teacher / A / B / C | 0.493 / 0.554 / 0.558 / 0.490 / 0.523 | 0.401 / 0.581 / 0.630 / 0.808 / 0.565 | 0.175 / 0.222 / 0.296 / 0.759 / 0.206 | 0.703 / 0.793 / 0.714 / 0.771 / 0.726 | |

Paired on English-EWT: every B_noFIN seed beats bert-base (+0.052 [+0.027, +0.075], +0.069 [+0.044, +0.095], +0.057 [+0.035, +0.081]) and is within or just below B-with-FIN (-0.017 [-0.033, -0.001], +0.001 [-0.015, +0.018], -0.011 [-0.028, +0.006]). B_noFIN minus A_noFIN, same seed: +0.026 [+0.010, +0.039], +0.011 [-0.005, +0.027], +0.025 [+0.009, +0.043].

Reading it:
* **Dropping FIN costs nothing outside FIN itself**: B_noFIN equals B on EWT (0.762 vs 0.771, within seed spread), on WNUT names (0.590 vs 0.595) and BTC names (0.792 vs 0.787). FIN falls to 0.19 (it was in-domain for B: 0.759).
* **B_noFIN is the licence-safe English candidate**: +0.05 to +0.07 over the current default on text nobody trained on, stable across seeds (sd 0.009), same size and speed as student A. A_noFIN is also better than bert-base on EWT (0.742) but varies more between seeds and is below B_noFIN.
* B_noFIN's low standard WNUT score (0.483) is the "@" convention learned from BTC; on names only it is the best small model there too.
* Which seed to ship must be chosen on the validation score (the `training_log.json` files in Google Drive `ws2b_results`), not on these test scores.

### Replacing the unlicensed multilingual model: tuned zero-shot models and specialists (items 1 and 3 of the brief, research R4; 2026-11-19; Colab T4, `ws3g_runs`)

Zero-shot thresholds were chosen on the MultiCoNER **train** samples (`mc_*_tune`, 600 sentences each for de/es/fr; `evaluation/tune_threshold.py`), never on a test set: **Otter 0.25, GLiNER-X base 0.70, GLiNER-X small 0.60** (default 0.5). The tuned values were then used for every language. Strict F1:

| test set | current (Davlan, no licence) | hrl_base (AFL-3.0) | uner_all (MIT) | Otter (Apache-2.0) | GLiNER-X base | GLiNER-X small | GLiNER multi (0.5) | specialist (permissive) | best open vs current; vs hrl_base |
|---|---|---|---|---|---|---|---|---|---|
| German, MultiCoNER (lower-case) | 0.341 | 0.435 | - | **0.642** | 0.608 | 0.494 | 0.552 | - | +0.301; +0.207 |
| German, UNER PUD (cased) | 0.514 | **0.821** | 0.816 | 0.779 | 0.777 | 0.729 | - | - | +0.307 |
| Spanish, MultiCoNER (lower-case) | 0.310 | 0.362 | - | **0.700** | 0.693 | 0.610 | 0.647 | - | +0.390; +0.338 |
| Spanish, WikiANN (cased, silver; in-domain for Davlan) | 0.849 | 0.606 | - | **0.716** | 0.700 | 0.664 | - | - | -0.133; +0.110 |
| French, MultiCoNER (lower-case) | 0.264 | 0.459 | - | 0.686 | 0.683 | 0.620 | 0.608 | **CamemBERT-dates 0.752** | +0.488; +0.294 |
| French, WikiANN (cased, silver; in-domain for Davlan) | 0.853 | 0.564 | - | **0.702** | 0.696 | 0.664 | - | CamemBERT-dates 0.651 | -0.151; +0.138 |
| Russian, UNER PUD | 0.488 | **0.740** | 0.736 | 0.634 | 0.690 | 0.667 | - | - | +0.253 |
| Japanese, UNER PUD | 0.384 | **0.718** | 0.677 | 0.714 | 0.708 | 0.614 | - | (llm-book: no output) | +0.335 |
| Japanese, stockmark | 0.346 | 0.637 | 0.400 | **0.731** | 0.681 | 0.598 | - | - | +0.385; +0.094 |
| Chinese Traditional, UNER PUD | 0.455 | **0.775** | 0.769 | 0.745 | 0.032 | 0.046 | - | bert4ner 0.718 | +0.320 |
| Chinese Simplified, UNER GSDSIMP | 0.434 | 0.709 | **0.779** (in-domain) | 0.766 | 0.664 | 0.560 | - | bert4ner 0.706 | +0.346; +0.070 |
| Chinese, MultiCoNER | 0.281 | 0.526 | - | **0.651** | 0.313 | 0.183 | 0.023 | bert4ner 0.551 | +0.370; +0.125 |
| Arabic, ANERcorp | 0.493 | 0.812 (*) | 0.626 | 0.653 | 0.513 | 0.485 | - | **CAMeLBERT-MSA 0.843** (mix 0.828) | +0.350; +0.032 |

All "vs" values are paired bootstrap differences with intervals above zero except the two WikiANN rows (Davlan was trained on WikiANN). (*) hrl_base was trained on ANERcorp with an unstated split. Confidence intervals for every cell: `ws3g_runs` + `tune_threshold.py`; the per-test-set threshold effect is in the IMPROVEMENT_LOG entry of 2026-11-20.
GLiNER multi again produced no output for the new sets and the Japanese llm-book model produced no file (error not captured).

Reading it:
* **Every open model tested beats the current model on every human-labelled set**, usually by +0.25 to +0.49.
* **No single open model wins everywhere.** hrl_base is best (or tied) on cased, edited text: German, Russian, Japanese PUD, Traditional Chinese; Otter is best on lower-case text (MultiCoNER de/es/zh), on WikiANN fr/es (among open models) and on Japanese stockmark; specialists win French lower-case (CamemBERT) and Arabic (CAMeLBERT-MSA).
* **GLiNER-X base (the brief's item 3):** with a tuned threshold it beats GLiNER multi (+0.06 to +0.08 on MultiCoNER de/es/fr) and the current model on every human-labelled set, and is +0.04 on cased German PUD after tuning (0.777). But **Otter is equal or better on 12 of 13 sets** (all but Russian, where GLiNER-X base 0.690 vs Otter 0.634) at 309M instead of 494M parameters, so wiring GLiNER-X for de/fr/es is not recommended; Otter or hrl_base is.
* Caveat: Otter's and GLiNER-X's thresholds were tuned on MultiCoNER train, the same corpus family as the MultiCoNER test rows, which favours them on those rows; on the other sets the tuned threshold was transferred from de/es/fr.

### Item 1: GLiNER adapter, offset-less entities (2026-11-19)
The improved matching (case-folded and whitespace-flexible search) recovered **no additional entities**: the six re-runs (Marathi and Tamil for base and small, German and French for base) have exactly the same predictions and F1 as before (difference 0.000). The 2-6 entities per run that still cannot be placed are entities whose returned text does not occur in the sentence; they are 0.14-0.93% of predictions and worth at most +0.007 F1. Closed as measured, no effect.

### Language identification (2026-11-21; `evaluation/eval_language_id.py`)

The app's `detect_language()` against langdetect alone, sentence by sentence, on the first 1,000 sentences of each frozen test set (all of a set if smaller). Correct = detected as the set's own language.

| test set | language | sentences | langdetect alone | app (script first, then langdetect) | most frequent wrong answers (app) |
|---|---|---|---|---|---|
| Naamapadam / HiNER / MultiCoNER Hindi | hi | 864 / 1,000 / 514 | 96.4% / 97.7% / 98.1% | same | mr, ne |
| Hinglish (ours) | hi | 125 | 0.0% | 97.6% | id, so |
| Marathi | mr | 1,000 | 96.4% | 96.4% | hi (22), ne (10) |
| Bengali | bn | 607 | 99.7% | 99.7% | |
| **Assamese** (silver set) | as | 51 | **0.0%** | **92.2%** | bn (3: no ৰ/ৱ in the sentence) |
| **Odia** (silver set) | or | 994 | **0.0%** (langdetect fails) | **99.9%** | |
| Punjabi / Gujarati / Tamil / Telugu / Kannada / Malayalam | | 993 / 1,000 / 758 / 847 / 1,000 / 965 | 99.8-100% | same | |
| English, WNUT-2017 / English-EWT | en | 1,000 / 1,000 | 89.3% / **76.4%** | same | de, tl, none |
| German MultiCoNER / UNER PUD | de | 512 / 1,000 | 80.7% / 99.7% | same | en, nl |
| Spanish / French MultiCoNER | es / fr | 854 / 857 | 92.9% / 91.5% | same | en, it |
| Russian / Chinese / Japanese / Arabic | | 1,000 / 500 / 1,000 / 924 | 99.2% / 95.2% / 99.9% / 99.0% | same | bg / ko / en / fa |
| **all** | | 19,365 | 90.2% | **96.2%** | |

Reading it: the script rule fixes Odia and Assamese and changes nothing else. **Open problem:** langdetect is unreliable on short English sentences (EWT 76.4%), and on lower-case MultiCoNER text; short English typed with the language set to "auto" can be routed to the multilingual model.

## Caveats

* Small samples: differences under about 0.03 F1 or 2 chrF++ points are noise.
* Entity scoring is strict (type and both boundaries must match) and covers only person, organization and location.
* Translation scores use one human reference per segment, so a correct paraphrase loses points. They compare providers on
  the same sample; they are not an absolute quality grade.
* WMT24++ segments are informal (social media, literary), harder than typical business text.
* `Davlan/xlm-roberta-base-wikiann-ner` was trained on WikiANN and the CoNLL models on CoNLL-2003, so scores on those
  sets are optimistic.

## Appendix A: raw tables from earlier runs
These were separate files written by the evaluation scripts; they were merged here on 20 Nov 2026 so all results live in one file. They are kept exactly as the scripts wrote them (the headings were demoted by one level). `translation_results.md` was empty and was removed.

### A.1 NER on BTC and FIN with a confidence-threshold sweep (first 500 sentences; 29 Oct 2026) (was `ner_btc_fin.md`)

### Entity recognition (PER / ORG / LOC, exact entity match)

| dataset | lang | model | sentences | gold entities | precision | recall | F1 |
|---|---|---|---|---|---|---|---|
| btc | en | `bert-base-NER` | 500 | 1076 | 0.630 | 0.289 | 0.396 |
| btc | en | `bert-large-NER` | 500 | 1076 | 0.622 | 0.321 | 0.423 |
| btc | en | `roberta-large-ner-english` | 500 | 1076 | 0.753 | 0.475 | 0.582 |
| fin | en | `bert-base-NER` | 305 | 311 | 0.151 | 0.209 | 0.175 |
| fin | en | `bert-large-NER` | 305 | 311 | 0.266 | 0.183 | 0.217 |
| fin | en | `roberta-large-ner-english` | 305 | 311 | 0.299 | 0.177 | 0.222 |

#### Effect of a minimum confidence score (cut-off picked on the dev split, reported on test)

| dataset | lang | model | min score | precision | recall | F1 |
|---|---|---|---|---|---|---|
| btc | en | `bert-base-NER` | 0.00 (best on dev) | 0.630 | 0.289 | 0.396 |
| btc | en | `bert-base-NER` | 0.50 | 0.643 | 0.280 | 0.390 |
| btc | en | `bert-base-NER` | 0.60 | 0.662 | 0.277 | 0.391 |
| btc | en | `bert-base-NER` | 0.70 | 0.687 | 0.267 | 0.384 |
| btc | en | `bert-base-NER` | 0.80 | 0.718 | 0.260 | 0.382 |
| btc | en | `bert-base-NER` | 0.90 | 0.758 | 0.242 | 0.366 |
| btc | en | `bert-base-NER` | 0.95 | 0.777 | 0.224 | 0.348 |
| btc | en | `bert-large-NER` | 0.00 (best on dev) | 0.622 | 0.321 | 0.423 |
| btc | en | `bert-large-NER` | 0.50 | 0.638 | 0.309 | 0.416 |
| btc | en | `bert-large-NER` | 0.60 | 0.652 | 0.296 | 0.408 |
| btc | en | `bert-large-NER` | 0.70 | 0.687 | 0.285 | 0.403 |
| btc | en | `bert-large-NER` | 0.80 | 0.726 | 0.276 | 0.400 |
| btc | en | `bert-large-NER` | 0.90 | 0.781 | 0.256 | 0.385 |
| btc | en | `bert-large-NER` | 0.95 | 0.816 | 0.235 | 0.365 |
| btc | en | `roberta-large-ner-english` | 0.00 (best on dev) | 0.753 | 0.475 | 0.582 |
| btc | en | `roberta-large-ner-english` | 0.50 | 0.756 | 0.472 | 0.581 |
| btc | en | `roberta-large-ner-english` | 0.60 | 0.759 | 0.465 | 0.576 |
| btc | en | `roberta-large-ner-english` | 0.70 | 0.766 | 0.451 | 0.568 |
| btc | en | `roberta-large-ner-english` | 0.80 | 0.781 | 0.445 | 0.567 |
| btc | en | `roberta-large-ner-english` | 0.90 | 0.809 | 0.425 | 0.557 |
| btc | en | `roberta-large-ner-english` | 0.95 | 0.829 | 0.406 | 0.545 |
| fin | en | `bert-base-NER` | 0.00 | 0.151 | 0.209 | 0.175 |
| fin | en | `bert-base-NER` | 0.50 (best on dev) | 0.155 | 0.199 | 0.174 |
| fin | en | `bert-base-NER` | 0.60 | 0.163 | 0.190 | 0.175 |
| fin | en | `bert-base-NER` | 0.70 | 0.173 | 0.183 | 0.178 |
| fin | en | `bert-base-NER` | 0.80 | 0.187 | 0.177 | 0.182 |
| fin | en | `bert-base-NER` | 0.90 | 0.211 | 0.170 | 0.189 |
| fin | en | `bert-base-NER` | 0.95 | 0.250 | 0.164 | 0.198 |
| fin | en | `bert-large-NER` | 0.00 | 0.266 | 0.183 | 0.217 |
| fin | en | `bert-large-NER` | 0.50 (best on dev) | 0.281 | 0.180 | 0.220 |
| fin | en | `bert-large-NER` | 0.60 | 0.288 | 0.177 | 0.219 |
| fin | en | `bert-large-NER` | 0.70 | 0.297 | 0.174 | 0.219 |
| fin | en | `bert-large-NER` | 0.80 | 0.301 | 0.167 | 0.215 |
| fin | en | `bert-large-NER` | 0.90 | 0.299 | 0.148 | 0.198 |
| fin | en | `bert-large-NER` | 0.95 | 0.288 | 0.129 | 0.178 |
| fin | en | `roberta-large-ner-english` | 0.00 (best on dev) | 0.299 | 0.177 | 0.222 |
| fin | en | `roberta-large-ner-english` | 0.50 | 0.299 | 0.177 | 0.222 |
| fin | en | `roberta-large-ner-english` | 0.60 | 0.296 | 0.170 | 0.216 |
| fin | en | `roberta-large-ner-english` | 0.70 | 0.305 | 0.170 | 0.219 |
| fin | en | `roberta-large-ner-english` | 0.80 | 0.313 | 0.167 | 0.218 |
| fin | en | `roberta-large-ner-english` | 0.90 | 0.331 | 0.167 | 0.222 |
| fin | en | `roberta-large-ner-english` | 0.95 | 0.352 | 0.161 | 0.221 |

### A.2 Offline translation for Chinese, Arabic and Japanese (40 WMT24++ segments; 30 Oct 2026) (was `translation_local_zh_ar_ja.md`)

### Translation quality, English -> X (WMT24++, human references)

| language | provider | segments | failed | chrF++ | BLEU | characters/second |
|---|---|---|---|---|---|---|
| zh | Local open-source models | 40 | 0 | 17.8 | 22.5 | 22 |
| ar | Local open-source models | 40 | 0 | 32.1 | 7.9 | 37 |
| ja | Local open-source models | 40 | 0 | 4.4 | 2.6 | 28 |

#### Examples (first three sampled segments)

**zh / Local open-source models**

* EN: Bc the figures do look cool, and I like they way they're constructed, and how the design reflects certain characteristics, but then every set is like "oh by the
  * system: Bc数字看起来确实很酷,我喜欢他们的方式, 以及设计如何反映某些特点, 但是每套图像“哦,顺便说一句,这里有你的激光剑和(至少一)枪!”
  * reference: 这些机器人的模型确实很酷，我很喜欢它们的构造方式，以及设计如何体现某些角色特征，但它们总是配备着各种武器，激光剑啦，枪啦，好像在说，“对了，这些是你杀人的家伙！”
* EN: The ESP didn't go online immediately, let's give it a few minutes.
  * system: ESP没有马上上线,让我们给它几分钟。
  * reference: ESP没有马上上线，再等几分钟吧。
* EN: “Shit. Its getting bad on the front, '' Nemic muttered. “All the news leaving this place is ‘Support the war effort. Send aid to the Kurgan Front’ how about sen
  * system: “妈的,它在前线变得很糟糕,'''超人说,“离开这个地方的所有新闻都是`支持战争努力,向库尔根阵线提供援助 ' ,如何向豪兰阵线提供援助呢?”
  * reference: “真糟糕。前线战事吃紧啊，”尼米奇低声说道。“从这里传出的消息都是‘支援战争，援助库尔干前线’，怎么不支援一下豪兰前线呢？”

**ar / Local open-source models**

* EN: Bc the figures do look cool, and I like they way they're constructed, and how the design reflects certain characteristics, but then every set is like "oh by the
  * system: Bc الأرقام تبدو باردة، وأنا أحب الطريقة التي يتم بها بناؤها، وكيف يعكس التصميم خصائص معينة، ولكن بعد ذلك كل مجموعة هي مثل "أوه من خلال الطريق هنا هو سيف الليزر 
  * reference: لأن الشخصيات واضح إنها مرحة، وأعجبنتي طريقة تصميمها وكيف أن التصميم يعكس سمات معينة، لكن بعد ذا كل مجموعة ملخصها "أوه هنا سيفك الليز وبنقدية (واحدة على أقل شيء)
* EN: The ESP didn't go online immediately, let's give it a few minutes.
  * system: الـ "إي إس بي" لم يتم تشغيله على الإنترنت فوراً لنعطيه بضع دقائق
  * reference: الـ ESP ما اتصلت بالإنترنت فورًا، راح أنتظر شوي.
* EN: “Shit. Its getting bad on the front, '' Nemic muttered. “All the news leaving this place is ‘Support the war effort. Send aid to the Kurgan Front’ how about sen
  * system: "الخطيئة: إنها تزداد سوءاً في الجبهة، "Nemisk brected. "كل الأخبار التي تغادر هذا المكان هي "دعم جهود الحرب. أرسل المعونة إلى جبهة كورغان "ما رأيك بإرسال المعون
  * reference: نيميتش تمتم وقال: "للأسف، الوضع بيزيد سوء على الجبهة". "كل الأخبار اللي بنسمعها من هناك عن دعم المجهود الحربي وإرسال المساعدات لجبهة كورغان، طب وحّنا؟ جبهة هاول

**ja / Local open-source models**

* EN: Bc the figures do look cool, and I like they way they're constructed, and how the design reflects certain characteristics, but then every set is like "oh by the
  * system: ただあなたのつるぎによって,のろいの事を見た. 彼らは階段を敷く. また みめめき,ボゼバテ・ハケブの近くで,つるぎのかたわらに置く. (つるぎ のみ,またホラマタ)・ハシタタタと なぜそれが置かれる."
  * reference: だってフィギュアはすごくかっこいいし、その造りだったり、デザインに一定の特徴が表現される感じも好きだけど、それでもどのセットも「あ、そうだ、はい、レーザーの剣と銃（最低でも一台）ね！」ってな感じじゃん
* EN: The ESP didn't go online immediately, let's give it a few minutes.
  * system: 災 は たちまち , あお ぎ 分け に な ら な い よう に , 全体 を 少な く し て み た .
  * reference: ESPではすぐにはネット接続ができなかったので、少し待とう。
* EN: “Shit. Its getting bad on the front, '' Nemic muttered. “All the news leaving this place is ‘Support the war effort. Send aid to the Kurgan Front’ how about sen
  * system: 罪 を 犯 し た 理由 が あ り , 前 に 悪 い こと が あ る . そこ の 知らせ は ことごとく 去 っ て , この 所 を 去 っ て い る . 戦い は どこ に あ る の か , 大いなる 場所 に のぼ っ て 行 き なさ い . われわれ の 倒れ た 所 に い る から , どこ 
  * reference: 「くそ。前線の戦況が悪くなってきた」ネミックがつぶやいた。「ここから発信されるニュースは『戦争努力に応援を。クルガン前線に支援物資を』ばかりだ。ハウランド前線に支援物資を送ったらどうなんだ」

### A.3 WS1: Japanese with opus-tatoeba-en-ja (16 Nov 2026) (was `ws1_ja_tatoeba.md`)

### Translation quality, English -> X (WMT24++, human references)

| language | provider | segments | failed | chrF++ | BLEU | characters/second |
|---|---|---|---|---|---|---|
| ja | Local open-source models | 40 | 0 | 17.9 | 19.9 | 43 |

#### Examples (first three sampled segments)

**ja / Local open-source models**

* EN: Bc the figures do look cool, and I like they way they're constructed, and how the design reflects certain characteristics, but then every set is like "oh by the
  * system: この図面はクールに見えるが,この図は,その形質が,ある特徴を反映しているのを好きであるが,いずれのセットも,"この方法では,レーザー剣と(最低でも1つ)銃のように"という.
  * reference: だってフィギュアはすごくかっこいいし、その造りだったり、デザインに一定の特徴が表現される感じも好きだけど、それでもどのセットも「あ、そうだ、はい、レーザーの剣と銃（最低でも一台）ね！」ってな感じじゃん
* EN: The ESP didn't go online immediately, let's give it a few minutes.
  * system: 数分間 投稿します YouTubeで YouTubeで Twitterで
  * reference: ESPではすぐにはネット接続ができなかったので、少し待とう。
* EN: “Shit. Its getting bad on the front, '' Nemic muttered. “All the news leaving this place is ‘Support the war effort. Send aid to the Kurgan Front’ how about sen
  * system: 2019年3月26日閲覧. ^ ""[[[リンク切れ]"前線で不吉な状況になった"". ニッチェネ・ニッセンサート (2019年3月26日).
  * reference: 「くそ。前線の戦況が悪くなってきた」ネミックがつぶやいた。「ここから発信されるニュースは『戦争努力に応援を。クルガン前線に支援物資を』ばかりだ。ハウランド前線に支援物資を送ったらどうなんだ」

### A.4 WS1: Japanese with staka/fugumt-en-ja (16 Nov 2026) (was `ws1_ja_fugumt.md`)

### Translation quality, English -> X (WMT24++, human references)

| language | provider | segments | failed | chrF++ | BLEU | characters/second |
|---|---|---|---|---|---|---|
| ja | Local open-source models | 40 | 0 | 16.4 | 20.6 | 48 |

#### Examples (first three sampled segments)

**ja / Local open-source models**

* EN: Bc the figures do look cool, and I like they way they're constructed, and how the design reflects certain characteristics, but then every set is like "oh by the
  * system: 図はカッコいいように見え、その作り方が気に入って、その設計はどんな特徴を反映しているかが気に入ってる。でもどの一組も「あぁ、レーザーの刀と銃がここにある!」みたいなものなんだ!
  * reference: だってフィギュアはすごくかっこいいし、その造りだったり、デザインに一定の特徴が表現される感じも好きだけど、それでもどのセットも「あ、そうだ、はい、レーザーの剣と銃（最低でも一台）ね！」ってな感じじゃん
* EN: The ESP didn't go online immediately, let's give it a few minutes.
  * system: すぐにはオンラインに行かなかったので、数分で終わります。
  * reference: ESPではすぐにはネット接続ができなかったので、少し待とう。
* EN: “Shit. Its getting bad on the front, '' Nemic muttered. “All the news leaving this place is ‘Support the war effort. Send aid to the Kurgan Front’ how about sen
  * system: 「うるせえ。前がひどくなりつつある。ネミックは、こうつぶやいた。この場所を去るすべての知らせが『戦争の努力を後援する、クルガニ戦線の救援を」
  * reference: 「くそ。前線の戦況が悪くなってきた」ネミックがつぶやいた。「ここから発信されるニュースは『戦争努力に応援を。クルガン前線に支援物資を』ばかりだ。ハウランド前線に支援物資を送ったらどうなんだ」

### A.5 WS1: Chinese with the target-language token (16 Nov 2026) (was `ws1_zh_token.md`)

### Translation quality, English -> X (WMT24++, human references)

| language | provider | segments | failed | chrF++ | BLEU | characters/second |
|---|---|---|---|---|---|---|
| zh | Local open-source models | 40 | 0 | 16.4 | 20.5 | 101 |

#### Examples (first three sampled segments)

**zh / Local open-source models**

* EN: Bc the figures do look cool, and I like they way they're constructed, and how the design reflects certain characteristics, but then every set is like "oh by the
  * system: Bc数字看起来确实很酷,我喜欢他们的方式, 以及设计如何反映某些特点, 但每套图像"哦,顺便说一句,这里是你的激光剑和(至少是)枪!"
  * reference: 这些机器人的模型确实很酷，我很喜欢它们的构造方式，以及设计如何体现某些角色特征，但它们总是配备着各种武器，激光剑啦，枪啦，好像在说，“对了，这些是你杀人的家伙！”
* EN: The ESP didn't go online immediately, let's give it a few minutes.
  * system: ESP没有马上上线,让我们给它几分钟。
  * reference: ESP没有马上上线，再等几分钟吧。
* EN: “Shit. Its getting bad on the front, '' Nemic muttered. “All the news leaving this place is ‘Support the war effort. Send aid to the Kurgan Front’ how about sen
  * system: “妈的,它在前线越来越糟糕,'''“超人”的胡言乱语。 “离开这个地方的所有新闻都是`支持战争努力。 向库尔干阵线提供援助 ' , 如何向豪兰阵线提供援助呢?”
  * reference: “真糟糕。前线战事吃紧啊，”尼米奇低声说道。“从这里传出的消息都是‘支援战争，援助库尔干前线’，怎么不支援一下豪兰前线呢？”

### A.6 WS1: Arabic with the target-language token (16 Nov 2026) (was `ws1_ar_token.md`)

### Translation quality, English -> X (WMT24++, human references)

| language | provider | segments | failed | chrF++ | BLEU | characters/second |
|---|---|---|---|---|---|---|
| ar | Local open-source models | 40 | 0 | 30.4 | 8.3 | 81 |

#### Examples (first three sampled segments)

**ar / Local open-source models**

* EN: Bc the figures do look cool, and I like they way they're constructed, and how the design reflects certain characteristics, but then every set is like "oh by the
  * system: Bc الأرقام تبدو رائعة، وأنا أحب الطريقة التي بنيت بها، وكيف يعكس التصميم خصائص معينة، ولكن بعد ذلك كل مجموعة هي مثل "أوه بالمناسبة هنا سيف الليزر الخاص بك و (عل
  * reference: لأن الشخصيات واضح إنها مرحة، وأعجبنتي طريقة تصميمها وكيف أن التصميم يعكس سمات معينة، لكن بعد ذا كل مجموعة ملخصها "أوه هنا سيفك الليز وبنقدية (واحدة على أقل شيء)
* EN: The ESP didn't go online immediately, let's give it a few minutes.
  * system: لم يتم تشغيل الـ "إي إس بي" فوراً، لنمنحه بضع دقائق.
  * reference: الـ ESP ما اتصلت بالإنترنت فورًا، راح أنتظر شوي.
* EN: “Shit. Its getting bad on the front, '' Nemic muttered. “All the news leaving this place is ‘Support the war effort. Send aid to the Kurgan Front’ how about sen
  * system: "اللعنة، لقد ساءت الأمور في الجبهة، نيميش غمض. "كل الأخبار التي تغادر هذا المكان هي "دعم الجهود الحربية، أرسل المعونة إلى جبهة كورغان " ماذا عن إرسال المعونة إل
  * reference: نيميتش تمتم وقال: "للأسف، الوضع بيزيد سوء على الجبهة". "كل الأخبار اللي بنسمعها من هناك عن دعم المجهود الحربي وإرسال المساعدات لجبهة كورغان، طب وحّنا؟ جبهة هاول
