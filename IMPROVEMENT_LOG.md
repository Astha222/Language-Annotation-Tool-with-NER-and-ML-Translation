# Improvement log: what we did, why, and what we got

This is the running record of the work done from `NER_Improvement_Plan_share.zip` (the improvement plan). It is
updated after every step. For each step it says **what** was done, **why**, **how**, the **result** (with real numbers
from `evaluation/`), and **which problem of the project it solves**. A workstream is only closed after the project
owner acknowledges it.

Machine used for every timing: AMD Ryzen 3 5300U, 4 physical cores / 8 threads, 7.3 GB RAM, Windows 11, CPU only.

| Workstream | Status |
|---|---|
| WS0 Benchmark harness | finished 2026-11-16 06:58, acknowledged |
| WS1 Quick wins | finished 2026-11-16, acknowledged |
| WS2 English NER | finished 2026-11-16 (step 2.6 not done), acknowledged |
| WS3 Hindi and other-language NER | finished 2026-11-17 except IndicNER (needs the owner's Hugging Face login), waiting for the owner's acknowledgement |
| WS4 Consistency sweep and rules | not started |
| WS5 Translation speed and quality | not started |
| WS6 Annotation features | not started |
| WS7 Robustness | not started |
| WS8 Release | not started |

---

## WS0. Benchmark harness

### Why this comes first
Until now every comparison in the project rested on 40 translation segments and 500 NER sentences, with timings taken
on a shared laptop (one run took 37 minutes). With that little data, differences under about 2 chrF++ or 0.03 F1 are
noise, and "faster" or "more accurate" were opinions. Every later decision in the plan (which English model, whether a
change helped, whether Japanese is fixed) needs a fair, repeatable measuring stick. WS0 builds it.

### Step 1. Freeze the test data (`evaluation/download_data.py`, `evaluation/data/MANIFEST.json`)
- **What:** downloads each NER test and dev split once and records, for every file, the number of sentences and its
  sha256 checksum in `MANIFEST.json`. `python download_data.py --verify` checks that nothing has changed.
- **Why:** a score only means something if it is measured on exactly the same sentences each time. Before this, files
  were cached but nothing proved they were unchanged, and runs used "the first 500" without saying so.
- **Result:** 11 files frozen. Test sets: WikiANN English 10,000 sentences, WikiANN Hindi 1,000, CoNLL-2003 3,453,
  WNUT-2017 1,287, BTC 2,000, FIN 305. Dev sets (used only for tuning, never for reporting): WikiANN en/hi, WNUT-2017,
  BTC, FIN. Splits are used in file order, never sampled at random.
- **Problem it solves:** results that cannot be reproduced.

### Step 2. Accuracy runner (`evaluation/run_all.py`)
- **What:** scores any model on the frozen sets and saves every prediction to `evaluation/runs/*.jsonl`. It reports
  strict entity F1 (type and both boundaries must match), partial-match F1 (right type, overlapping span), and F1 per
  type (PER, ORG, LOC). It refuses to run if a data file no longer matches the manifest. `--seqeval-check` recomputes
  the strict score with the independent `seqeval` library.
- **Why:** strict F1 alone hides *why* a model fails. The gap between strict and partial F1 shows how many errors are
  only boundary mistakes. Saved predictions let two runs be compared later without running the models again.
- **Check:** on 50 WNUT sentences our strict F1 was 0.727 and `seqeval` gave exactly 0.727, so the scoring is correct.
  Partial F1 was 0.848.
- **Problem it solves:** one number (F1) that could not be explained or re-compared.

### Step 3. Confidence intervals (`evaluation/bootstrap.py`)
- **What:** given two saved runs on the same sentences, resamples the sentences 1,000 times (the same resample for both
  systems) and reports the 95% interval of the F1 difference, plus a permutation test p-value for small sets such as FIN.
  System A wins only if the whole interval is above zero.
- **Why:** the plan's whole point is to stop trusting small differences. This tells us which differences are real.
- **Check:** comparing a run with itself gives a difference of +0.000 with interval [0, 0].
- **Problem it solves:** deciding between models on noise.

### Step 4. Error analysis (`evaluation/error_analysis.py`)
- **What:** writes a CSV with one row per gold or predicted entity, labelled `correct`, `missed`, `wrong_type`,
  `wrong_boundary` or `spurious`, plus `found_elsewhere_in_doc` (was the same text found correctly in another sentence).
- **Why:** the plan's lesson was that many "misses" are not misses at all but boundary or type errors, which are cheaper to
  fix. The last column predicts how much the document-level sweep (WS4) can recover before we build it.
- **Check:** on the 50-sentence sample, 75% of entities were correct, 4 of the 8 errors were boundary errors, 3 misses
  and 1 wrong type.
- **Problem it solves:** not knowing *what kind* of mistakes the model makes.

### Step 5. Latency runner (`evaluation/bench_latency.py`) and Pareto plot (`evaluation/pareto.py`)
- **What:** times the app's real NER path (chunking, model, clean-up) on fixed short, medium and long texts with warm-up,
  fixed CPU threads, p50 and p95 times, characters per second, peak memory, model size on disk and cold-start time, at
  batch 1 (interactive) and 32 (documents). `pareto.py` plots F1 against p95 latency and marks the models that are
  not beaten on both.
- **Why:** "roberta-large is 3 to 4 times slower" came from timings on a busy laptop. The mentor's decision should rest on
  a clean speed-versus-accuracy plot.
- **Problem it solves:** the speed-versus-accuracy deadlock had no clean speed numbers.

### Results of WS0

**1. Acceptance test (reproduce the old numbers, first 500 sentences, bert-base-NER, tolerance +-0.01): passed.**

| Test set | Earlier report | Harness | Difference |
|---|---|---|---|
| WikiANN | 0.557 | 0.550 | 0.007 |
| WNUT-2017 | 0.592 | 0.583 | 0.009 |
| BTC | 0.396 | 0.396 | 0.000 |
| FIN | 0.175 | 0.175 | 0.000 |
| BTC, roberta-large | 0.582 | 0.582 | 0.000 |

**2. The harness corrected an earlier number.** On the full test sets the current model scores:

| Test set | Sentences | Strict F1 [95% CI] | Partial F1 | PER / ORG / LOC |
|---|---|---|---|---|
| WNUT-2017 | 1,287 | **0.493** [0.459, 0.525] | 0.535 | 0.614 / 0.305 / 0.531 |
| BTC | 2,000 | 0.401 [0.382, 0.422] | 0.430 | 0.342 / 0.380 / 0.610 |
| FIN | 305 | 0.175 [0.127, 0.225] | 0.270 | 0.186 / 0.127 / 0.327 |

The first 500 WNUT-2017 sentences were easier than the whole set (0.583 versus 0.493), so every earlier WNUT-2017
number (0.592, 0.617, 0.657) was too high. The README line was corrected; only bert-base has been re-run on the full
set, so bert-large and roberta-large still need re-running. WikiANN was not run in full (10,000 sentences, about
16 minutes, silver labels); only its first 500 are scored.

**3. What the model's mistakes really are** (share of gold entities):

| Test set | Correct | Missed | Wrong type | Wrong boundary | Spurious predictions |
|---|---|---|---|---|---|
| BTC | 29% | **61%** | 8% | 2% | 268 |
| WNUT-2017 | 49% | 37% | 10% | 4% | 265 |
| FIN | 21% | 26% | **40%** | 13% | 190 |

Twitter is a real recall problem (the model never sees 61% of entities). Financial text is mostly a *type* problem,
which the earlier reports described wrongly as "misses most entities". On FIN, 57% of the not-found entities were
found correctly elsewhere in the set (3% on BTC and WNUT-2017, which are unrelated single tweets). That is a hint,
not a measurement, that a document-level sweep (WS4) can help financial documents.

**4. Is roberta-large really better on Twitter?** Yes: on BTC (500 sentences) it beats bert-base by +0.186 F1,
95% CI [+0.156, +0.219], permutation p = 0.0002.

**5. First clean speed numbers** (PyTorch fp32, 4 threads, 30 timed calls, 1 repeat; plan asks for 100 calls and 3
repeats, so treat as steady to about +-10%):

| Model | Input | p50 | p95 | Speed | Peak RAM |
|---|---|---|---|---|---|
| bert-base-NER (0.43 GB, cold start 15 s) | ~200 chars | 230 ms | 378 ms | 1,076 chars/s | 0.88 GB |
| bert-base-NER | ~2,000 chars | 1,437 ms | 1,858 ms | 1,364 chars/s | 0.91 GB |
| roberta-large (1.42 GB, cold start 17.5 s) | ~200 chars | 906 ms | 1,643 ms | 262 chars/s | 1.75 GB |
| roberta-large | ~2,000 chars | 4,776 ms | 6,371 ms | 405 chars/s | 1.46 GB |

roberta-large is about 3.3 to 3.5 times slower, which confirms the earlier "3 to 4 times" claim with clean timings.
Batch size 32 changed nothing for texts of at most three chunks; long documents are not yet timed.

**6. Pareto plot** (`evaluation/pareto.png`): on BTC both models are on the frontier. roberta-large buys +0.19 F1 for
about 3.4 times the latency. Nothing is faster *and* better yet, which is exactly the gap WS1 (ONNX INT8) and WS2
(distilled student) aim to close.

### Problems met in WS0 and how they were fixed
- A small 50-sentence check overwrote the saved 500-sentence BTC run. Fix: run files now carry the sentence count in
  their name (`..__n500.jsonl`), so a smaller run can no longer replace a bigger one.
- `bench_latency.py` could only time the app's own model. Fix: it now swaps in any model id.
- Timestamps in my progress messages were at first typed from memory, not read from the clock. Corrected; every stamp
  now comes from the system clock.

### Not done in WS0 (honest list)
- Latency repeated only once, with 30 calls; the long-text size (20,000 chars) was not timed.
- WikiANN not run in full; bert-large and roberta-large not re-run on the full WNUT-2017 and FIN sets.
- Hindi test sets are frozen in the manifest but not scored (that is WS3).
- The error CSV for the sweep estimate treats a whole test set as one "document".

### Problem it solves for the project
Before WS0 the project could not say which model was better, by how much, or whether a change helped; one headline
number was wrong by 0.1. Now every number comes with a sample size, an interval, a saved prediction file and (for
speed) the hardware. This is the measuring stick that WS1 to WS8 will be judged against.


---

## WS1. Quick wins

Hardware for every number: AMD Ryzen 3 5300U (4 cores / 8 threads), 7.9 GB RAM, CPU only, Windows 11, Python 3.14, torch 2.11,
transformers 4.57.6. Full tables are in `evaluation/RESULTS.md` (section "WS1").

### Step 1. Offline English to Japanese
- **Diagnosis (confirmed in code and model cards):** `local_translator.py` loaded `Helsinki-NLP/opus-mt-en-jap`. `jap` is ISO 639-3 Jaruara, not Japanese,
  and the model card reports a single test set, `bible-uedin.en.jap`. This explains the unrelated output (chrF++ 4.4).
- **Change:** default `Helsinki-NLP/opus-tatoeba-en-ja` (Apache-2.0, 155 MB); `LOCAL_MT_JA_MODEL` overrides it.
- **Result (same 40 WMT24++ segments, beam 2):** chrF++ 4.4 -> **17.9**, BLEU (char) 2.6 -> 19.9. Bake-off with `evaluation/bakeoff_translation.py`: fugumt-en-ja 16.4 / 20.6,
  with 1 of 40 outputs flagged suspect (length ratio < 0.4 or > 2.5 against the reference) against 5 of 40 for Tatoeba. chrF++ differences of this size are noise at n = 40,
  but reading the outputs fugumt is clearly less prone to hallucinated Wikipedia-style text.
- **Decision:** Tatoeba stays the default because fugumt is CC-BY-SA-4.0; fugumt is opt-in. `facebook/m2m100_418M` was not scored (its cache entry was 3.6 GB, too large for this
  machine); `madlad400-3b-mt` is deferred to WS5 (needs CTranslate2 INT8 to fit in RAM).
- **Plan target not met:** acceptance was chrF++ >= 25; we reached 17.9 (Azure 38.3). The plan's fallback applies: Japanese goes to Azure when a key exists (already first in the
  provider chain), and the README states that offline Japanese is weak.

### Step 2. Target-language tokens for the multi-target Marian models
- **Change:** `>>cmn_Hans<<` for en->zh and `>>ara<<` for en->ar are prepended to every source sentence (the model cards say the token is required); the engine raises if the token
  is not in the tokenizer vocabulary; unit tests assert presence, position, and absence for other pairs.
- **Result:** **no measurable effect.** zh 17.8 -> 16.4 chrF++ (BLEU 22.5 -> 20.5), ar 32.1 -> 30.4 (BLEU 7.9 -> 8.3), all inside the +-2 noise of 40 segments. The plan's hypothesis
  (missing token as the cause of weak zh/ar) is **not supported**: the models already produced Simplified Chinese and Standard Arabic. The token is kept because the model card requires it.
- **Side finding (not in the plan):** the zh output uses half-width ASCII punctuation whereas the references use full-width; chrF++ counts punctuation characters, so normalising
  punctuation for zh/ja could raise scores. Not implemented.

### Step 3. Preload and warm models at startup
- **Change:** FastAPI `lifespan` runs `preload_models()` (`PRELOAD_MODELS`, default `ner_en`; also `ner_multi`, `mt:<src>-<tgt>`); failures are logged and never stop startup;
  `GET /health/models` reports preload status, loaded NER/MT models, and disk space.
- **Result (real model, fresh process each):** first user request **17.36 s -> 0.09 s**; the 15.2 s moves to startup.

### Step 4. CPU threads
- **Measurement 1 (one NER call, medium text, p50, three alternating rounds, 10 calls each):** 4 threads 1,668-1,784 ms vs 8 threads 1,383-1,420 ms (8 is about 20% faster).
- **Measurement 2 (4 NER calls + 14 translated sentences, as in a document job):** global 8 threads: NER 4.7 s, MT 24.6 s, together 26.4 s; global 4 threads: 5.3 s, 18.4 s, 20.6 s;
  4 threads set per task: 5.2 s, 18.2 s, 20.2 s. Eight threads makes translation 34% slower and the combined job 28% slower (SMT over-subscription).
- **Change:** `configure_threads()` sets `torch.set_num_threads` explicitly: `TORCH_THREADS`, else physical cores (psutil), else PyTorch's default. `TORCH_THREADS` is also passed to
  ONNX Runtime sessions. 8 threads stays available as an opt-in for single-request NER servers.

### Step 5. ONNX Runtime / INT8 for English NER
- **Export tool:** `backend/export_onnx_ner.py` (torch.onnx legacy exporter, opset 17, dynamic batch and sequence axes; `quantize_dynamic`, QInt8, per_channel=False as the plan advises).
  Runtime: `backend/onnx_ner.py` wraps an `onnxruntime` session in a `PreTrainedModel` stub so the Hugging Face pipeline (tokenizer, label map, `aggregation_strategy="first"`) and the
  app's `_clean_entities` are unchanged. Selected with `NER_ONNX_DIR` and `NER_ONNX_FILE`.
- **Sizes:** fp32 431 MB (= 107,726,601 parameters x 4 bytes), INT8 108 MB.
- **Raw speed, 3 chunks, 4 threads:** PyTorch fp32 1,274 ms, ONNX fp32 862 ms (1.48x), ONNX INT8 796 ms (1.60x). PyTorch `quantize_dynamic`: 1.32x and 3 of 12 entities lost on a test text.
- **Accuracy gate (full test sets, paired bootstrap vs PyTorch fp32):**
  - ONNX fp32: **identical predictions** on WNUT-2017, BTC and FIN (difference exactly 0.000, CI [0, 0]).
  - ONNX INT8: WNUT-2017 +0.002 [-0.008, +0.012]; FIN -0.001 [-0.014, +0.011]; **BTC -0.012 [-0.018, -0.006], permutation p = 0.0004.**
    INT8 therefore **fails the plan's "<= 0.01 F1 lost on every set" gate** on BTC.
- **End-to-end latency (batch 1, 30 calls, 1 repeat):** short p50 230 -> 124 ms; medium p50 about 1,437-1,561 -> 861 ms, p95 about 1,858-2,003 -> 1,274 ms; INT8 medium p50 854 ms
  (no faster than fp32 here) with peak RAM 0.67 GB against 0.99 GB.
- **Decision:** recommend **ONNX fp32** (about 1.7-1.9x faster, identical output); INT8 only if memory matters. ONNX is opt-in (requires an export step).
- **Environment incident:** `pip install onnx` pulled protobuf 7.x, which broke three unrelated Google packages; reverted to protobuf 5.29.6 and kept `onnx` in an isolated `--target` folder
  used only by the export script (`ONNX_TOOLS_DIR`). A fresh virtualenv was rejected by Windows Application Control (a numpy DLL), so none is used.

### Step 6. Chunking versus overlapping windows
- **Observation:** the existing chunker already splits at sentence ends, so cuts inside a name are rare (abbreviations such as "N.A." or "St. Louis").
- **Change:** `NER_STRIDE_TOKENS` (Hugging Face stride, 3,000-character pieces) and `NER_CHUNK_CHARS` are configurable; defaults unchanged.
- **Result (documents of 15 consecutive test sentences, exact character-span F1, ONNX fp32):**

| documents | chunks 800 (default) | windows, stride 64 | chunks 300 |
|---|---|---|---|
| FIN (20 docs, ~3,500 chars) | 0.178 | 0.161 | 0.185 |
| WNUT-2017 (60 docs) | 0.542 | 0.510 | 0.556 |
| BTC (60 docs) | 0.307 | 0.279 | 0.303 |

  Windows were worse on all three sets and up to 25% slower; chunk size 300 vs 800 is within noise. **Not adopted.** Caveat: the documents join unrelated sentences.
  Open question: one-tweet-at-a-time BTC F1 is 0.401 but 0.307 in 15-tweet documents; the cause is not established.

### Step 7. Batching
- **Measurement (20,078-character document, 28 chunks, 4 threads):** PyTorch batch 1 / 4 / 8 / 16 / 32 = 15.3 / 16.7 / 17.8 / 18.4 / 17.7 s; sorting chunks by length made no difference
  (15.3 / 17.9 / 17.8 / 18.2 / 17.9 s), so padding is not the cause. ONNX fp32: 9.9 / 12.3 / 14.6 / 14.4 s for batch 1 / 4 / 8 / 32.
- **Change:** `run_ner` batch size now defaults to 1 (`NER_BATCH_SIZE`); it was 8. Together with ONNX a long document takes 9.9 s against 17.8 s before.

### Step 8. Result cache
- **Change:** `result_cache.py`, SQLite in `~/.cache/annotation-tool/results.sqlite` (outside OneDrive), key = sha256 of (schema version, task, model identity and settings, language(s), text).
  NER is cached per chunk with offsets stored relative to the chunk and shifted on reuse; translation per piece and only when the preferred provider produced it. Errors degrade to a miss;
  size-capped (`RESULT_CACHE_MAX_ROWS`); `RESULT_CACHE_PATH=off` disables it.
- **Result:** the same 20,078-character document: 16.0 s -> 57 ms; the same 14 sentences en->fr: 28.6 s -> 3 ms; results identical.

### Step 9. Model cache location and disk space
- **Change:** `HF_HOME` documented with model sizes; `disk_status()` logs a warning at startup and reports free GB in `/health/models` when below `MIN_FREE_GB` (default 3).

### Tests
Backend tests: 142 -> **183 (181 passed, 2 skipped)**. New: target tokens and Japanese model (4), preload and `/health/models` (7), threads (4), ONNX selection (3),
chunk/stride/batch settings (5), disk space (3), result cache (13). The ONNX adapter itself was tested manually against the real exported model, not in the unit suite.

### Problems met in WS1 and how they were handled
- A timing experiment overwrote the WS0 latency baseline file: result files for non-default thread counts now get a `__t<N>` suffix, and `--no-save` exists.
- A unit test meant to fake the model loader still tried to download a 1 GB multilingual model (the patch did not take effect), so the test run appeared to hang; the test now replaces
  `transformers` in `sys.modules`, and the partial download was deleted.
- `facebook/m2m100_418M` took 3.6 GB of the model cache and brought free disk to 1.2 GB; stopped, deleted, and recorded as not evaluated.
- Several long jobs hit the 10-minute foreground limit or a paused machine and were moved to the background by the tool; each was re-checked and re-run in the foreground where needed.
- Two of my own progress timestamps were not read from the clock; all are read from the clock again.

### Not done / limits
- Offline Japanese did not reach 25 chrF++ (17.9). Chinese and Arabic did not improve. zh/ja punctuation normalisation was noticed but not tried.
- ONNX is opt-in and the INT8 file fails the accuracy gate; the multilingual NER model has no ONNX build yet; roberta-large was not re-measured with ONNX.
- Latency numbers use 30 calls and one repeat (plan: 100 and 3); the long-text latency size (20,000 chars) was measured only as a whole-document time.
- Translation cache keys cover the beam setting and the Japanese model override only; other future translation settings must be added to the key.

### Decisions needed from the mentor (raised at the end of WS1, still open)
1. **Offline English to Japanese default model.** `Helsinki-NLP/opus-tatoeba-en-ja` (Apache-2.0) is the default: chrF++ 17.9, but 5 of 40 outputs were flagged as suspect (hallucinated
   Wikipedia-style text). `staka/fugumt-en-ja` scored 16.4 with only 1 of 40 suspect and visibly more coherent output, but is CC-BY-SA-4.0 (share-alike). Should the share-alike model become
   the default, or stay opt-in (`LOCAL_MT_JA_MODEL`)? Either way Azure (38.3) is the recommended route for Japanese.
2. **ONNX Runtime fp32 as the recommended NER runtime.** It gives identical predictions and is about 1.7-1.9x faster, but needs a one-time export step and an optional dependency, so it is
   opt-in. Should it be made the default (export performed at install or first start)? The INT8 file is not recommended (it lost 0.012 F1 on Twitter text).

### Problems it solves for the project
Cold-start latency (17.4 s -> 0.09 s), CPU inefficiency (1.7-1.9x faster NER with identical output, document NER about 1.8x faster, a harmful batching default removed),
repeat-work cost (cache), the broken offline Japanese model (4.4 -> 17.9), and disk-full failures (visible warning and a documented model location).


---

## WS2. English NER: large-model accuracy at small-model speed

Status 2026-11-16: steps 2.1-2.5 and 2.7 done (training on a free Colab T4, evaluation on the laptop). Step 2.6 (financial rules) not done. Results are in the section 'WS2 results' below.

### Step 2.1. Bake-off on the full test sets (strict F1 [95% CI]; paired bootstrap against bert-base-NER)

| model | WNUT-2017 (1,287) | BTC (2,000) | FIN (305) | p95, 2,000 chars, batch 1 |
|---|---|---|---|---|
| dslim/bert-base-NER (current) | 0.493 [0.459, 0.525] | 0.401 [0.382, 0.422] | 0.175 [0.127, 0.225] | 2,003 ms (PyTorch), 1,274 ms (ONNX fp32) |
| dslim/distilbert-NER | 0.427 [0.396, 0.459]; **-0.066 [-0.089, -0.043]** | 0.337 [0.318, 0.358]; **-0.064 [-0.076, -0.052]** | 0.182 [0.134, 0.234]; +0.007 (n.s.) | 1,117 ms |
| Jean-Baptiste/roberta-large-ner-english | 0.554 [0.522, 0.584]; **+0.062 [+0.031, +0.093]** | 0.581 [0.566, 0.596]; **+0.181 [+0.164, +0.196]** | 0.222 [0.139, 0.295]; +0.047 [-0.019, +0.106] (n.s.) | 6,371 ms |

- roberta-large is the clear accuracy leader and therefore the **teacher**; it is 3.0-3.3x slower than bert-base in PyTorch (822 s vs 275 s on BTC, 751 s vs 228 s on WNUT).
- A plain small model (DistilBERT) loses 0.06 F1 on WNUT-2017 and BTC, so "use a smaller model" does not remove the trade-off; bert-base through ONNX (861 ms) matches DistilBERT's
  speed (844 ms) with better accuracy, so ONNX bert-base is the speed bar a student must beat.
- Not run: bert-large-NER (full sets), spaCy `en_core_web_trf` (installing spaCy risks the environment again), GLiNER-small (left out on the owner's instruction to follow the plan
  document without the teacher's suggestions), TweetNER7.
- Mistake found and fixed: the WS1 result cache made a first DistilBERT latency run look like 3 ms (warm-up texts equal to timed texts); `evaluation/common.load_backend` now forces
  `RESULT_CACHE_PATH=off`.

### Step 2.2. Label harmonisation
`evaluation/label_maps.py`: one table mapping every dataset's types to PER, ORG, LOC, MISC (WNUT: person->PER, corporation/group->ORG, location->LOC, product/creative-work->MISC).
Finding: the label schemes differ between models and must never be assumed: roberta-large predicts **plain types without B-/I-** (O, PER, ORG, LOC, MISC), bert-base uses
O,B-MISC,I-MISC,B-PER,..., and distilbert-NER orders its BIO labels differently again.

### Step 2.3. Training mix (`training/build_mix.py`)
Train splits only: CoNLL-2003 (12,589 sentences after cleaning), BTC (6,312), WNUT-2017 (2,296), FIN (978) = **22,175 sentences**; dev mix (WNUT 1,004, BTC 993, FIN 150 = 2,147) for
checkpoint selection only. Leakage control: every training sentence whose normalized text or "skeleton" (letters and digits only, URLs and @mentions removed) occurs in any test set we
report on (CoNLL, WNUT, BTC, FIN, WikiANN test) was dropped: 310 exact + 16 near-duplicate CoNLL, 10 near-duplicate BTC, 11 FIN. Exact duplicates inside the training data were also removed
(1,126 in CoNLL, 99 in WNUT, 16 in BTC, 29 in FIN).
Consequence (as the plan notes): using BTC, WNUT and FIN training data makes those three test sets **in-domain** for the resulting student, so their scores are not comparable with the
zero-shot baselines. Experiment C (CoNLL-only) exists to give a fair zero-shot comparison.

### Step 2.4-2.5. Distillation code (`training/distill.py`, tested)
- Teacher labelling: per-word **type logits** (O, PER, ORG, LOC, MISC) at the first sub-word of each word, for three case variants (original, lower-case, upper-case). B-/I- teacher labels are
  added up per type, so teachers with either label scheme work.
- Student: 9 BIO labels. Loss = alpha * CE(gold BIO) + (1 - alpha) * T^2 * KL(teacher type distribution || student type distribution), with alpha 0.5 and T = 2; the student's type
  probability is the log-sum-exp of its B- and I- logits per type (temperature applied first). Boundaries therefore come from the human labels only.
  (The plan writes KL(student || teacher); the standard direction, teacher as the target, is used.)
- Case augmentation: per sentence and epoch, original 0.65, lower-case 0.30, upper-case 0.05 (the plan: lower 0.3, upper 0.05); the teacher scores the augmented text itself.
- AdamW, lr 5e-5, 10% warm-up with linear decay, batch 32, 4 epochs, gradient clipping 1.0, fp16 autocast on GPU, dev F1 every half epoch and best-checkpoint saving.
- Unit tests (7) cover label grouping for both teacher schemes, BIO span decoding, case variants, the loss (matched student small, wrong student large, B vs I identical under the
  teacher term, alpha = 1 equals plain cross-entropy, no NaN when a batch has no gold word). A NaN bug for a batch without labelled words was found by these tests and fixed.
- End-to-end smoke tests on the laptop CPU (teacher dslim/bert-base-NER, student distilroberta-base): the pipeline runs, and on 480 sentences / 90 steps the loss fell 1.73 -> 0.47 and
  WNUT dev F1 rose 0.37 -> 0.45 -> 0.53. The saved student loads and predicts through `evaluation/run_all.py`. The DeBERTa-v3-small tokenizer maps every word correctly.
- **Why Colab:** measured CPU training speed on this laptop: DistilBERT 4.6 s, DistilRoBERTa 4.4 s, DeBERTa-v3-small 7.9 s per batch of 16, i.e. 107 / 102 / 183 minutes per epoch over the
  22,175 sentences; teacher labelling at about 2.4 sentences/s would take about 2.5 hours. A full distillation would need 8+ hours of laptop time, so (the owner's decision) it runs on a free
  Colab T4 GPU: `WS2_Colab/WS2_Distillation_Colab.ipynb` + `colab_bundle.zip` (distill.py, colab_runner.py, train and dev mixes), four experiments: A distilroberta + teacher (main),
  B distilroberta without teacher (ablation), C distilroberta trained on news only (zero-shot control), D deberta-v3-small + teacher.
- Differences from the plan: no extra unlabelled in-domain text (the plan suggests 100k-300k sentences; none is available without licence questions), so the teacher labels the 22,175 gold
  sentences only; one seed by default (the notebook has an optional cell for seeds 1 and 2).


### WS2 results (added after the Colab run, 2026-11-16)

**Training run.** `WS2_Colab` notebook on a free T4: four experiments, 22 minutes in total (D alone 7.8 minutes, so A, B and C together about 14), one seed, 4 epochs, batch 32, lr 5e-5.
Dev-set mean F1 at the best checkpoint (WNUT/BTC/FIN dev, used only to choose checkpoints): A 0.678, B 0.815, C 0.526, D 0.747.

**Test-set results** (strict F1 [95% CI], full splits; table with intervals and the speed table are in `evaluation/RESULTS.md`, section "WS2"):

| model | WNUT-2017 | BTC | FIN | status of the three sets for this model |
|---|---|---|---|---|
| bert-base-NER (default) | 0.493 | 0.401 | 0.175 | zero-shot |
| roberta-large (teacher) | 0.554 | 0.581 | 0.222 | zero-shot |
| A distilled, in-domain | **0.558** | 0.630 | 0.296 | in-domain |
| B no teacher, in-domain | 0.490 | 0.808 | 0.759 | in-domain |
| C distilled, CoNLL only | 0.523 | 0.565 | 0.206 | zero-shot |
| D deberta distilled, in-domain | 0.549 | not run | 0.342 | in-domain |

**Findings**
1. **Memorisation inflates the in-domain scores.** `evaluation/overlap_check.py`: 2.3% of WNUT test entities, 41.7% of BTC and 73.6% of FIN (91% of FIN person names) also occur as entities in the training
   data. B's BTC 0.808 and FIN 0.759 therefore mostly reflect memorised names and contract boilerplate. The sentence-level duplicate filter removed exact and near-duplicate sentences but cannot remove
   repeated entities. Only WNUT-2017 (new names) and the zero-shot student C are fair tests of generalisation.
2. **The teacher is what generalises.** On WNUT-2017: A 0.558 and D 0.549 match the teacher (0.554, A-teacher +0.004 [-0.018, +0.025]); B (identical data, no teacher) scores 0.490, i.e. no better than bert-base.
   B is best on the dev set (0.815) because the dev sets resemble its training data; the dev set is a misleading selector for generalisation here.
3. **Zero-shot distillation works.** C, trained only on CoNLL news with teacher soft labels and case augmentation, beats bert-base on Twitter by +0.164 [+0.146, +0.180] and is within 0.017-0.031 of the
   teacher, at about half of bert-base's compute. (An ablation "C without teacher" was not run, so the contribution of the teacher versus case augmentation in this setting is not separated.)
4. **Speed.** Student A: 956 ms p95 in PyTorch (bert-base 2,003 ms), 670 ms with ONNX fp32 (3.0x faster than today's default, 9.5x faster than the teacher), 0.74-0.88 GB RAM. ONNX predictions are identical to
   PyTorch for this student (difference 0.000 on WNUT and FIN).
5. **DeBERTa (D)** is no clearly better than DistilRoBERTa (WNUT 0.549 vs 0.558, FIN 0.342 vs 0.296 in-domain) and about 1.5x slower on CPU (FIN 107 s vs 71 s), so DistilRoBERTa is preferred.

**Gate (WS2.7).** Part (a): A is above bert-base on all three sets with CIs above 0 (WNUT +0.065 [+0.033, +0.096], BTC +0.230, FIN +0.121). Part (b): p95 956 ms <= 2,003 ms. **A passes.**
C passes on BTC (+0.164), is borderline on WNUT (+0.031 [0.000, +0.062]) and not reliable on FIN (+0.030 [-0.024, +0.080]).

**Decision taken: not made the default.** The app gets `NER_MODEL_EN` (a model id or local folder) so the student can be tried, but the default stays bert-base-NER because (i) publishing or bundling a 316 MB model
and its licence questions (CoNLL-2003 text is Reuters-copyrighted, the FIN dataset's original licence is unverified, distilroberta-base is Apache-2.0, the teacher is MIT) need the mentor's decision,
(ii) accuracy outside tweets, WNUT-style text and finance text is unmeasured. Model card: `evaluation/MODEL_CARD_student_A.md`.

**Step 2.6 (ORG-suffix and defined-term rules for financial text): not done.** FIN's test set is 73.6% memorisable, so any rule gain cannot be measured cleanly there; the plan's in-house gold set (section 8.3)
is the right test and does not exist yet.

**Problems met in WS2 and how they were handled**
- Disk filled to 0.38 GB after downloading 1.5 GB of models plus the zip; five experiment-only caches (deberta-v3-small, distilroberta-base, opus-mt-en-fr, distilbert-NER, fugumt, about 1.9 GB) were deleted, and the
  models were placed outside OneDrive. The Recycle Bin (1.6 GB) was left untouched.
- The teacher label scheme differs from the student's (plain types versus BIO) and distilbert-NER orders its BIO labels differently: handled by distilling type probabilities (B- and I- added up) and learning boundaries from gold labels.
- The WS1 result cache made a DistilBERT timing look like 3 ms: evaluation scripts now force the cache off.
- A NaN in the loss for a batch without gold words was found by unit tests and fixed.
- Two result files would have shared the name `best`: `run_all.py` and `bench_latency.py` accept `name=path`.
- A DeBERTa evaluation run reported 8,682 s because the laptop was suspended mid-run; that timing is invalid and not used.
- Colab showed a harmless pip dependency warning (gradio / diffusers) and the quick check, the real run and the Drive copy all worked on the first attempt.

**Not done / limits.** One seed only (the plan asks for three; a notebook cell exists); no extra unlabelled in-domain text (the plan suggests 100k-300k sentences); D not run on BTC and not timed; student C not timed with ONNX;
INT8 of the student not evaluated for accuracy; the Pareto plot labels overlap slightly; accuracy beyond the three domains is unmeasured.

**Problems it solves for the project.** It shows that a small model can reach the accuracy of the large one on unseen names at 3x the speed of today's default, that the earlier "speed versus accuracy" choice
(question 1 to the mentor) was a false choice, and, equally important, it shows how an easy-to-compute benchmark number (BTC 0.81, FIN 0.76) can be mostly memorisation, which changes what must be built next
(an in-house gold set built from realistic documents).


---

## WS3. Hindi and other-language NER

Status 2026-11-18 12:50 (WS3 CLOSED for the agreed scope; see the closing entry at the end of this section's updates). Earlier status line: Hindi and the regional-language scope are done and measured (IndicNER, indicbert-hi, GLiNER multi and GLiNER-X on Hindi and 10 other Indian languages; per-language status system in the app; open-licence Hindi default).
**Still open from the WS3 brief:** Hinglish fix and test set (3.4, unmeasured); NER measurement for Russian, Arabic and Japanese (no UNER PUD / ANERcorp / stockmark runs; `uner_all` is 4.4 GB); latency protocol E3 on Hindi text; a licence-clean model for the seven non-Indian languages (GLiNER-X base is the measured candidate for de/fr/es, not yet wired); Odia and Assamese (no usable open model); live end-to-end test of the app with regional languages (blocked by RAM).
(Original 2026-11-17 status, now superseded: IndicNER was not yet measured; 3.5 fine-tuning not needed; 3.7 had numbers for de, es, fr, zh only.) Full tables: `evaluation/RESULTS.md`, section "WS3".

### 3.1 Test data (frozen in `MANIFEST.json`, scored through `run_all.py` via new `LOCAL` datasets in `eval_ner.py`)
- Naamapadam Hindi (ai4bharat, CC0): the **gold** test split, 867 sentences (864 after dropping 3 empty ones), 1,929 entities (PER, ORG, LOC); the validation split is also kept. The 492 MB silver training split is not kept.
- HiNER-collapsed (cfilt, CC-BY-SA-4.0): test split 21,657 sentences, a fixed deterministic subset of every 7th sentence (3,094 sentences, 7,840 entities); the full set would take about an hour per model on this CPU.
- MultiCoNER v2 (CC-BY-4.0) dev splits for de, es, fr, zh, hi (about 510-860 sentences each). Its 33 fine-grained types are mapped as the plan says: all person subtypes to PER, group and corporation types to ORG, location types to LOC, the rest to O.
  Caveat: the benchmark is lower-cased, unpunctuated, short and low-context on purpose, so it is a stress test. Universal NER has no Hindi and was not used.

### 3.3 Hindi results (strict F1 with 95% CI; `neuronsbyisshu/indicbert-hi-ner-naamapadam`, CC0, BERT with a 250k-token multilingual vocabulary, 7 BIO labels, versus the previous `Davlan/xlm-roberta-base-wikiann-ner`)

| test set | previous | new | difference |
|---|---|---|---|
| Naamapadam gold (the new model's home dataset) | 0.650 [0.626, 0.675] | 0.827 [0.807, 0.847] | +0.177 [+0.155, +0.198] |
| HiNER (independent) | 0.527 [0.516, 0.537] | 0.755 [0.746, 0.763] | +0.228 [+0.218, +0.238] |
| MultiCoNER v2 hi dev (independent, hard) | 0.421 [0.381, 0.466] | 0.601 [0.555, 0.640] | +0.179 [+0.134, +0.218] |

ORG is the weakest type for both (new: 0.73, 0.64, 0.49 on the three sets); the previous model's wrong-boundary share on HiNER was 24% against 6%. The 0.827 equals the score reported on the model's card, which is a reproduction check.
The first attempt to load the model failed (`Tokenizer class TokenizersBackend does not exist`): its tokenizer was saved by a newer transformers than the pinned 4.57.6. The fix builds a `PreTrainedTokenizerFast` directly from `tokenizer.json` and the special tokens in `tokenizer_config.json`
(`eval_ner.fast_tokenizer`, `main._plain_fast_tokenizer`), so the environment was not upgraded.

### 3.4 Hinglish
Five realistic sentences: `langdetect` said Indonesian x3, Swahili and English (29%), never Hindi; the same sentences in Devanagari were detected as Hindi with probability 1.00. Entities were partly found; "Rahul Mumbai" was merged and "Priya" missed.
A sample of five, not a test set. The plan's remedy (transliteration with IndicXlit, then the Hindi model) was not built (heavy install in an environment that was already broken once by a dependency upgrade); no hand-labelled Hinglish set exists. Recorded as a known limitation.

### 3.6 Replacing the Hindi model in the app
`NER_MODEL_HI` (default: the new model; any id or folder) and `ner_model_name(lang)`; Hindi no longer uses the unlicensed Davlan model. `THIRD_PARTY_NOTICES.md`, the README (Hindi numbers, limitations), `project_documentation.md` and `.env.example` updated.
**Correction (2026-11-17, found while planning the next step):** this paragraph originally said the previous model also tagged DATE, so Hindi would lose date highlighting. That was wrong. The previous model's output head has B-/I-DATE labels, but its saved predictions contain **0 dates on 4,472 Hindi sentences** (Naamapadam 864, HiNER 3,094, MultiCoNER 514) and 2 dates on 2,737 German, Spanish and French sentences, so dates were effectively never detected. Nothing was lost; the new model also tags PER, ORG and LOC only. `NER_MODEL_HI` still allows going back to the old model.
The model is 1.1 GB and downloads on first use like the others. The NER result cache's format version was bumped to 2 so no stale entry (including old Chinese results) is reused.

### 3.7 Other languages (the app's multilingual model, MultiCoNER v2 dev, strict F1): German 0.341, Spanish 0.310, French 0.264, Chinese 0.281 (after the fix); stress-test numbers that underestimate ordinary text. No test set for Russian, Arabic or Japanese; no alternative model evaluated
(`uner_all` is 4.4 GB; GLiNER-based candidates excluded by the owner's instruction), so **the Davlan model, which has no declared licence, is still used for every language except English and Hindi** (the release blocker remains for those).

### A bug found by the multilingual measurement: Chinese and Japanese entities spanned whole clauses
MultiCoNER Chinese scored strict F1 **0.000** (partial 0.165). Two causes in the app's own code: (1) Chinese and Japanese have no spaces, so the tokenizer's single "word" is the whole clause and aggregation `first` labels all of it; (2) `_snap_to_token` then grew each entity
to the nearest space or punctuation mark. Fix: aggregation `simple` and no snapping for zh and ja (`NO_SPACE_LANGS`). Result: 0.020 with (1) alone, **0.281** [0.241, 0.317] with both. The app's real Chinese and Japanese highlighting was affected, not only the benchmark.
Japanese receives the same fix but there is no Japanese test set, so its effect is unmeasured. Three unit tests pin the behaviour.

### Tests
Backend tests: 183 to **191 (189 passed, 2 skipped)**: model selection per language, Hindi warm-up, tokenizer fallback (and that other errors are not swallowed), snapping off for unspaced scripts, grouping mode per language.

### Problems met in WS3 and how they were handled
- Disk fell to 0.47 GB (model downloads, the extracted 492 MB Naamapadam training split, Windows swap): deleted the training split at once, the roberta-large teacher cache (1.35 GB) and the zh/ar/en-jap translation caches (1.2 GB, re-downloadable).
- A segmentation fault while loading weights, once, under low memory (1.2 GB free of 7.3 GB); the identical command succeeded on retry. Memory and disk are tight on this laptop.
- Two hand-written code strings contained raw newlines (a tool-quoting trap seen before); fixed with the edit tool.
- The first evaluation of the Naamapadam files exposed three empty sentences that a pipeline cannot read; they are dropped and documented (864 of 867 used).

### Not done / limits
IndicNER (needs login); the two HiNER-trained models; fine-tuning; the Hinglish fix and test set; any NER measurement for Russian, Arabic and Japanese; a replacement for the Davlan model for the seven non-English, non-Hindi languages; the in-house gold sets for English and Hindi (plan section 8.3, still pending the mentor's decision).
MultiCoNER dev (about 500-850 sentences per language) gives wide intervals (about +-0.03-0.04).

### Problems it solves for the project
Hindi, the project's core goal, now has measured accuracy (0.83 / 0.76 / 0.60 on three datasets), a much better model (+0.18 to +0.23 F1) with a clear CC0 licence that removes the release blocker for Hindi, and a faster one; the app no longer highlights whole clauses for Chinese;
and the multilingual claims in the README are replaced by measured, caveated numbers.


### Indian regional languages: assessment (2026-11-17, requested by the owner after WS3)

**Question:** are Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Odia and Assamese supported? **Answer: not exposed; partly working underneath.**
- App language lists (frontend `HomeScreen.tsx`, `HistoryScreen.tsx`; backend `local_translator.SUPPORTED_LANGS`): English, Hindi, French, Spanish, German, Chinese, Arabic, Japanese, Russian only.
- `langdetect` has profiles for hi, bn, ta, te, mr, gu, kn, ml, pa, ur, ne; **not for Odia or Assamese**.
- The multilingual NER model (Davlan) lists as, bn, gu, mr, pa, ur and hi among its 20 training languages (model card); not ta, te, kn, ml, or. It also does not list de, ja or ru, which the app offers. Its licence is undeclared.
- Azure Translator translated a test sentence correctly into bn, ta, te, mr, gu, kn, ml, pa, or, as, ur, ne and hi (13 calls, one transient network error on the first Hindi call, retried successfully).
- Offline translation: `indic_processor._FLORES_CODES` already covers about two dozen Indian language codes and the pinned IndicTrans2 rotary en-indic / indic-en models are multi-language (22 scheduled languages per the model family; not tested here); the app enables only en-hi.
- Chunking: an 800-character chunk is 203-330 XLM-R pieces in every Indian script (Assamese highest), under the 512 limit.

**Measured NER baseline** (Davlan multilingual model, Naamapadam human-annotated test sets, CC0, frozen in `MANIFEST.json` as `nm_<lang>`, first 300 sentences per language, strict F1 [95% CI]):
Malayalam 0.687 [0.627, 0.737]; Gujarati 0.652 [0.601, 0.702]; Kannada 0.635 [0.579, 0.681]; Telugu 0.629 [0.577, 0.679]; Marathi 0.617 [0.559, 0.669]; Bengali 0.559 [0.499, 0.615]; Tamil 0.517 [0.466, 0.565]; Punjabi 0.480 [0.413, 0.537];
Assamese 0.196 [0.053, 0.320] (51 sentences); Odia 0.140 [0.104, 0.175]. Tamil, Telugu, Kannada and Malayalam work although unlisted (transfer from related languages); Odia and Assamese do not.
Test-set sizes: as 51, bn 607, gu 1,076, kn 1,019, ml 974, mr 1,080, or 994, pa 993, ta 758, te 847.

**Suggested plan** (see `Indian_Regional_Languages_Assessment.pdf`): (1) measure `ai4bharat/IndicNER` (MIT, as/bn/gu/hi/kn/ml/mr/or/pa/ta/te; gated, needs the owner's Hugging Face login) on the ten frozen sets; (2) expose the languages with per-language quality labels and a language-to-model table;
(3) script-based detection for Odia and Assamese; (4) enable IndicTrans2 targets and compare with Azure on FLORES+/IN22; (5) only if needed, fine-tune or distil from IndicBERTv2 (MIT, 24 languages) with the WS2 Colab recipe; (6) transliteration (IndicXlit) for romanised input; (7) native-speaker test sets for priority languages.
Open questions to the owner/mentor: which languages are the priority, the Hugging Face login, whether a 'beta' label is acceptable.


### Indian regional languages: first open-licence measurements (2026-11-17, after the owner's priorities: Hindi first, everything open source)
Owner decisions: the mentor wants regional-language support with open-source models; Hindi is the top priority; the order of the other languages is mine. The owner created the Hugging Face login (read token, user `muskan-424`; the token was never shown in the chat).
Full table in `evaluation/RESULTS.md` ("Indian languages: open-licence NER candidates"). Summary (strict F1, 300 sentences per language): `ai4bharat/IndicNER` (MIT) beats the unlicensed old model on Bengali (+0.158), Gujarati (+0.084) and Tamil (+0.101), ties on Telugu, and is **worse on Marathi (-0.112)**
and on Hindi (0.704 against 0.827 for the CC0 Hindi model). **The Hindi-only CC0 model scores 0.792 on Marathi (+0.175 over the old model)**: zero-shot transfer through the shared Devanagari script. Not yet measured with IndicNER: Kannada, Malayalam, Punjabi, Odia, Assamese.
**Correction of an earlier statement:** the WS3 text that Hindi "loses date highlighting" was wrong; the old model predicted 0 dates on 4,472 Hindi sentences (documents corrected, see the correction note in WS3, 3.6).
Operational problems: IndicNER's weights download stalled twice (xet transfer, then a slow route) and succeeded with `HF_HUB_DISABLE_XET=1`; the laptop was overloaded (CPU 100%, under 1 GB RAM free; VS Code background processes), the machine suspended during one run (a 21,410 s timing artefact), and the evaluation of the remaining five languages was not run.
Proposed language order (owner asked for my arrangement): Hindi, Marathi (now 0.79 with no new model), Bengali, Tamil, Telugu, Gujarati, Kannada, Malayalam, Punjabi, Odia, Assamese; beta-label thresholds (Stable F1 >= 0.75 with CI floor >= 0.70 on >= 500 human-labelled sentences, Beta 0.50-0.75, Experimental < 0.50) proposed and awaiting the owner's and mentor's agreement.


### Colab evaluation package and the per-language status table (2026-11-17, evening)
- **Beta thresholds agreed by the owner** and implemented in `backend/language_status.py`: stable = strict F1 >= 0.75, lower end of the 95% range >= 0.70, >= 500 human-labelled sentences, open licence, and a test made by native speakers; beta = 0.50-0.75 or any
  condition for stable still missing (the reasons are listed); experimental = F1 < 0.50; blocked = no open licence. `evaluation/make_language_status.py` computes the table from the evaluation results (best open-licence model per language; the run file with the most sentences wins)
  and writes `backend/language_status.json`; `GET /api/languages` serves it (public). The frontend badge is not built yet. 14 tests (rules, edge cases, endpoint, loader, consistency of the shipped file).
  Current table (before the Colab run): Hindi beta (0.827, n=864; only "no native-speaker test" is missing), Marathi beta (0.792, n=300, Hindi model), Bengali beta (0.716), Gujarati beta (0.736), Tamil beta (0.617), Telugu beta (0.623); Kannada, Malayalam, Punjabi, Odia and Assamese "blocked"
  (only the unlicensed model has been measured). `NATIVE_TEST` in the generator is False for every language, deliberately conservative: whether Naamapadam's annotators count as native speakers is the mentor's call.
- **Colab evaluation notebook** (`major/WS3_Colab/WS3_Evaluate_Models_Colab.ipynb` + `colab_eval_bundle.zip`, built by `training/make_eval_bundle.py`): scores IndicNER, the Hindi CC0 model and the old multilingual model on the full test sets of Hindi and ten more Indian languages (13 test sets) on a GPU;
  the Hugging Face token is read from a Colab secret, never typed. To make this possible the entity clean-up code moved from `main.py` into `backend/ner_clean.py` (no web imports; `main.py` imports it, behaviour unchanged, all tests pass), `run_all.py` loads the web app only for `app`,
  and `eval_ner.make_pipeline` uses a GPU when present. Verified in a clean folder with no web app: the Assamese test set scored through the bundle (the Hindi model scored 0.467 [0.300, 0.609] on Assamese against 0.196 for the old model; 51 sentences).


### Indian languages on the full test sets (Colab GPU, 2026-11-17) - results
Notebook `WS3_Evaluate_Models_Colab.ipynb` run by the owner (Hugging Face token supplied as a Colab secret; results returned as `ws3_runs.zip`, 39 prediction files, copied into `evaluation/runs/`). The old model's Colab scores equal the laptop's (reproducibility check). Full table: `evaluation/RESULTS.md`, section "Indian languages on full test sets".
- **The CC0 Hindi-only model `indicbert-hi-ner-naamapadam` is the best open-licence model on Hindi and eight other languages**: Marathi 0.831, Bengali 0.808, Tamil 0.750, Telugu 0.854, Gujarati 0.825, Kannada 0.826, Malayalam 0.847, Punjabi 0.732 (strict F1, 600-1,080 sentences each), +0.10 to +0.29 over IndicNER (MIT) and ahead of the unlicensed old model everywhere.
  Its model card reports the same zero-shot numbers for bn / ta / te (0.808 / 0.751 / 0.853); this run gives 0.808 / 0.750 / 0.854. Transfer works through IndicBERTv2's pretraining on 24 Indian languages.
- **Odia: unusable** (best 0.271, IndicNER; indicbert 0.167; old 0.127). **Assamese:** IndicNER 0.500 [0.291, 0.732] on 51 sentences, not reliably different from the Hindi model (0.467).
- **Per-language status table regenerated** (`backend/language_status.json`): Hindi, Marathi, Bengali, Tamil, Telugu, Gujarati, Kannada, Malayalam beta (the only missing condition is "no test made by native speakers yet"), Punjabi beta (F1 0.73 < 0.75 as well), Assamese beta (0.50, 51 sentences, wide range), Odia experimental. No language is "stable" until the native-speaker condition is met.
- **Caveats recorded:** in-domain test sets (Hindi falls from 0.827 on Naamapadam to 0.755 on HiNER and 0.601 on MultiCoNER; the other languages have no independent test); silver training labels; ORG is the weakest type (0.57-0.75); offline translation into these languages unmeasured.
- **Recommendation to the owner / mentor:** (1) add an *independent test set* condition to "stable" (the Hindi spread shows why); (2) decide whether Naamapadam's annotators count as native speakers; (3) offer the eight beta languages in the app (language list, one NER model for all of them, status badges), keep Odia hidden, treat Assamese as experimental until a larger test set exists.


### Regional languages offered in the app + GLiNER prepared (2026-11-18)
Owner's decisions (after reading `NEXT_STEPS_WS3-WS8.md`): regional languages first; keep the Hindi model (indicbert, CC0) as the Hindi default; GLiNER may be evaluated again (it had been excluded earlier on the owner's instruction).
- **Backend:** `INDIC_NER_LANGS` (hi, mr, bn, ta, te, gu, kn, ml, pa) route to the one open model; Odia and Assamese stay on the old model and are not offered. Translation errors now say what to do (offline translation exists only for Hindi; the others need Azure). `/api/languages` adds
  `offline_translation` and `azure_configured`. `evaluation/make_language_status.py` marks the nine as offered. Language detection check on real test sentences: 99-100% correct for all nine even for single sentences; Odia has no profile in the detector (it raises), so Odia and Assamese would need a script-based fallback before they could be offered.
- **Frontend:** `src/utils/languages.ts` (names, API loading, badge and note logic) and badges plus a quality note in `HomeScreen.tsx`; `HistoryScreen.tsx` shows the new names. `tsc --noEmit` passes (exit 0); the pure logic was run against the real `/api/languages` data with node (nine languages with beta badges, Odia and Assamese hidden, Marathi note, Bengali Azure note).
  Not built: export columns carrying the language status, and a UI test runner (the frontend has none).
- **GLiNER (evaluation only, Colab):** `evaluation/gliner_adapter.py` makes a GLiNER model look like a pipeline (labels person / organization / location, default threshold 0.5, untuned); `run_all.py --models "name=gliner:<model id>"`; 5 unit tests with a fake `gliner` module. Notebook `WS3_Colab/WS3b_GLiNER_Colab.ipynb`
  scores `knowledgator/gliner-x-base`, `knowledgator/gliner-x-small` and `urchade/gliner_multi-v2.1` (Apache-2.0) on 17 test sets (the Indian ones, Hindi, and de / es / fr / zh MultiCoNER). Not yet run. The `gliner` package is installed only in Colab (the brief's rule 8).
- **Tests:** backend 205+ passing (new: routing, translation error messages, endpoint enrichment); evaluation tests 5 (GLiNER adapter).
- **Not verified live:** an end-to-end run of the app with a Marathi, Bengali and Tamil sentence crashed twice with a segmentation fault while loading the model (0.6-1.6 GB RAM free, 0.6 GB disk free); the model file passes size and sha256 checks, so this is the laptop's memory, not the model.
  The routing is covered by unit tests and the model's accuracy was measured through the evaluation path.
- **Constraints recorded against the brief:** Rule 10 (5 GB free) cannot be met (0.6 GB free). Rule 5: the Hindi default was changed in WS3 before the brief existed; the owner chose to keep it.


### 2026-11-18 - E1 (seen / unseen), annotation artefacts, GLiNER results, and CORRECTIONS to WS2
- **Goal:** brief task A1 / E1: report accuracy on entities a model could have memorised versus entities it has not seen; add GLiNER as a candidate.
- **What was run:** `python evaluation/seen_unseen.py --table english | english-common | english-names | indic` on the saved prediction files (no model was re-run); `evaluation/overlap_check.py` extended; GLiNER-multi run on Colab by the owner (`ws3b_runs`, copied into `evaluation/runs/`, none overwritten). Tables: `evaluation/RESULTS.md`, section "E1".
- **Findings:**
  1. **Annotation artefacts dominate the earlier "seen" figures.** 31% of the Twitter set's gold entities are a bare `@` and 63% of the finance set's are the words "Borrower" and "Lender". Excluding them, the share of gold *names* seen in training is WNUT 2.3%, BTC 14.8%, FIN 28.7% (115 names), not 41.7% / 73.6%.
  2. **Names-only scores** (artefacts removed from gold and predictions): bert-base 0.493 / 0.511 / 0.244; teacher 0.554 / 0.719 / 0.369; student A 0.565 / 0.759 / 0.378; **student B (no teacher) 0.595 / 0.787 / 0.476**; student C 0.524 / 0.691 / 0.342 (WNUT / BTC / FIN).
  3. **Indian languages:** for the seven languages in scripts other than Devanagari there is 0-2% overlap with the Hindi training entities, so their scores are on unseen names (cross-lingual transfer is real). Hindi itself: 78% of test entities are seen and the unseen-subset F1 is 0.56 (Naamapadam) and 0.48 (HiNER) against 0.83 and 0.76 overall.
  4. **GLiNER-multi (Apache-2.0):** 0.03-0.26 on Indian languages and 0.023 on Chinese; **0.55-0.65 on German, Spanish and French against 0.26-0.34 for the unlicensed model** - a candidate licence-clean replacement for those three languages (threshold untuned; MultiCoNER overlap with its training data unchecked). `gliner-x-base` and `-small` failed in Colab; retry notebook `WS3c_GLiNER_X_retry_Colab.ipynb` prepared (installs `gliner[stanza]` and sentencepiece, shows the full error).
- **CORRECTIONS (statements made earlier in this log, in RESULTS.md, the mentor report of 16 November, the plain-language Part 3 guide and the student A model card):**
  - "41.7% of BTC and 73.6% of FIN test entities also occur in training" / "B's BTC 0.808 and FIN 0.759 are largely memorised names": the percentages counted `@` and "Borrower"/"Lender" as seen names. B's inflated scores come mainly from **labelling conventions**, not memorised names (finance F1 0.759 falls to 0.476 without the role words).
  - "The teacher is what generalises (B, with no teacher, is no better than bert-base on WNUT)": **not supported.** B's low WNUT score came from predicting `@`, which WNUT does not label; on names only B is the best of all models on all three sets, and distillation (A) did not beat plain fine-tuning (B) on the same data.
  - Earlier note that other Indian languages' scores are "likely lower on independent text" is partly superseded: for seven languages the test entities are unseen by construction; what is still unmeasured is robustness to other text domains and labelling guidelines.
  - The WS2 gate decision (student A beats bert-base on every set) is unchanged; the choice of student for the default is reopened (see item 4 of the E1 section in RESULTS.md).
- **Not done:** the E1 table for UNER English-EWT (E5: needs the dataset loader and the models, two of which need downloads the disk could not take until the Recycle Bin was emptied); seeds for student B; GLiNER-X results.
- **Tests:** backend 205+ passing; evaluation tests 12 (5 GLiNER adapter, 7 seen/unseen).
- **Housekeeping:** the disk reached 0 GB free during the Naamapadam stream (the Hindi training zip, 82 MB, failed to download); recovered by deleting the IndicNER cache (667 MB) and scratch folders; the owner then emptied the Recycle Bin (1.6 GB freed, 2.6 GB free).

### 2026-11-18 01:35 - GLiNER-X retry results (Colab T4, `ws3c_runs`)
- **Ran:** `knowledgator/gliner-x-base` and `-small` (Apache-2.0, trained on synthetic multilingual data, per model card) zero-shot with the untuned 0.5 threshold, on the Naamapadam sets, HiNER and MultiCoNER (hi, es, fr, de, zh). Full table: `evaluation/RESULTS.md`, section "GLiNER-X".
- **Result:** 0.57-0.79 strict F1 on Hindi, Bengali, Gujarati, Kannada, Malayalam, Telugu and Punjabi without any Indian-language training, against 0.03-0.14 for GLiNER multi. Still 0.05-0.15 below `indicbert-hi` on every one of those, so the default stays `indicbert-hi`. Odia 0.13 and Assamese 0.33-0.38 (51 sentences): unsolved.
- **Failed:** Marathi and Tamil for both sizes (`TypeError`, message not captured in the Colab output); German and French for base. Cause not diagnosed; to be reproduced on Colab with a full traceback.
- **Decision impact:** none on the app. GLiNER-X is recorded as a second open-licence candidate, mainly relevant if the mentor requires a model that was not trained on Hindi data, or for languages where `indicbert-hi` is weak.
- **01:46 - cause of the Marathi/Tamil crash found** (traceback from a Colab diagnostic cell): for 113 of 1,060 Marathi and 57 of 758 Tamil sentences GLiNER-X returned an entity with `start`/`end` = `None`, and the adapter's `int(None)` raised `TypeError`. Single-sentence checks on Marathi and Tamil text worked, so it is not a language problem. Fix in `evaluation/gliner_adapter.py`: recover the span by locating the entity's own text in the sentence (first occurrence not already taken); entities that cannot be found are dropped and counted (`recovered`, `unresolved`, printed by `run_all.py`). New unit test (13 evaluation tests pass). New notebook `WS3d_GLiNER_X_fix_Colab.ipynb` reruns only the four crashed runs (Marathi and Tamil for both models, German and French for base); the bundle was rebuilt. The recovered spans are an approximation (repeated entity texts are assigned in order), and the note line in the Colab output reports how many were affected.
- **01:59 - WS3d rerun done (`ws3d_runs`).** After the adapter fix: Marathi 0.631 / 0.582, Tamil 0.658 / 0.623 (GLiNER-X base / small); German 0.592 and French 0.653 for base. Paired bootstrap against GLiNER multi: German +0.040 [+0.007, +0.073], French +0.045 [+0.021, +0.069], Spanish +0.021 [-0.005, +0.048]; so GLiNER-X base is the better open-licence candidate for de/fr (Spanish: no reliable difference). Against indicbert-hi it is lower on every Indian language (Marathi -0.199, Tamil -0.092). About 3% of the entity spans that came back WITHOUT offsets had to be dropped (span not found) in the four reruns: 0.14-0.93% of all predicted entities per run, worth at most +0.001 to +0.007 F1 (correction of wording, 2026-11-18 13:05). Hindi default unchanged. No app change made.

### 2026-11-18 02:18 - WS3 completion, item 1: Hinglish (Hindi in English letters), stage 1 of 2
- **Goal:** the brief's step 3.5 / log 3.4: detect Romanized Hindi and find the best way to read it; the earlier work had five sentences and no test set.
- **Test set built:** `evaluation/hinglish/hinglish_sentences.txt` (125 sentences; see the correction below: 182 entities, 11 without entities), frozen in `MANIFEST.json` as dataset `hinglish` by `evaluation/make_hinglish_set.py`. **Written by the assistant, not by native speakers**, so every number from it is provisional until a Hindi speaker reviews the labels. Test use only. It is entity-dense and news-like; real chat text has more function words.
- **Detector (`backend/hinglish.py`, not yet wired into language detection):** a text is Romanized Hindi when at least 85% of its letters are Latin and at least 2 words, and at least 20% of all words, come from a list of unambiguous Hindi function words (hai, ka, ki, mein, nahi, kya ...). Words that are also English/Spanish/French/German words (to, the, me, main, par ...) are excluded. Thresholds were fixed before scoring.
  - First version of the word list: recall 100/125 (80.0%) on the test set; 0 false positives in 7,010 negatives (English tweets BTC and WNUT, English Wikipedia, German, Spanish, French).
  - After looking at the 25 misses I added about 40 more function words (ne, ji, naya, liya, kaam ...). **That change was tuned on the same test set**, so 122/125 (97.6%) is an optimistic figure. To check, 40 fresh sentences (`hinglish_detect_holdout.txt`) were written before the extension was run: 39/40 with the first list, 40/40 with the extended list. False positives still 0/7,010. langdetect called 0 of the 125 Hinglish sentences Hindi.
  - Limits: single-marker short messages (for example "kya haal") are not detected by design; text of mixed English and Hinglish with few function words is missed; Roman-script Marathi, Bengali etc. is not handled.
- **Candidate fixes prepared, not yet scored:** `evaluation/make_hinglish_deva.py` transliterates each word to Devanagari with Azure Translator (word by word, so the tags stay aligned; only the invented test sentences were sent) into dataset `hinglish_deva`. Notebook `WS3e_Hinglish_Colab.ipynb` scores on the T4: Davlan (today's behaviour), the English bert-base, indicbert-hi on the raw text, GLiNER-X base, GLiNER multi, indicbert-hi on the Devanagari text, GLiNER-X base on the Devanagari text. IndicXlit (the plan's transliterator) needs fairseq, which does not install reliably on current Colab Python, so Azure transliteration stands in for it; the trade-off is that it needs the Azure key.
- **Tests:** `backend/tests/test_hinglish.py` (5), `evaluation/tests/test_hinglish_set.py` (3); all backend/evaluation/training tests pass.
- **Next (stage 2):** run WS3e, compare, choose a pipeline, wire detection into `/api/detect-language` and NER routing, add unit tests and the UI note.

### 2026-11-18 02:35 - WS3 completion, item 1: Hinglish, stage 2 (results, correction, wiring)
- **Correction to stage 1:** ten lower-case "chat style" sentences in the test set were saved without entity markup, so they counted as having no entities (164 entities instead of 182; the stage 1 note above that says "20 without entities" is wrong, the set has 11). Found because the Hindi model's "extra" predictions on them were obviously correct (amit, neha, delhi). The file was fixed and the Colab predictions were rescored (`evaluation/rescore_hinglish.py`); the manifest hash was updated. First-pass scores before the fix were lower for every model (Hindi model 0.924 instead of 0.939).
- **Result (ws3e_runs, strict F1, 125 sentences, rescored):** indicbert-hi on the raw English-letter text 0.939 [0.907, 0.966]; after Azure transliteration to Devanagari 0.901; GLiNER-X base 0.846 (raw) / 0.561 (transliterated); Davlan, what the app did before: 0.714; English bert-base: 0.656. Hindi model minus Davlan +0.225 [+0.158, +0.287]. Full table and caveats: `evaluation/RESULTS.md`, section "Hinglish". The labels were written by the assistant and need native-speaker review before the numbers are quoted as final.
- **Decision (no transliteration):** the Hindi model reads English-letter Hindi directly; transliterating first adds a dependency on Azure for no reliable gain.
- **Found while testing translation:** Azure Translator with `from=hi` returns English-letter Hindi unchanged and without any error (silent failure); `from=hi-Latn` translates it correctly. Checked live through `azure_translator.py`.
- **Wired into the app (`backend/main.py`):** `detect_language` returns `hi` for Romanized Hindi (word-list rule in `backend/hinglish.py`) so NER uses the Hindi model; `/api/detect-language` also returns `romanized: true/false`; `_translate_piece` switches the source to `hi-Latn` for such text (also when the user picked Hindi by hand); the offline model and MyMemory are skipped for it (MyMemory previously had no readiness check at all); Google already falls back to auto-detection. 6 new tests in `backend/tests/test_hinglish_routing.py`; backend, evaluation and training suites all pass.
- **Not done:** native-speaker review of the Hinglish labels; a UI hint for `romanized` (the field is returned, the front end does not show it); Roman-script versions of the other Indian languages; a larger, real-world Hinglish test set (the plan's in-house gold set); the live end-to-end run of the app (RAM).

### 2026-11-18 02:49 - WS3 completion, item 2: Russian, Arabic, Japanese NER (stage 1: data and notebook)
- **Test sets frozen** (`evaluation/foreign_data.py`, entries added to `MANIFEST.json`; `download_data.py` now builds them too):
  Universal NER v1 PUD test sets for Russian, German and Chinese (1,000 parallel sentences each; the Chinese PUD set is in Traditional characters), Universal NER Chinese GSDSIMP test (500, Simplified), ANERcorp CAMeL Lab test split for Arabic (924 sentences, 24,996 words: 12 fewer than the official 25,008, from the mirror's sentence split), stockmark Japanese Wikipedia NER (5,343 sentences, all used as test, one token per character). Licences: CC-BY-SA-4.0 (UNER, ANERcorp), CC-BY-SA-3.0 (stockmark).
  - Universal NER has no Japanese set (the brief assumed one). **Corrected 2026-11-18 14:45: wrong.** UNER Japanese-PUD exists on GitHub (`UNER_Japanese-PUD`); only the Hugging Face loader script does not list it. It is now frozen as `uner_ja` (1,000 sentences).
  - ANERcorp's official download requires a Google Form with personal details, so it was not used; the mirror `asas-ai/ANERCorp` has damaged text (the letters ح and ف are broken bytes), so `KFUPM-JRCAI/ANERcorp_experimental` (clean, same sentences) is used instead.
  - Japanese type decision (made once): persons -> PER; companies, political and other organisations -> ORG; places and facilities -> LOC; products and events not scored.
- **Scorer change:** predicted labels are now normalised (`PERS` -> `PER` etc., `eval_ner.LABEL_ALIASES`); before, a model that says `PERS` (the CAMeL Arabic model) would have had every person silently dropped.
- **Candidates and licences checked:** `Davlan/xlm-roberta-base-ner-hrl` (AFL-3.0, an OSI open-source licence; same size as the current model; trained on news in 10 languages, including ANERcorp for Arabic with an unstated split, so its Arabic score may include training sentences), `universalner/uner_all` (MIT, XLM-R large; trained on Universal NER train sets, so `uner_zhs` is in-domain for it), `CAMeL-Lab/bert-base-arabic-camelbert-mix-ner` (Apache-2.0; trained on the ANERcorp CAMeL train split, so the test split is held out), GLiNER-X base and GLiNER multi. Not used: `tsmatz/xlm-roberta-ner-japanese` (trained on the whole stockmark set, so it cannot be tested on it); Babelscape WikiNEuRal (non-commercial licence).
- **Notebook:** `WS3f_Russian_Arabic_Japanese_Colab.ipynb` (bundle rebuilt, 1.8 MB; clean-room check of the bundle passed for all new sets). It also scores `hrl_base` on MultiCoNER de/es/fr/zh, because it is a possible licence-clean replacement for the current model (item 3, a mentor decision; measurement only).
- **Tests:** `evaluation/tests/test_foreign_data.py` (5); all backend, evaluation and training tests pass.

### 2026-11-18 03:22 - WS3 completion, item 2: Russian, Arabic, Japanese NER (stage 2: results)
- **Ran** (`ws3f_runs`, 29 of 30 files; GLiNER multi produced none, as in the Hinglish run): the current model, `Davlan/xlm-roberta-base-ner-hrl` (AFL-3.0), `universalner/uner_all` (MIT), `CAMeL-Lab/bert-base-arabic-camelbert-mix-ner` (Apache-2.0, Arabic only) and GLiNER-X base on the six new test sets; hrl_base also on MultiCoNER de/es/fr/zh. Full table: `evaluation/RESULTS.md`, section "Russian, Arabic, Japanese".
- **First measurements for these languages with the current model (no licence):** Russian 0.488, Arabic 0.493, Japanese 0.346, German 0.514, Chinese 0.455 / 0.434 (strict F1, ordinary text).
- **Main finding:** `Davlan/xlm-roberta-base-ner-hrl` (open licence AFL-3.0, same size and labels) is better on every set: +0.25 to +0.32 on ordinary text (Russian 0.740, Arabic 0.812, Japanese 0.637, German 0.821, Chinese 0.775 / 0.709) and +0.05 to +0.25 on lower-case MultiCoNER; all intervals above zero. Its Arabic score may include training sentences (ANERcorp split unstated); the held-out CAMeL Arabic model scores 0.828, so the level is plausible.
- **Recommendation for the owner and mentor (item 3, not acted on):** replace the unlicensed multilingual model with `Davlan/xlm-roberta-base-ner-hrl`. It removes the release blocker and improves every measured language. Per-language specialists add at most +0.02 to +0.07 (Arabic camel_ar, Japanese GLiNER-X, Simplified Chinese uner_all) and are not needed for a first release.
- **App change (no behaviour change):** `NER_MODEL_MULTI` can now be set in `backend/.env` like `NER_MODEL_HI` (helper `_env_model`; default unchanged; documented in `.env.example` with the numbers); 1 new test. All tests pass.
- **Not measured:** hrl_base on Odia and Assamese (they use `NER_MODEL_MULTI` today) and on the other Indian languages; Japanese and Russian with native-speaker tests; GLiNER multi on these sets.

### 2026-11-18 12:50 - WS3 completion, items 5 and 6, and WS3 closed
- **Item 5, Hindi latency (E3):** `bench_latency.py` gained `--lang hi` (texts from the frozen Naamapadam + HiNER test sentences; the Hindi model is timed through the app's `run_ner`) and `--merge` (run the sizes one at a time; a slow machine cannot finish all three in one 10-minute command). Results: batch 1 p50/p95 402/617 ms (~200 chars), 1,971/2,463 ms (~2,000), 14.3/17.0 s (~20,000); batch 32: 263/363 ms, 1,418/1,582 ms, 12.8/16.2 s; cold start 20-26 s; 1.12 GB on disk. English bert-base in the same session: 283/444 ms and 1,566/2,094 ms, so Hindi is about 25% slower on medium texts. The protocol was reduced (30 x 3 calls for short/medium at batch 1, fewer for long and batch 32) and only about 0.6 GB RAM was free during the runs (paging), so the numbers are pessimistic. Table: `evaluation/RESULTS.md`, "Hindi NER latency".
- **Item 6, live end-to-end:** new `evaluation/e2e_live.py` drives a running backend over HTTP. Run against uvicorn with a throw-away SQLite file and MongoDB database `annotationtool_e2e` (dropped afterwards; the user's data was not touched): **38/38 checks passed** (languages list, sign-up/login, detection + NER + Azure translation for nine Indian languages and Hinglish, save/list/export, a Hindi document job). One expected name missed (Bengaluru in Kannada); entity spans include case endings in the agglutinative languages ("in Pune"); first Hindi request 24.7 s (model loaded on first use). Server log: no errors. The browser UI was not driven.
- **WS3 is closed for the agreed scope** (Hindi first, then the regional languages, plus Hinglish, Russian/Arabic/Japanese measurement, latency and a live check). Open items carried forward, all decisions or larger efforts:
  1. **Mentor decision:** switch `NER_MODEL_MULTI` to `Davlan/xlm-roberta-base-ner-hrl` (AFL-3.0; better on every measured language; removes the release blocker).
  2. **Mentor decision / no open model:** Odia (best 0.27) and Assamese (51-sentence test); Odia also lacks language detection.
  3. Native-speaker review of the Hinglish test set and an independent (in-house) test set for every language; until then no language can be "stable".
  4. Case endings inside entity spans for agglutinative languages (a display issue; could be addressed by a suffix-stripping step, not attempted).
  5. Hindi model speed: no ONNX export tried; adding `ner_hi` to `PRELOAD_MODELS` (the option already exists; the default preloads only `ner_en`) removes the 25 s first-request delay at the cost of about 1 GB of RAM from startup.
  6. GLiNER multi produced no output in the last two Colab runs (not investigated; not needed).

### 2026-11-18 14:05 - Review of the new instructions (pasted brief + MODELS_AND_DATASETS_RESEARCH.md), disk fix, item 5 status
- **Review notes (agreed with the owner before starting):** (1) the brief's reason for skipping student B ("teacher effect is clear, WNUT 0.490 vs 0.558") is the claim E1 overturned (B's 0.490 comes from tagging a bare "@"; names-only B 0.595 > A 0.565); skipping B's seeds is a scheduling choice, not a finding. (2) The GLiNER-X entities "dropped" by the adapter are 0.14-0.93% of predictions (at most +0.007 F1), not 3%; my earlier wording was wrong and is corrected in RESULTS.md and above. They came back with NO character offsets, so offset snapping cannot place them; better text matching is possible on Colab, low priority. (3) Our MultiCoNER scores use the validation split as the test set, so a threshold must be tuned on MultiCoNER train, not validation. (4) The research file adds `whoisjones/otter-cross-mmbert` (Apache-2.0) as a single-model Davlan replacement and warns that the Naamapadam Odia and Assamese test sets are **silver** (machine-projected): the "no open model works for Odia" verdict is withdrawn until human-labelled tests (IL-NER Odia, GUIT-AsTourNE) are scored.
- **Disk:** C: was down to 0.55 GB (the IndicNER download is stored twice because Windows has no symlinks). The Hugging Face cache (3.0 GB) was moved to `D:\huggingface` (182 GB free), `HF_HOME` set for the user account and in `backend/.env`. C: back to about 3 GB free.
- **Proposed model memory limit (for item 3, not implemented yet):** `NER_MAX_MODELS=2` (least-recently-used eviction of NER pipelines); a model larger than 1.5 GB on disk (GLiNER-X base) clears the others first; translation models a separate limit of 1. Basis: 0.9-1.0 GB resident per model measured today, about 0.6 GB free RAM with VS Code open, 20-30 s reload cost.
- **Item 5: WS3 close-out list, status on 2026-11-18:**

| item | status |
|---|---|
| Licence blocker for the 7 non-Indian languages (Davlan, no licence) | **Not done.** Measured candidates (hrl_base AFL-3.0, uner_all MIT, CAMeL Arabic Apache-2.0, GLiNER-X Apache-2.0); `NER_MODEL_MULTI` is now a setting; default unchanged (mentor). Otter and the per-language specialists still to test (R4, Colab). |
| Russian / Arabic / Japanese measurement | **Done** (UNER PUD ru, ANERcorp ar, stockmark ja; WS3f). No native-speaker or second test set yet. |
| RAM: lazy loading | **Partly.** Models already load on first use; there is no eviction, so memory grows with each language used. LRU limit proposed (above), to be built with item 3. Disk problem solved (cache on D:). |
| Live test | **Done** for the API (38/38 checks, 9 Indian languages + Hinglish, save/export, document job). The browser UI was not driven. |
| Hindi latency | **Done** (reduced protocol; RAM-starved laptop, so pessimistic): about 2.0 s per 2,000 characters, 25% slower than English. |
| Odia / Assamese script detection | **Not done.** langdetect has no Odia or Assamese profile (Odia raises an error). Fix planned as R1 (Unicode-script router + GlotLID). |
| Odia / Assamese NER | **Verdict withdrawn:** the Naamapadam test sets for both are silver; re-measure on IL-NER Odia and GUIT-AsTourNE (R2). |

### 2026-11-18 14:25 - Item 4 (Indian-language table) and item 2 (E5, English-EWT) done
- **Item 4:** found that the app-style evaluation path was unfair to IndicNER (its uncased tokenizer strips Indian vowel signs, so reported character offsets do not cover whole words). New `evaluation/word_level_eval.py` scores word by word (first sub-word label), reproducing IndicNER's published numbers within 0.01 (Hindi 0.828, Bengali 0.798, Marathi 0.805, Tamil 0.734, Telugu 0.814). indicbert-hi scores identically on both paths. Even scored fairly, IndicNER is tied with indicbert-hi on Hindi-Naamapadam, Hindi-MultiCoNER, Bengali and Tamil and 0.016-0.044 below it on HiNER, Marathi, Telugu, Gujarati, Kannada, Malayalam and Punjabi (paired bootstrap). **Best permissive model per language = indicbert-hi for all nine gold-tested languages**; Odia and Assamese undecided (silver tests). Every earlier IndicNER number in RESULTS.md was produced through the pipeline path and is too low; the new section says so. Table: RESULTS.md, "Indian languages: best permissive model per language".
- **Item 2 (E5):** `uner_en_ewt` added to `foreign_data.py` / `MANIFEST.json`; `seen_unseen.py` now includes it in the English tables. Strict F1: bert-base 0.703, teacher 0.793, A 0.714, B 0.771, C 0.726. A is not reliably better than bert-base (+0.011 [-0.015, +0.034]); B is +0.068 [+0.045, +0.094] and within 0.022 of the teacher. Unseen-subset and common-unseen-sentence columns added. **The English-default candidate is student B, not A**; both were trained with FIN (licence unverified), so B-noFIN and A-noFIN with 3 seeds each are needed (not yet trained; for the next Colab notebook). Student A's model card updated.
- **Disagreement with the pasted brief, recorded for the mentor:** the brief skips student B as "an ablation, not a candidate". On the fair English test (EWT) and on names-only E1, B is the best small model. Recommend training B-noFIN (3 seeds) in the next Colab run rather than skipping B.

### 2026-11-18 14:45 - Two Colab notebooks prepared (not yet run)
- **`WS2_Colab/WS2b_noFIN_Seeds_Colab.ipynb` + `ws2b_bundle.zip`:** trains `A_noFIN_distill` and `B_noFIN_plain` (CoNLL + WNUT + BTC, no FIN), seeds 0, 1, 2; best checkpoint chosen on a dev file with no FIN sentences (1,997 WNUT + BTC sentences); then scores all six on WNUT, BTC, FIN and English-EWT on the GPU. `colab_runner.py` has the two new experiments, run only when named (the original WS2 notebook still runs only A-D). Clean-room check passed (bundle unpacked in an empty folder; dry run lists the six runs with the right sources and dev file; all four test sets load and match the manifest).
- **`WS3_Colab/WS3g_Replacement_Models_Colab.ipynb` + rebuilt `colab_eval_bundle.zip` (2.5 MB):** items 1 and 3 of the brief and research R4. Fixed-label models: Davlan and hrl_base on WikiANN fr/es (first 3,000 test sentences, cased, silver) and UNER Japanese-PUD; uner_all on Japanese-PUD; CamemBERT-with-dates (fr), bert4ner (zh), CAMeLBERT-MSA (ar), llm-book (ja, on PUD only). Zero-shot models (Otter, GLiNER-X base and small, GLiNER multi) run once at threshold 0.1 on the MultiCoNER TRAIN tuning samples and on all test sets, so `evaluation/tune_threshold.py` can choose the threshold on the tuning samples afterwards without re-running. Improved adapter re-run on the four sets that lost entities.
- **New frozen data:** MultiCoNER train samples `mc_de_tune`, `mc_es_tune`, `mc_fr_tune` (600 sentences each, every k-th, role "tune"), WikiANN fr/es test and dev, UNER Japanese-PUD. All earlier manifest hashes unchanged (checked).
- **Code:** `gliner_adapter.py` handles Otter (`otter:` prefix; `trust_remote_code` with the revision pinned to 8729188e) and finds offset-less entities by case-folded and whitespace-flexible matching, printing the ones it still cannot place; `bootstrap.counts_per_sentence` takes a `min_score`; new `tune_threshold.py`; `eval_ner.LABEL_ALIASES` maps the Japanese stockmark type names. 6 new tests; all suites pass.
- **Proposed memory limit** `NER_MAX_MODELS=2` still to be implemented together with wiring any chosen model.

### 2026-11-20 00:05 - Results of WS2b (English without FIN) and WS3g (replacement models), items 1 and 3 done
- **Runs:** `ws2b_runs` (24 files: A_noFIN and B_noFIN, seeds 0-2, on WNUT/BTC/FIN/EWT) and `ws3g_runs` (67 files). Missing from WS3g: the Japanese llm-book model and GLiNER multi produced no files (errors not captured; a Colab screenshot is needed to diagnose; neither is needed for a decision).
- **English (WS2b):** B_noFIN English-EWT 0.762 +- 0.009 over 3 seeds (every seed +0.052 to +0.069 over bert-base, all intervals above 0), equal to B-with-FIN within seed spread; names-only WNUT 0.590, BTC 0.792. A_noFIN 0.742 +- 0.017. **Recommendation for decision 1 (English default): B_noFIN**, seed chosen on validation F1 (training_log.json in Drive `ws2b_results`), after the owner confirms the CoNLL / WNUT / BTC licences (CC-BY-4.0 for WNUT and BTC, attribution needed).
- **Zero-shot thresholds** chosen on MultiCoNER train samples (never test): Otter 0.25 (tuning F1 0.683 vs 0.646 at 0.5), GLiNER-X base 0.70 (0.678 vs 0.646), GLiNER-X small 0.60 (0.593 vs 0.586). Effect on test sets at the tuned value vs 0.5 (paired): Otter +0.027 to +0.097 on 9 sets (Russian +0.097), -0.010 / -0.005 on WikiANN fr / es, n.s. on mc_fr, mc_zh, Arabic; GLiNER-X base +0.007 to +0.071 on 11 sets (Japanese PUD +0.071), n.s. on mc_de, Arabic, Chinese PUD; GLiNER-X small mostly +0.002 to +0.034, -0.017 (n.s.) on mc_de.
- **Item 3 (GLiNER-X for de/fr/es):** after tuning it beats GLiNER multi and the current model on every human-labelled set (cased German PUD 0.777, MultiCoNER de/es/fr 0.608/0.693/0.683), but **Otter (Apache-2.0, 309M) is equal or better on 12 of 13 test sets** at a smaller size, so GLiNER-X is not wired. The brief's condition ("if it still beats GLiNER multi and Davlan, wire it") is met only against Davlan outside WikiANN, where Davlan is in-domain.
- **R4 (replacement for the unlicensed model):** no single winner. Cased edited text: hrl_base best or tied (de 0.821, ru 0.740, ja-PUD 0.718, zh-Trad 0.775). Lower-case and Wikipedia-style text: Otter (MultiCoNER de/es/zh 0.642/0.700/0.651, WikiANN fr/es 0.702/0.716 among open models, stockmark ja 0.731). Specialists: CamemBERT-with-dates for French lower-case (0.752), CAMeLBERT-MSA for Arabic (0.843). Every open model beats the current one on every human-labelled set (+0.25 to +0.49).
- **Item 1:** improved offset matching recovered no additional entities (identical predictions in all six re-runs). Closed.
- **Not wired yet:** any replacement model, and the `NER_MAX_MODELS=2` memory limit; both wait for the owner's choice between (a) hrl_base as a one-model drop-in, (b) Otter (needs `trust_remote_code`, pinned revision, about 1.2 GB RAM), (c) per-language routing (hrl_base / Otter / specialists) with the memory limit.

### 2026-11-20 06:15 - Documents consolidated: one report, one guide, one technical record
- **Mentor report:** `major/Project_Report.pdf` (version 1.0, 8 pages; source `_guide_source/build_project_report.py`), with a change history, a decision register (D1-D10), a corrections register and current results. It replaces and contains the content of: Project_Progress_Report (29 Oct), Progress_Report_Accuracy_and_NER_Issue (30 Oct), Project_Report_Pipeline_Workflow_Results (15 Nov), Mentor_Report_WS0-WS2_and_Decisions (16 Nov), Indian_Regional_Languages_Assessment (17 Nov), Correction_Note_E1 (18 Nov), Mentor_Update_2026-11-20. Those PDFs and their builder scripts were deleted (OneDrive keeps deleted files in its recycle bin for 30 days). References to them in older log entries are historical.
- **Plain-language guide:** `major/Plain_Language_Guide.pdf` (version 1.0, 25 pages, 6 chapters; source `_guide_source/build_plain_guide.py`). Replaces WS0_Explained_for_Fullstack_Developers and Plain_Language_Guide_Part1-4 (deleted). WS0's chapter was recreated from its PDF text (its source was not kept); chapter 6 (after WS3) is new. `build_guide_part1-4.py` are now chapter sources only.
- **Technical record:** IMPROVEMENT_LOG.md (this file) and evaluation/RESULTS.md. The six small result tables in evaluation/ (ner_btc_fin, translation_local_zh_ar_ja, ws1_*) were merged into RESULTS.md Appendix A and deleted; translation_results.md was empty and was deleted. eval_translation.py's example output path changed accordingly.
- **Rule from now on:** update these documents (add a change-history row and bump the version) instead of creating new report files.
- Test suites: 253 tests (backend 220 with 2 skipped, evaluation 26, training 7); one backend test (empty-file upload) failed once and passed in three re-runs: flaky, not investigated.

### 2026-11-21 16:10 - Change of direction (mentor) and a team task brief
- **Mentor's guidance (reported by the owner, 21 Nov 2026):** making the models more accurate is a separate research topic; the project must deliver an automated annotation tool that is open source, properly hosted and works for every task. Research items (plan R2-R9, most of WS4-WS5) become future work; WS6-WS8 (features, reliability, release) become the main work.
- **New maintained document:** `major/Team_Task_Brief.pdf` (version 1.0, 4 pages; source `_guide_source/build_team_tasks.py`) for the teammates: why the plan changed, the project in one minute, working rules, a task table and task cards. Teammate tasks, in order: T1 review/correction screen, T2 CoNLL/BIO and JSONL export, T3 user testing, T4 native check of the Hinglish test set (start here); T5 bulk upload, T6 Docker, T7 OCR (next). Internal tasks kept by the owner: M1 model memory limit, M2 script-based language detection, M3 flaky test, M4 licence-clean models (after the mentor's decision), M5 offline translation for all Indian languages, M6 persistent jobs.

### 2026-11-21 16:20 - Code uploaded to GitHub (pull request, not merged)
- Repository: Astha222/Language-Annotation-Tool-with-NER-and-ML-Translation (public; last change before this: 4 Oct 2026). Pushed branch `improvements-oct-nov-2026` from a fresh clone on D:\repos\annotation-tool (the OneDrive folder is not a git repository) and opened **pull request #1** into `main`: 93 files, 82 new, 11 updated, none deleted; commit 5a8f95e, authored with the GitHub noreply address (the personal email stays private).
- Before the upload: example paths with the local Windows user name in the Colab notebook texts replaced by `<your folder>`; `.gitignore` now keeps `evaluation/data/MANIFEST.json` (checksums) and ignores `evaluation/runs/` (saved predictions, 107 MB). A first version put comments at the end of `.gitignore` lines, which git does not support (the rules silently failed); fixed before committing. No secrets, `.env`, databases or test data were uploaded (checked). All suites passed from the clean clone (backend 220 with 2 skipped, evaluation 26, training 7).
- Not merged: the repository owner (Astha) or Muskan merges after review. Future changes: update the clone on D:, commit on a branch, open a pull request.
- **2026-11-21 16:28 merged:** pull request #1 merged into `main` by muskan-424 (merge commit 345fcf8). GitHub CI passed on it (backend on Python 3.11 and 3.12, frontend). Copilot's automatic review did not run (its quota was used up). The D: clone is on the updated `main`.

### 2026-11-21 16:45 - M1-M3: NER memory limit, language detection by script, flaky upload test
- **Working copy moved:** from now on the project is developed in the git clone `D:\repos\annotation-tool` (branch per task, pull request into `main`). The OneDrive folder is an old snapshot (OneDrive and git do not mix well, and copying OneDrive over the clone would erase teammates' merged work). `.env`, test data and saved predictions were copied into the clone; git ignores them.
- **M1 memory limit (`NER_MAX_MODELS`, default 2):** at most two NER pipelines stay loaded (about 1 GB each); the least recently used is unloaded when a third is needed and reloaded on demand (20-30 s). Loading is serialised so two requests never load the same model twice; `/health/models` lists the loaded models in use order and the limit; a startup warning when `PRELOAD_MODELS` asks for more NER models than the limit. `.env.example` documents it. 5 new tests (eviction order, reuse, 0 = unlimited, health report, four concurrent requests load once).
- **M2 language detection by script (`backend/script_lang.py`):** Odia, Punjabi, Gujarati, Tamil, Telugu, Kannada and Malayalam are decided by their script; Bengali script is Assamese when it contains ৰ or ৱ, else Bengali; Devanagari goes to langdetect (hi / mr / ne) with Hindi as fallback; other scripts unchanged. New `evaluation/eval_language_id.py` (19,365 sentences from 24 frozen test sets, first 1,000 per set): **Odia 0% -> 99.9%, Assamese 0% -> 92.2%** (the 3 misses are Assamese sentences without ৰ/ৱ, read as Bengali), every other set unchanged (no regression); overall 90.2% -> 96.2% (that includes Hinglish 0% -> 97.6% from 18 Nov). 5 new tests.
- **Finding (open):** langdetect is weak on short English: only 76.4% of English-EWT sentences and 89.3% of WNUT-2017 sentences are detected as English (wrong answers: German, Tagalog, none). Short English input with "auto" language may therefore get the multilingual model. Candidate fix: GlotLID or a minimum-length rule; not done.
- **M3 flaky test:** the failure (upload of a whitespace-only file, once, in the OneDrive folder) could not be reproduced (300 uploads in a row: all correct). Plausible cause fixed: the temporary upload file was deleted in a `finally` block, and on Windows a scanner or sync tool can lock a just-written file, so a failed delete replaced the real 400 answer with a server error. The delete is now guarded (logged; `_purge_old` removes leftovers later). New test simulates the locked file; it fails without the fix and passes with it.
- **Tests:** backend 231 (2 skipped), evaluation 26, training 7; all pass.

### 2026-11-23 16:47 - M5: offline translation for all 11 Indian languages
- **What:** `backend/local_translator.py` now serves English <-> Hindi, Marathi, Bengali, Assamese, Odia, Punjabi, Gujarati, Tamil, Telugu, Kannada and Malayalam with the two pinned IndicTrans2 distilled models already used for Hindi. One model per direction serves every Indian language (the language is a per-sentence tag), so all 11 cost the same memory as Hindi alone (about 0.9 GB per direction; `LOCAL_MT_MAX_LOADED=2` keeps both). Indian <-> Indian and Indian <-> European pairs pivot through English. `LOCAL_INDICTRANS=0` restores the old behaviour (Hindi via Marian, other Indian languages Azure only). `/api/languages` now reports `offline_translation: true` for all 11.
- **Bug fixed:** `indic-nlp-library-itt` 0.1.1 defines `DevanagariNormalizer.NUKTA` as the text `\u093C`, so normalising ऱ, ऩ, ऴ, क़ ... wrote a literal backslash escape into the model input; seen as `farmers\ u093C` in a Marathi round trip. Patched in `backend/indic_processor.py`; regression test fails without the patch and passes with it.
- **Laptop environment:** model loading failed with `operator torchvision::nms does not exist`: Windows Smart App Control blocks torchvision's `_C.pyd` (`WinError 4551`); a matching CPU wheel did not help. torchvision is not used by the project and was uninstalled from the user's Python (reinstall with `pip install torchvision` if another project needs it). README troubleshooting note added.
- **Measured** (`evaluation/eval_translation.py`, WMT24++ codes added for bn, gu, kn, ml, mr, pa, ta, te): offline vs Azure chrF++ on the same 40 segments: hi 35.2/36.4, mr 45.7/43.7, bn 41.2/45.2, gu 53.4/45.6 (Azure 2 failures), pa 54.8/50.2, ta 47.1/44.6, te 47.0/45.9, kn 57.6/56.1, ml 46.7/49.2. Offline speed 7-34 chars/s on this CPU (RAM-starved). Odia and Assamese not measured (no WMT24++ set); round-trip check fine for all 11. Table: RESULTS.md "Offline translation for the Indian languages". `make_language_status.py` records these as translation status "beta" (no agreed translation thresholds yet); `backend/language_status.json` regenerated (NER entries unchanged).
- **Tests:** local translator tests rewritten for shared models (all 11 supported, one model per direction, the language tag reaches the model, switched-off mode); two tests that used Bengali as the "no offline model" example now use Korean; processor tests for the nukta bug and all 11 scripts. Backend suite passes.

### 2026-11-23 18:38 - M6: document jobs survive a server restart
- **Before:** job state lived only in a Python dict; a restart (crash, deploy) lost every job, and the page then showed "Job not found (the server may have restarted)". The uploaded document text was already on disk.
- **Now:** a `jobs` table (`JobDB`) mirrors every job. Status changes are written at once, stage/progress at most every 2 s (`JOB_SAVE_INTERVAL`); a failed write is logged and never stops the job. At startup (`lifespan` -> `recover_jobs()`) jobs that were queued or running are restarted once from their saved document (`MAX_JOB_ATTEMPTS = 2`, to avoid a crash loop); jobs that cannot be restarted (document expired, or already restarted) are marked failed with a clear message. `GET /api/jobs/{id}` falls back to the table for finished jobs; new `GET /api/jobs` lists the user's 10 most recent jobs (for reconnecting the page after a reload); cancelling a job that is not running is a harmless no-op; old rows are purged with the documents (6 h). `DOC_DIR` is now configurable (Docker volume).
- **Live check** (real uvicorn server, throw-away SQLite/Mongo/DOC_DIR, offline translation English -> Hindi): job started, server killed with `Stop-Process -Force` while translating (saved row: running, progress 0.168, attempts 1), server restarted: log "Unfinished jobs after restart: 1 restarted, 0 marked failed", job finished 64 s later (12 entities, Hindi translation, row: done, attempts 2), `GET /api/jobs` listed it. With models loaded, job-status requests during a job took median 12 ms, max 22 ms; right after the restart one status request took over 30 s while the job loaded two models on this RAM-starved laptop (about 0.6 GB free): a model-loading effect, not the job saving.
- **Tests:** 7 new (`tests/test_job_persistence.py`: finished job readable after a restart, saved row follows the job, interrupted job restarted once, not restarted twice or without its document, list shows only my jobs newest first, cancelling a finished job, progress writes throttled); the test setup clears saved jobs between tests. Backend 244 tests pass.
- **Not done:** the front end does not yet call `GET /api/jobs` to reconnect after a page reload (it keeps the job id in memory); a small front-end change for the teammates' task list.
- **CI fix (19:17):** the first CI run failed on Python 3.12 with "database is locked": a document job left running by one test kept saving its progress (new in M6) while the next test's setup cleared the jobs table. The test setup now cancels running jobs and waits for their threads after every test, and the app's SQLite connections wait up to 30 s for a lock instead of the default 5 s (several threads write in production too). Three local full runs pass.
