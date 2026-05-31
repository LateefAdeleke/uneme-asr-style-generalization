# ASR Experiment Plan: Speech Style Generalization

## Overview

This project investigates how "speech style affects the generalization of ASR models" in low-resource language documentation settings.

### Core Hypothesis

> ASR models trained on constrained speech (e.g., read or careful speech) do "not reliably generalize" to realistic naturalistic speech (e.g., narratives, interviews), even when cross-lingual transfer is applied.

---

## Dataset Summary

The dataset consists of:
*about 7hours 
* ~7,900 utterances
* Two coarse speech styles:

  * Naturalistic (narratives, interviews)
  * Constrained (careful speech, read speech)

### Key Characteristics

* Naturalistic speech: multiple speakers
* Constrained speech: limited speakers
* All splits are:

  * utterance-disjoint
  * speaker-disjoint (where applicable)
  * path-resolved via `audio_path`

---

## Split Regimes

### 1. Main Split (Primary Benchmark)

**Purpose:** Evaluate generalization to realistic natural speech

| Split | Content                    |
| ----- | -------------------------- |
| Train | Constrained + Naturalistic |
| Dev   | Naturalistic only          |
| Test  | Naturalistic only          |

Location:

```
data/metadata/main/
```

---

### 2. Reverse Auxiliary Split

**Purpose:** Evaluate reverse generalization (natural → constrained)

| Split | Content           |
| ----- | ----------------- |
| Train | Naturalistic only |
| Dev   | Naturalistic only |
| Test  | Constrained only  |

Location:

```
data/metadata/reverse_aux/
```

---

### 3. Mixed → Constrained Auxiliary Split

**Purpose:** Evaluate robustness of mixed-style training on constrained speech

| Split | Content                            |
| ----- | ---------------------------------- |
| Train | Mixed (Constrained + Naturalistic) |
| Dev   | Mixed (Constrained + Naturalistic) |
| Test  | Constrained only                   |

Location:

```
data/metadata/mixed_to_constrained_aux/
```

---

## Experiment Matrix

### Tier 1: Core Experiments (Primary Results)

These directly test the paper’s main claim (evaluation on **naturalistic** speech).

#### E1 — Naturalistic → Naturalistic (Baseline)

* Train: naturalistic only
* Test: naturalistic
* Split: `main`

#### E2 — Constrained → Naturalistic

* Train: constrained only
* Test: naturalistic
* Split: `main`

#### E3 — Mixed → Naturalistic

* Train: constrained + naturalistic
* Test: naturalistic
* Split: `main`

---

### Tier 2: Auxiliary (Non-transfer)

These provide supporting evidence under constrained evaluation.

#### E4 — Naturalistic → Constrained

* Train: naturalistic
* Test: constrained
* Split: `reverse_aux`

#### E5 — Mixed → Constrained

* Train: mixed
* Test: constrained
* Split: `mixed_to_constrained_aux`

---

### Tier 3: Cross-lingual Transfer (Primary + Supplementary)

#### Primary (Naturalistic Target)

These are **central to the paper**.

#### E6 — Transfer + Naturalistic → Naturalistic

* Split: `main`

#### E7 — Transfer + Constrained → Naturalistic

* Split: `main`

#### E8 — Transfer + Mixed → Naturalistic

* Split: `main`

---

#### Supplementary (Constrained Target Diagnostics)

These test whether transfer appears stronger under easier evaluation conditions.

#### E9 — Transfer + Naturalistic → Constrained

* Split: `reverse_aux`

#### E10 — Transfer + Mixed → Constrained

* Split: `mixed_to_constrained_aux`

---

## Models

Two model families:

1. **Whisper-style (Seq2Seq)**
2. **Wav2Vec2 / MMS-style (CTC)** (I am consedering running this too, if important) 

---

## Evaluation Metrics

For all experiments:

* **WER (Word Error Rate)**
* **CER (Character Error Rate)**
  

Also store:

* predictions
* references
* metadata (style, speaker, session)

---

## Experiment Naming Convention

```
W_E1_nat2nat_main_noXfer
W_E2_cons2nat_main_noXfer
W_E3_mix2nat_main_noXfer
W_E6_nat2nat_main_xfer
W_E7_cons2nat_main_xfer
W_E8_mix2nat_main_xfer
W_E9_nat2cons_rev_xfer
W_E10_mix2cons_aux_xfer
```

For MMS:

```
M_E1_...
```

---

## Execution Order (IMPORTANT)

### Phase 1 — Core validation

1. E1 (baseline)
2. E2 (key failure case)
3. E3 (robustness)

### Phase 2 — Auxiliary (non-transfer)

4. E4
5. E5

### Phase 3 — Transfer (primary)

6. E6
7. E7
8. E8

### Phase 4 — Transfer diagnostics (supplementary)

9. E9
10. E10

---

## Key Comparisons

* **E2 vs E1** → does constrained fail on naturalistic?
* **E3 vs E2** → does mixed training help?
* **E7 vs E6** → does transfer survive style mismatch?
* **E8 vs E7** → does mixed stabilize transfer?
* **E6 vs E9** → does transfer perform better on constrained vs naturalistic?
* **E8 vs E10** → does mixed transfer benefit more on constrained?

---

## Expected Findings

* Constrained-trained models degrade on naturalistic speech
* Mixed training improves robustness
* Transfer gains are **stronger under constrained evaluation**
* Transfer gains weaken or become unstable under naturalistic evaluation
* Evaluation on constrained speech overestimates real-world performance

---



---

## Outputs

Each experiment save:

* model checkpoints
* WER/CER results
* predictions
* logs

All outputs go to:

```
results/<experiment_id>/
```

---

## Final Note

The **main split (E1–E3, E6–E8)** drives the paper’s contribution.

Experiments targeting constrained speech (E4, E5, E9, E10) are included as **diagnostic and supplementary analyses** to contextualize performance differences across evaluation conditions.

Focus on clarity, reproducibility, and clean comparisons.
