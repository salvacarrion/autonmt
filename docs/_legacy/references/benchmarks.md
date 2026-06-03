# Benchmarks

Performance reference across AutoNMT versions. Whenever a new version is tagged,
the same small Multi30k experiment is re-run so regressions (in BLEU **or** speed)
are easy to spot.

> [!NOTE]
> These numbers are a sanity check, not a leaderboard. When AutoNMT and Fairseq
> are compared the models are only *approximately* equivalent (same size and
> hyper-parameters, different implementations), so small BLEU gaps are expected.
> Early versions also lacked features (bucketing, dynamic batching, warm-up,
> iterative decoding), which mostly shows up as slower training/translation.

**Standard setup** (unless noted otherwise per version):

- **Dataset:** Multi30k, de→en
- **Model:** Small Transformer — 256 emb / 3 layers / 8 heads / 512 ffn / 0.1 dropout
  (AutoNMT: 7.0M params; Fairseq: 7,025,664)
- **Training:** `batch_size=128`, `max_tokens=None`, `lr=0.001`, `optimizer=adam`,
  `seed=1234`, `patience=10`, `num_workers=12`
- **Predict:** beam=1, `sacrebleu_bleu`, unknowns corrected

---

## AutoNMT v1.0 — _pending_

Placeholder for the v1.0 run. Fill in the parameters actually used and the result
tables below.

- **Date:** TBD
- **Commit:** TBD
- **Dataset:** Multi30k, de→en (lowercase? normalization?)
- **Model:** TBD
- **Training:** TBD
- **Predict:** TBD
- **Hardware:** TBD

### AutoNMT toolkit

| max_epochs | subword_model | vocab_size | sacrebleu_bleu | train_time | translate_time (beam=1) |
|:----------:|:-------------:|:----------:|:--------------:|:----------:|:-----------------------:|
| 10         | unigram       | 4000       |                |            |                         |
| 10         | word          | 4000       |                |            |                         |
| 10         | bpe           | 4000       |                |            |                         |
| 10         | char          | 101        |                |            |                         |
| 10         | bytes         | 260        |                |            |                         |

### Fairseq toolkit (optional comparison)

| max_epochs | subword_model | vocab_size | sacrebleu_bleu | train_time | translate_time (beam=1) |
|:----------:|:-------------:|:----------:|:--------------:|:----------:|:-----------------------:|
| 10         | unigram       | 4000       |                |            |                         |
| 10         | word          | 4000       |                |            |                         |

**Notes:**
- _vs v0.6: …_
- _Regressions / wins: …_

---

## AutoNMT v0.6 — 27/10/2023

Same model size as the standard setup, swept across architectures.

- **Preprocessing:** `[NFKC(), Strip()]`, no lowercase, `force_overwrite=True`, no split shuffle
- **Training:** `batch_size=128`, `adam`, `lr=0.001`, `seed=1234`, `iter=10`
- **Data:** `multi30k_de-en_original`, de→en, evaluated on `multi30k`

### Transformer

| subword_model | vocab_size | sacrebleu_bleu |
|:-------------:|:----------:|:--------------:|
| bytes         | 260        | 7.583900       |
| char          | 101/101    | 20.450112      |
| bpe           | 4000       | 32.808826      |
| word          | 4000       | 32.626757      |

### Simple LSTM (2 layers, 512 hidden ≈ 7.4M params)

| subword_model | vocab_size | sacrebleu_bleu |
|:-------------:|:----------:|:--------------:|
| bytes         | 260        | 4.688523       |
| char          | 101/101    | 4.404792       |
| bpe           | 4000       | 9.013423       |
| word          | 4000       | 10.356350      |

### z+GRU (1 layer, 512 hidden ≈ 10.3M params)

| subword_model | vocab_size | sacrebleu_bleu |
|:-------------:|:----------:|:--------------:|
| bytes         | 260        | 4.221123       |
| char          | 101/101    | 4.441055       |
| bpe           | 4000       | 13.061608      |
| word          | 4000       | 13.239970      |

### z+GRU (2 layers, 512 hidden ≈ 10.3M params)

| subword_model | vocab_size | sacrebleu_bleu |
|:-------------:|:----------:|:--------------:|
| word          | 4000       | 13.032078      |

---

## AutoNMT v0.5 — 27/10/2023

Transformer (standard setup), de→en, evaluated on `multi30k`. Three preprocessing /
optimizer variants were tried.

### Variant A — NFKC + Strip + **lowercase**, batch=128, adam

| subword_model | vocab_size | sacrebleu_bleu |
|:-------------:|:----------:|:--------------:|
| word          | 4000       | 34.179303      |
| bpe+bytes     | 4000       | 34.019237      |

### Variant B — NFKC + Strip, **no lowercase**, batch=128, adam

| subword_model | vocab_size | sacrebleu_bleu |
|:-------------:|:----------:|:--------------:|
| word          | 4000       | 34.207387      |
| bpe+bytes     | 4000       | 33.085672      |

### Variant C — NFKC + Strip, **no lowercase**, batch=1024, adamw

| subword_model | vocab_size | sacrebleu_bleu |
|:-------------:|:----------:|:--------------:|
| word          | 4000       | 32.651625      |
| word          | 8000       | 32.140449      |
| word          | 10000      | 30.840177      |
| bpe+bytes     | 4000       | 32.062014      |
| bpe+bytes     | 8000       | 32.079096      |
| bpe+bytes     | 10000      | 32.649577      |
| char+bytes    | 357/357    | 32.819399      |

> Bigger batch (1024 + adamw) traded ~1–2 BLEU for throughput here; larger vocabs
> did not help on this small dataset.

---

## AutoNMT v0.4 — 09/04/2022

Comparison of AutoNMT (custom toolkit) and Fairseq
(fork `31d94f556bd49bc7e61511adbda482b2c54652b5`). Standard setup, dataset lowercased,
`cf/multi30k` de→en, `max_epochs=10`.

### Exp. 1 — Local: 1× GeForce GTX 1070, AMD Ryzen 7 2700X

| toolkit  | subword_model | vocab_size | bleu      | train_time     | translate_time (beam=1) |
|:--------:|:-------------:|:----------:|:---------:|:--------------:|:-----------------------:|
| AutoNMT  | unigram       | 4000       | 33.817927 | 0:04:18.731907 | 0:00:02.005633          |
| AutoNMT  | word          | 4000       | 34.556056 | 0:03:55.216156 | 0:00:01.819264          |
| Fairseq  | unigram       | 4000       | 35.671859 | 0:02:37.182229 | 0:00:06.438150          |
| Fairseq  | word          | 4000       | 34.991464 | 0:02:28.685353 | 0:00:05.941582          |

### Exp. 2 — Remote: 2× NVIDIA TITAN XP, Intel i7-7800X @ 3.50GHz

| toolkit                        | GPUs | subword_model | vocab_size | bleu      | train_time     | translate_time (beam=1) |
|:------------------------------:|:----:|:-------------:|:----------:|:---------:|:--------------:|:-----------------------:|
| AutoNMT                        | 2    | unigram       | 4000       | 35.171231 | 0:02:50.929962 | 0:00:01.328614          |
| AutoNMT                        | 2    | word          | 4000       | 34.698708 | 0:02:54.390875 | 0:00:01.209333          |
| Fairseq (no venv) — **BUG**¹   | 2    | unigram       | 4000       | 35.520457 | 0:12:14.061107 | 0:00:06.639795          |
| Fairseq (with venv)            | 2    | unigram       | 4000       | 35.520457 | 0:05:18.730773 | 0:00:06.789658          |
| Fairseq (with venv)            | 1    | unigram       | 4000       | 36.509620 | 0:02:13.011535 | 0:00:06.667130          |

¹ Launched from Python, Fairseq's data parallelization runs ~200% slower than from
the command line (12m vs 5m for the same run).

---

## AutoNMT v0.2a — 06/01/2021

Earliest comparison. Take with a grain of salt: many training parameters were not
yet supported by AutoNMT, so this is more of a "does it learn at all" check than a
fair race. Standard setup, dataset lowercased, `multi30k_test` de→en,
1× GeForce GTX 1070 + AMD Ryzen 7 2700X.

### Custom toolkit (v0.2a)

| max_epochs | subword_model | vocab_size | bleu      | train_time     | translate_time (beam=1) |
|:----------:|:-------------:|:----------:|:---------:|:--------------:|:-----------------------:|
| 1          | unigram       | 4000       | 5.559104  | 0:00:26.414141 | 0:00:02.620312          |
| 1          | word          | 4000       | 7.392883  | 0:00:21.968405 | 0:00:17.251382          |
| 5          | unigram       | 4000       | 29.632340 | 0:02:15.064932 | 0:00:08.767339          |
| 5          | word          | 4000       | 30.972321 | 0:01:57.834483 | 0:00:12.347725          |
| 10         | unigram       | 4000       | 32.816378 | 0:04:11.125762 | 0:00:08.315648          |
| 10         | word          | 4000       | 34.682657 | 0:03:38.593067 | 0:00:02.082928          |

### Fairseq (1.0a)

| max_epochs | subword_model | vocab_size | bleu      | train_time     | translate_time (beam=1) |
|:----------:|:-------------:|:----------:|:---------:|:--------------:|:-----------------------:|
| 1          | unigram       | 4000       | 13.707765 | 0:00:21.697430 | 0:00:11.276265          |
| 1          | word          | 4000       | 10.338827 | 0:00:19.449526 | 0:00:11.645491          |
| 5          | unigram       | 4000       | 33.460104 | 0:01:28.053523 | 0:00:10.084800          |
| 5          | word          | 4000       | 33.347637 | 0:01:17.936529 | 0:00:10.063523          |
| 10         | unigram       | 4000       | 35.123375 | 0:02:38.338926 | 0:00:09.955724          |
| 10         | word          | 4000       | 34.706139 | 0:02:31.938635 | 0:00:10.285918          |

---

## Takeaways (historical)

**AutoNMT — wishlist that drove later versions:**
- Bucketing is needed to speed up training.
- Max-tokens (dynamic batching) is desirable; truncation alone loses part of the batch.
- Iterative decoding is needed to use beam search in real use cases.
- Warm-up is needed to speed up Transformer convergence.

**Fairseq — why it was eventually dropped:**
- Many bugs and dependency incompatibilities; can't be installed alongside the rest of
  AutoNMT's dependencies.
- Needs a fork to fix a few things.
- Driven through the command line, which is fragile.
- Hangs after the first epoch when used with W&B.
- Data-parallel bug: ~200% slower when launched from Python vs the CLI.
