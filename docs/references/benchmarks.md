# Benchmarks


## AutoNMT v0.4


### Tests (09/04/2022)

Comparison of AutoNMT (custom toolkit) and Fairseq (forked: 31d94f556bd49bc7e61511adbda482b2c54652b5). 
The compared models are not exactly the same but the results should be similar.


- **Params:**
  - **Dataset:** Multi30k, Lowercase
  - **Model:** Small transformer (256emb/3l/8h/512ffn/0.1do). (autonmt: 7.0M params; fairseq: 7,025,664)
  - **Training:** max_epochs=[1, 5, 10], batch_size=128, max_tokens=None, lr=0.001, optimizer=adam, seed=1234, patience=10, num_workers=12
  - **Predict:** Beam 1, sacrebleu_bleu, correct unknowns

**Exp. 1: Hardware (local):** 1 GPU GeForce GTX 1070, AMD Ryzen 7 2700X Eight-Core Processor
```text
max_epochs=10
# AutoNMT toolkit
train_dataset eval_dataset lang_pair subword_model vocab_size  autonmt_bleu  Training time   Translate time (beam=1)
  cf/multi30k  cf/multi30k     de-en       unigram       4000     33.817927  0:04:18.731907  0:00:02.005633
  cf/multi30k  cf/multi30k     de-en          word       4000     34.556056  0:03:55.216156  0:00:01.819264


max_epochs=10
# Fairseq toolkit
train_dataset eval_dataset lang_pair subword_model vocab_size  fairseq_bleu  Training time   Translate time (beam=1)
  cf/multi30k  cf/multi30k     de-en       unigram       4000     35.671859  0:02:37.182229  0:00:06.438150
  cf/multi30k  cf/multi30k     de-en          word       4000     34.991464  0:02:28.685353  0:00:05.941582

```

**Exp. 2: Hardware (remote):** 2 GPU NVIDIA TITAN XP, Intel(R) Core(TM) i7-7800X CPU @ 3.50GHz
```text
max_epochs=10
# AutoNMT toolkit
train_dataset eval_dataset lang_pair subword_model vocab_size  autonmt_bleu  Training time   Translate time (beam=1)
  cf/multi30k  cf/multi30k     de-en       unigram       4000     35.171231  0:02:50.929962  0:00:01.328614
  cf/multi30k  cf/multi30k     de-en          word       4000     34.698708  0:02:54.390875  0:00:01.209333


max_epochs=10 | 2 GPUs
# Fairseq toolkit (no venv) => BUG!  // (2248MiB + 2090MiB)
train_dataset eval_dataset lang_pair subword_model vocab_size fairseq_bleu  Training time   Translate time (beam=1)
  cf/multi30k  cf/multi30k     de-en       unigram       4000    35.520457  0:12:14.061107  0:00:06.639795
  
max_epochs=10 | 2 GPUs
# Fairseq toolkit (with venv)  // (1960MiB + 2090MiB)
train_dataset eval_dataset lang_pair subword_model vocab_size fairseq_bleu  Training time   Translate time (beam=1)
  cf/multi30k  cf/multi30k     de-en       unigram       4000    35.520457  0:05:18.730773  0:00:06.789658
  
max_epochs=10 | 1 GPUs
# Fairseq toolkit (with venv)
train_dataset eval_dataset lang_pair subword_model vocab_size fairseq_bleu  Training time   Translate time (beam=1)
  cf/multi30k  cf/multi30k     de-en       unigram       4000    36.50962  0:02:13.011535  0:00:06.667130
```

**Conclusions:**
- **AutoNMT:** Bucketing is need to speed-up training
- **AutoNMT:** Max-tokens (dynamic batching) is a desirable feature to have (we know truncate is needed, losing part of the batch)
- **AutoNMT:** Iterative decoding is needed to use beam search in real use cases
- **AutoNMT:** Warm-up is needed to speed-up the convergence of the Transformer
- **Fairseq:** There is a bug related to data parallelization that makes the code 200% when launched from Python w.r.t the command line.


### Tests (06/01/2021)

These values should be taken with a grain of salt as the models are not exactly the same and some training parameters 
are still not supported by AutoNMT since this is an early version.

- **Params:**
  - **Dataset:** Multi30k, Lowercase
  - **Model:** Small transformer (256emb/3l/8h/512ffn/0.1do). (autonmt: 7.0M params; fairseq: 7,025,664)
  - **Training:** max_epochs=[1, 5, 10], batch_size=128, max_tokens=None, lr=0.001, optimizer=adam, seed=1234, patience=10, num_workers=12
  - **Predict:** Beam 1, sacrebleu_bleu, correct unknowns
  - **Hardware:** 1 GPU GeForce GTX 1070, AMD Ryzen 7 2700X Eight-Core Processor

**Results: (Custom v0.2a)**
```text
max_epochs=1
train_dataset  eval_dataset subword_model vocab_size  autonmt_bleu  Training time   Translate time (beam=1)
multi30k_test multi30k_test       unigram       4000      5.559104  0:00:26.414141  0:00:02.620312
multi30k_test multi30k_test          word       4000      7.392883  0:00:21.968405  0:00:17.251382

max_epochs=5
train_dataset  eval_dataset subword_model vocab_size  autonmt_bleu  Training time   Translate time (beam=1)
multi30k_test multi30k_test       unigram       4000     29.632340  0:02:15.064932  0:00:08.767339
multi30k_test multi30k_test          word       4000     30.972321  0:01:57.834483  0:00:12.347725

max_epochs=10
train_dataset  eval_dataset subword_model vocab_size  autonmt_bleu  Training time   Translate time (beam=1)
multi30k_test multi30k_test       unigram       4000     32.816378  0:04:11.125762  0:00:08.315648
multi30k_test multi30k_test          word       4000     34.682657  0:03:38.593067  0:00:02.082928
```

**Results: (Fairseq 1.0a)**
```text
max_epochs=1
train_dataset  eval_dataset subword_model vocab_size  fairseq_bleu  Training time   Translate time (beam=1)
multi30k_test multi30k_test       unigram       4000     13.707765  0:00:21.697430  0:00:11.276265
multi30k_test multi30k_test          word       4000     10.338827  0:00:19.449526  0:00:11.645491

max_epochs=5
train_dataset  eval_dataset subword_model vocab_size  fairseq_bleu  Training time   Translate time (beam=1)
multi30k_test multi30k_test       unigram       4000     33.460104  0:01:28.053523  0:00:10.084800
multi30k_test multi30k_test          word       4000     33.347637  0:01:17.936529  0:00:10.063523

max_epochs=10
train_dataset  eval_dataset subword_model vocab_size  fairseq_bleu  Training time   Translate time (beam=1)
multi30k_test multi30k_test       unigram       4000     35.123375  0:02:38.338926  0:00:09.955724
multi30k_test multi30k_test          word       4000     34.706139  0:02:31.938635  0:00:10.285918
```

**Conclusions:**
- **AutoNMT:** Bucketing is need to speed-up training
- **AutoNMT:** Max-tokens (dynamic batching) is a desirable feature to have (we know truncate is needed, losing part of the batch)
- **AutoNMT:** Iterative decoding is needed to use beam search in real use cases
- **AutoNMT:** Warm-up is needed to speed-up the convergence of the Transformer
- **Fairseq:** has many bugs and incompatibilities so it cannot be installed with the rest of our dependencies
- **Fairseq:** needs a fork to fix a few things
- **Fairseq:** is used through the command line and it can be problematic
- **Fairseq:** hangs after the first epoch if used with WandB

