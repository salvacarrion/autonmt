# *********************************
# ************ FAIRSEQ ************
# *********************************

# Default fairseq models and setups
FAIRSEQ_TRANSFORMER_SMALL_1 = [
    "--arch transformer",
    "--encoder-embed-dim 256",
    "--decoder-embed-dim 256",
    "--encoder-layers 3",
    "--decoder-layers 3",
    "--encoder-attention-heads 8",
    "--decoder-attention-heads 8",
    "--encoder-ffn-embed-dim 512",
    "--decoder-ffn-embed-dim 512",
    "--dropout 0.1",
]
FAIRSEQ_TRAINING_1 = [
    "--lr 0.001",
    "--optimizer adam",
    "--criterion cross_entropy",
    "--max-tokens 4096",
    # "--batch-size 128",
    "--max-epoch 10",
    "--clip-norm 1.0",
    "--update-freq 1",
    "--patience 10",
    "--seed 1234",
    # "--warmup-updates 4000",
    # "--lr-scheduler reduce_lr_on_plateau",
    "--no-epoch-checkpoints",
    "--maximize-best-checkpoint-metric",
    "--best-checkpoint-metric bleu",
    "--eval-bleu",
    "--eval-bleu-args '{\"beam\": 5}'",
    "--eval-bleu-print-samples",
    "--scoring sacrebleu",
    "--log-format simple",
    "--task translation",
    # "--num-workers $(nproc)",
]
FAIRSEQ_1 = FAIRSEQ_TRANSFORMER_SMALL_1 + FAIRSEQ_TRAINING_1
