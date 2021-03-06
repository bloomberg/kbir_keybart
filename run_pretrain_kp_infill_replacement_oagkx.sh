BASE_RESOURCES=<ADD-YOUR-RESOURCES-FOLDER>

run pretrain_runner.py \
    --roberta-tokenizer-dir ${BASE_RESOURCES}/infill-replacement-models/checkpoint-130000 \
    --roberta-mlm-model-dir ${BASE_RESOURCES}/infill-replacement-models/checkpoint-130000 \
    --train-data-dir ${BASE_RESOURCES}/resources/oagkx \
    --eval-data-dir ${BASE_RESOURCES}/resources/oagkx_eval \
    --keyphrase-universe ${BASE_RESOURCES}/resources/oagkx_keyphrase_universe/keyphrase_universe.txt \
    --keyphrase-universe-size 500000 \
    --train-batch-size 2 \
    --eval-batch-size 2 \
    --learning-rate 1e-5 \
    --adam-epsilon 1e-6 \
    --max-steps 260000 \
    --save-steps 10000 \
    --eval-steps 340000 \
    --logging-steps 1000 \
    --warmup-steps 2500 \
    --mlm-probability 0.05 \
    --keyphrase-mask-percentage 0.2 \
    --keyphrase-replace-percentage 0.4 \
    --do-train \
    --do-eval \
    --do-keyphrase-infilling \
    --do-keyphrase-replacement \
    --task KLM \
    --eval-task KLM \
    --max-mask-keyphrase-pairs 10 \
    --max-keyphrase-pairs 20 \
    --kp-max-seq-len 10 \
    --mlm-loss-weight 1.0 \
    --keyphrase-infill-loss-weight 0.3 \
    --infill-num-tok-loss-weight 1.0 \
    --replacement-loss-weight 2.0 \
    --model-dir ${BASE_RESOURCES}/infill-replacement-models/ \
    --dataloader-num-workers 5 \
