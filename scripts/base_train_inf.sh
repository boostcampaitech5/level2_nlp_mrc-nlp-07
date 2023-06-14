MODEL_PATH="./models/base"
OUTPUT_PATH="./outputs/base"
WANDB_NAME="0_base"
RETRIEVER="tfidf" # ["tfidf", "dpr", "bm25plus"]

python train.py --output_dir ${MODEL_PATH} \
                --wandb_name ${WANDB_NAME} \
                --do_train --do_eval \
                --model_name_or_path "klue/bert-base" \
                --save_strategy steps \
                --save_steps 500 \
                --save_total_limit 1 \
                --logging_steps 100 \
                --evaluation_strategy steps \
                --eval_steps 500 \
                --per_device_train_batch_size 8 \
                --per_device_eval_batch_size 8 \
                --learning_rate 5e-5 \
                --weight_decay 0.0 \
                --num_train_epochs 3.0 \
                --lr_scheduler_type linear \
                --warmup_ratio 0.0 \
                --retriever ${RETRIEVER} \

python inference.py --output_dir ${OUTPUT_PATH} \
                    --do_predict --dataset_name ../data/test_dataset/ \
                    --model_name_or_path ${MODEL_PATH} \
                    --retriever ${RETRIEVER} \
                    --top_k_retrieval 10

######## TrainingArguments ######
# --fp16
# --*_strategy ["no", "steps", "epoch"]
# --lr_scheduler_type ["linear", ""]
# --report_to ["all", "wandb", "none"] -> wandb 아닌 걸로 바꾸면 오류 발생 가능성 높음(확인 안 됨)

######## DataTrainingArguments #####
# --max_seq_length 384
# --pad_to_max_length False
# --doc_stride 128
# --max_answer_length 30

# --num_clusters 64
# --use_faiss False


