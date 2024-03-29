"""
Open-Domain Question Answering 을 수행하는 inference 코드 입니다.

대부분의 로직은 train.py 와 비슷하나 retrieval, predict 부분이 추가되어 있습니다.
"""


import logging
import sys
from typing import Callable, Dict, List, Tuple

import numpy as np
from arguments import DataTrainingArguments, ModelArguments
from datasets import Dataset, DatasetDict, Features, Sequence, Value, load_from_disk
import evaluate
from retrieval import SparseRetrieval
from dense_retrieval import DenseRetriever
from bm25plus_retrieval import BM25PlusRetriever
from trainer_qa import QuestionAnsweringTrainer
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from utils_qa import check_no_error, postprocess_qa_predictions


logger = logging.getLogger(__name__)


def main():
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.do_train = True

    print(f"retriever uses {data_args.retriever}")
    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    logger.info("Training/evaluation parameters %s", training_args)

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(training_args.seed)

    datasets = load_from_disk(data_args.dataset_name)
    print(datasets)

    # AutoConfig를 이용하여 pretrained model 과 tokenizer를 불러옵니다.
    # argument로 원하는 모델 이름을 설정하면 옵션을 바꿀 수 있습니다.
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        use_fast=True,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
    )
    if data_args.retrieval_split:
        print("Using split retrieval")

    # True일 경우 : run passage retrieval
    if data_args.eval_retrieval:
        if data_args.retriever in ["tfidf", "bm25plus"]:  # default tfidf
            datasets = run_sparse_retrieval(
                tokenizer.tokenize,
                datasets,
                training_args,
                data_args,
                retrieval_split=data_args.retrieval_split,
                retrieval_result_save=data_args.retrieval_result_save,
            )
        elif data_args.use_dense or data_args.retriever == "dpr":
            datasets = run_dense_retrieval(
                datasets,
                training_args,
                data_args,
                p_encoder_ckpt=model_args.p_encoder_ckpt,
                q_encoder_ckpt=model_args.q_encoder_ckpt,
                model_name_or_path=model_args.encoder_base,
                stage="test",
                use_HFBert=model_args.use_HFBert,
                retrieval_split=data_args.retrieval_split,
                retrieval_result_save=data_args.retrieval_result_save,
            )

    # eval or predict mrc model
    if training_args.do_eval or training_args.do_predict:
        if data_args.retrieval_split:
            run_split_mrc(
                data_args, training_args, model_args, datasets, tokenizer, model
            )
        else:
            run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)


def run_sparse_retrieval(
    tokenize_fn: Callable[[str], List[str]],
    datasets: DatasetDict,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    data_path: str = "../data",
    context_path: str = "wikipedia_documents.json",
    retrieval_split=False,
    retrieval_result_save=False,
) -> DatasetDict:
    # Query에 맞는 Passage들을 Retrieval 합니다.

    if data_args.retriever == "tfidf":
        retriever = SparseRetrieval(
            tokenize_fn=tokenize_fn,
            data_path=data_path,
            context_path=data_args.context_path,
        )
        retriever.get_sparse_embedding()

        if data_args.use_faiss:
            retriever.build_faiss(num_clusters=data_args.num_clusters)
            df = retriever.retrieve_faiss(
                datasets["validation"], topk=data_args.top_k_retrieval
            )
        else:
            df = retriever.retrieve(
                datasets["validation"], topk=data_args.top_k_retrieval
            )
    elif data_args.retriever == "bm25plus":
        bm25plus_retriever = BM25PlusRetriever(
            model_name_or_path="klue/bert-base",
            data_path=data_path,
            context_path=data_args.context_path,
            retrieval_split=retrieval_split,
        )
        df = bm25plus_retriever.retrieve(
            datasets["validation"],
            topk=data_args.top_k_retrieval,
            retrieval_result_save=retrieval_result_save,
            output_dir=training_args.output_dir,
        )

    # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
    if training_args.do_predict:
        f = Features(
            {
                # split retrieval 시 사용
                "context": Sequence(
                    feature=Value(dtype="string", id=None),
                    length=-1,
                    id=None,
                )
                if retrieval_split
                else Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
        df = df[["context", "id", "question"]]

    # train data 에 대해선 정답이 존재하므로 id question context answer 로 데이터셋이 구성됩니다.
    elif training_args.do_eval:
        f = Features(
            {
                "answers": Sequence(
                    feature={
                        "text": Value(dtype="string", id=None),
                        "answer_start": Value(dtype="int32", id=None),
                    },
                    length=-1,
                    id=None,
                ),
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
        df = df[["answers", "context", "id", "question"]]

    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return datasets


def run_dense_retrieval(
    datasets: DatasetDict,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    data_path: str = "../data/test_dataset",
    context_path: str = "wikipedia_documents.json",
    model_name_or_path: str = "klue/roberta-large",
    p_encoder_ckpt: str = None,
    q_encoder_ckpt: str = None,
    stage="train",
    use_HFBert=False,
    retrieval_split=False,
    retrieval_result_save=False,
) -> DatasetDict:
    # Query에 맞는 Passage들을 Retrieval 합니다.
    if p_encoder_ckpt == None and q_encoder_ckpt == None:
        p_encoder_ckpt = model_name_or_path
        q_encoder_ckpt = model_name_or_path

    retriever = DenseRetriever(
        data_path=data_path,
        context_path=os.path.join(data_path, data_args.context_path),
        model_name_or_path=model_name_or_path,
        p_encoder_ckpt=p_encoder_ckpt,
        q_encoder_ckpt=q_encoder_ckpt,
        stage="test",
        use_HFBert=use_HFBert,
        retrieval_split=retrieval_split,
    )
    retriever.get_dense_embedding()

    df = retriever.retrieve(
        datasets["validation"],
        topk=data_args.top_k_retrieval,
        retrieval_result_save=retrieval_result_save,
        output_dir=training_args.output_dir,
    )

    # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
    if training_args.do_predict:
        f = Features(
            {
                "context": Sequence(
                    feature=Value(dtype="string", id=None),
                    length=-1,
                    id=None,
                )
                if retrieval_split
                else Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
        df = df[["context", "id", "question"]]

    # train data 에 대해선 정답이 존재하므로 id question context answer 로 데이터셋이 구성됩니다.
    elif training_args.do_eval:
        f = Features(
            {
                "answers": Sequence(
                    feature={
                        "text": Value(dtype="string", id=None),
                        "answer_start": Value(dtype="int32", id=None),
                    },
                    length=-1,
                    id=None,
                ),
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
        df = df[["answers", "context", "id", "question"]]

    df.to_csv("./dense_retrieval_result.csv", index=False)
    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return datasets


def run_mrc(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
) -> None:
    # eval 혹은 prediction에서만 사용함
    column_names = datasets["validation"].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding에 대한 옵션을 설정합니다.
    # (question|context) 혹은 (context|question)로 세팅 가능합니다.
    pad_on_right = tokenizer.padding_side == "right"

    # 오류가 있는지 확인합니다.
    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )

    # Validation preprocessing / 전처리를 진행합니다.
    def prepare_validation_features(examples):
        # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
        # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=False,  # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # 길이가 긴 context가 등장할 경우 truncate를 진행해야하므로, 해당 데이터셋을 찾을 수 있도록 mapping 가능한 값이 필요합니다.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # evaluation을 위해, prediction을 context의 substring으로 변환해야합니다.
        # corresponding example_id를 유지하고 offset mappings을 저장해야합니다.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # sequence id를 설정합니다 (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # 하나의 example이 여러개의 span을 가질 수 있습니다.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # context의 일부가 아닌 offset_mapping을 None으로 설정하여 토큰 위치가 컨텍스트의 일부인지 여부를 쉽게 판별할 수 있습니다.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        return tokenized_examples

    eval_dataset = datasets["validation"]

    # Validation Feature 생성
    eval_dataset = eval_dataset.map(
        prepare_validation_features,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    # Data collator
    # flag가 True이면 이미 max length로 padding된 상태입니다.
    # 그렇지 않다면 data collator에서 padding을 진행해야합니다.
    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )

    # Post-processing:
    def post_processing_function(
        examples,
        features,
        predictions: Tuple[np.ndarray, np.ndarray],
        training_args: TrainingArguments,
    ) -> EvalPrediction:
        # Post-processing: start logits과 end logits을 original context의 정답과 match시킵니다.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=data_args.max_answer_length,
            output_dir=training_args.output_dir,
        )
        # Metric을 구할 수 있도록 Format을 맞춰줍니다.
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()
        ]

        if training_args.do_predict:
            return formatted_predictions
        elif training_args.do_eval:
            references = [
                {"id": ex["id"], "answers": ex[answer_column_name]}
                for ex in datasets["validation"]
            ]

            return EvalPrediction(
                predictions=formatted_predictions, label_ids=references
            )

    metric = evaluate.load("squad")

    def compute_metrics(p: EvalPrediction) -> Dict:
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    print("init trainer...")
    # Trainer 초기화
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=eval_dataset,
        eval_examples=datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    logger.info("*** Evaluate ***")

    #### eval dataset & eval example - predictions.json 생성됨
    if training_args.do_predict:
        predictions = trainer.predict(
            test_dataset=eval_dataset, test_examples=datasets["validation"]
        )

        # predictions.json 은 postprocess_qa_predictions() 호출시 이미 저장됩니다.
        print(
            "No metric can be presented because there is no correct answer given. Job done!"
        )

    if training_args.do_eval:
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)


class QADataset(Dataset):
    def __init__(self, p_inputs):
        self.input_ids = p_inputs["input_ids"]
        self.attention_mask = p_inputs["attention_mask"]
        self.offset_mapping = p_inputs["offset_mapping"]
        self.original_context_idx = p_inputs["original_context_idx"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        input_ids = self.input_ids[index].to("cuda")
        attention_mask = self.attention_mask[index].to("cuda")
        offset_mapping = self.offset_mapping[index].to("cuda")
        original_context_idx = self.original_context_idx[index]
        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "offset_mapping": offset_mapping,
            "original_context_idx": original_context_idx,
        }
        return item


import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import re
import json


def run_split_mrc(data_args, training_args, model_args, datasets, tokenizer, model):
    questions = datasets["validation"]["question"]
    contexts = datasets["validation"]["context"]
    q_ids = datasets["validation"]["id"]
    model = model.to("cuda").eval()
    # 파라미터
    os.makedirs(training_args.output_dir, exist_ok=True)
    batch_size = 8
    topk = len(contexts[0])

    answer_dict = {}
    n_best_dict = {}
    n_best_size = 20
    p_bar = tqdm(range(len(datasets["validation"])))
    for q_idx in p_bar:
        # 테스트할 문항 하나씩 가져오기
        answer = ""
        query = questions[q_idx]
        context = contexts[q_idx]  # [passage-1, passage-2 , ... , passage-k]

        input_ids = None
        attention_mask = None
        offset_mapping = None
        original_context_idx = []

        # query와 각 context를 합쳐 각각 토큰화 후 하나의 tensor로 concat
        # context끼리 concat 되는 일 없이 한 텐서 안에는 단일 passage만 존재
        for k in range(topk):
            tokens = tokenizer(
                query,
                context[k],
                truncation="only_second",
                max_length=data_args.max_seq_length,
                stride=data_args.doc_stride,
                return_overflowing_tokens=True,
                padding="max_length",
                return_tensors="pt",
                return_offsets_mapping=True,
            )
            if k == 0:
                input_ids = tokens["input_ids"]
                attention_mask = tokens["attention_mask"]
                offset_mapping = tokens["offset_mapping"]
                # truncation 되면 여러 passage가 생기므로 그 개수만큼 context_idx 추가
                original_context_idx.extend(
                    [k for _ in range(len(tokens["input_ids"]))]
                )
            else:
                input_ids = torch.concat((input_ids, tokens["input_ids"]))
                attention_mask = torch.concat(
                    (attention_mask, tokens["attention_mask"])
                )
                offset_mapping = torch.concat(
                    (offset_mapping, tokens["offset_mapping"])
                )
                original_context_idx.extend(
                    [k for _ in range(len(tokens["input_ids"]))]
                )

        # 입력 데이터 구성
        input_data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "offset_mapping": offset_mapping,
            "original_context_idx": original_context_idx,
        }

        # 데이터셋 및 데이터로더 구성
        valid_dataset = QADataset(input_data)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)
        n_best_list = []
        max_logit = 0
        for batch_idx, batch in enumerate(valid_dataloader):
            # 모델에 안 들어가도 될 입력(답변 구할 때 필요한 정보)은 pop하여 빼주기
            original_context_idx = batch.pop("original_context_idx")
            offset_mapping = batch.pop("offset_mapping")

            # 모델 forward
            outputs = model(**batch)

            # 배치마다 담긴 길이가 다르므로 구해주기
            batch_len = len(outputs["start_logits"])

            # 각 query+passage 쌍에서의 start와 end의 max, argmax 구하기
            # 총 batch_len개의 max, argmax 존재
            s_max = outputs["start_logits"].max(dim=1)
            e_max = outputs["end_logits"].max(dim=1)

            # 각 query+passage 쌍에서 답변 확률과 위치 구하기
            for idx in range(batch_len):
                # 원래 토큰으로 돌리기 위한 offset
                offsets = offset_mapping[idx]

                # span의 확률
                start_logit = s_max.values[idx].item()
                end_logit = e_max.values[idx].item()
                logit = start_logit + end_logit
                s_pos = offsets[s_max.indices[idx].item()][0]
                e_pos = offsets[e_max.indices[idx].item()][1]
                original_context = context[original_context_idx[idx]]
                text = original_context[s_pos:e_pos]

                result = {
                    "start_logit": start_logit,
                    "end_logit": end_logit,
                    "text": text,
                    "score": start_logit + end_logit,
                }
                n_best_list.append(result)

                if max_logit < logit:
                    # 답변의 길이가 0이거나 [CLS]토큰이 답변이 된 케이스들 제외
                    if s_pos == e_pos:
                        continue

                    # 끝나는 위치가 시작점보다 앞에 위치한 케이스 제외
                    if e_pos < s_pos:
                        continue

                    # 너무 긴 답변 제외
                    if e_pos - s_pos > 30:
                        continue

                    max_logit = logit
                    answer = original_context[s_pos:e_pos]

        # GPU 공간을 위해 cache 비워주기
        torch.cuda.empty_cache()

        # answer 후처리
        answer = answer.strip()
        answer = re.sub(r"\\", "", answer)
        answer = re.sub(r'""?', '"', answer)
        answer = re.sub(r'^"|"$', "", answer)

        # 진행 상황 볼 수 있게 postfix로 답변 보여주기
        p_bar.set_postfix(answer=answer)
        # n_best_list.sort(key=lambda x: x["score"])
        # n_best_list = n_best_list[:n_best_size]
        predictions = sorted(n_best_list, key=lambda x: x["score"], reverse=True)[
            :n_best_size
        ]

        scores = np.array([x.pop("score") for x in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob

        # 답변 추가하기
        answer_dict[q_ids[q_idx]] = answer
        n_best_dict[q_ids[q_idx]] = predictions

    # 답변 저장하기
    with open(
        os.path.join(training_args.output_dir, "predictions.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(answer_dict, f, indent=4, ensure_ascii=False)
    with open(
        os.path.join(training_args.output_dir, "nbest_predictions.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(n_best_dict, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
