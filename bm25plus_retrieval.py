import json
import os
import pickle
import time
from contextlib import contextmanager
from typing import List, Optional, Tuple, Union

import faiss
import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
from transformers import AutoTokenizer
# from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm
import wandb
import datetime
from pytz import timezone
from utils import retieval_logging
from rank_bm25 import BM25Plus

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class BM25PlusRetriever:
    def __init__(
        self,
        model_name_or_path: Optional[str] = "klue/bert-base",
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json"
    ) -> None:
        """
        Arguments:

            data_path:
                데이터가 보관되어 있는 경로입니다.

            context_path:
                Passage들이 묶여있는 파일명입니다.

            data_path/context_path가 존재해야합니다.

        Summary:
            Passage 파일을 불러오고 BM25Plus를 선언하는 기능을 합니다.
        """

        self.data_path = data_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path) # default fast tokenizer if available
        self.contexts = None
        self.ids = None
        self.tokenized_contexts = None
        self.bm25plus = None
        
        context_name = context_path.split(".")[0]
        tokenized_wiki_path = os.path.join(data_path, context_name+"_tokenized.bin")
        print(f"context path:{os.path.join(data_path, context_path)}, tokenized context path: {tokenized_wiki_path}")        
        
        # contexts load
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        # contexts 중복 제거
        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))
        
        # context 미리 토큰화
        # tokenized_wiki_path = {context_name}_tokenized.bin
        if os.path.isfile(tokenized_wiki_path):
            # 토큰화 된 context 파일 이미 존재
            with open(tokenized_wiki_path, "rb") as file:
                self.tokenized_contexts = pickle.load(file)
            print("Contexts pickle loaded.")
        else: 
            # 토큰화 된 context 파일 없으면 생성
            self.tokenized_contexts = []
            for doc in tqdm(self.contexts, desc="Tokenizing context"):
                self.tokenized_contexts.append(self.tokenizer(doc)["input_ids"][1:-1])
                # 앞 [CLS], 뒤  [SEP] 토큰 제외
            with open(tokenized_wiki_path, "wb") as f:
                pickle.dump(self.tokenized_contexts, f)
            print("Tokenized contexts pickle saved.")
            
        # Fit tokenized_contexts to BM25Plus
        self.bm25plus = BM25Plus(self.tokenized_contexts)
        print("Tokenized contexts fitted to BM25Plus.")
        

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                각 query에 대해 `get_relevant_doc`을 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List) \n
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.\n
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
        """

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):
            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("BM25+ retrieval"):
                doc_scores = []
                doc_indices = []
                for query in tqdm(query_or_dataset["question"], desc="Retrieving documents for all questions"):
                    scores, indices = self.get_relevant_doc(query, k=topk)
                    doc_scores.append(scores)
                    doc_indices.append(indices)

            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Building result DataFrame")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]

                # 평가를 위한 df column 만들어주기
                for i in range(topk):
                    tmp["context" + str(i + 1)] = self.contexts[doc_indices[idx][i]]
                tmp['score'] = doc_scores[idx][:]

                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        """

        tokenized_query = self.tokenizer(query)["input_ids"][1:-1]

        scores = [(val, idx) for idx, val in enumerate(self.bm25plus.get_scores(tokenized_query))]
        scores.sort(reverse=True)
        scores = scores[:k]
            
        doc_scores = [val for val, _ in scores]
        doc_indices = [idx for _, idx in scores]
            
        return doc_scores, doc_indices

      
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--wandb_name", default="bm25plus", type=str, help="Name of wandb run"
    )
    parser.add_argument(
        "--dataset_name", default="../data/train_dataset", type=str, help=""
    )
    parser.add_argument(
        "--model_name_or_path",
        default="klue/bert-base",
        type=str,
        help="",
    )
    parser.add_argument("--data_path", default="../data", type=str, help="")
    parser.add_argument(
        "--context_path", default="wikipedia_documents.json", type=str, help=""
    )
    parser.add_argument("--use_faiss", default=False, type=bool, help="")

    args = parser.parse_args()
    print(args)
    
    wandb_name = args.wandb_name

    wandb.init(
        project="nlp07_mrc_retrieval_score",
        name=wandb_name
        + "_"
        + datetime.datetime.now(timezone("Asia/Seoul")).strftime("%m/%d %H:%M"),
    )

    # concat train + valid dataset
    org_dataset = load_from_disk(args.dataset_name)
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    print("*" * 40, "query dataset", "*" * 40)
    print(full_ds)
    
    bm25plus_retriever = BM25PlusRetriever(
        model_name_or_path=args.model_name_or_path,
        data_path=args.data_path,
        context_path=args.context_path
    )

    with timer("bulk query by exhaustive search"):
        
        top_k = 50  # ground truth 를 확인할 passage 개수
        term = 5  # term 단위로 확인함
        
        # 10 에서 부터 1/2 씩 감소 시키면서 계산 check_passage_cnt, term
        weight = [(1 / 2**i) for i in range(0, top_k // term)]

        # retriever.get_sparse_embedding()
        df = bm25plus_retriever.retrieve(full_ds, topk=top_k)
        df["correct"] = df["original_context"] == df["context"]
        print(
            "correct retrieval result by exhaustive search",
            df["correct"].sum() / len(df),
        )

        # check retrieval
        retieval_logging.retrieval_check(df, weight, top_k, term)

    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

    with timer("single query by exhaustive search"):
        scores, indices = bm25plus_retriever.retrieve(query, topk=top_k)
