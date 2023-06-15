from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    project_name: str = field(
        default="nlp07_mrc", metadata={"help": "Wandb project name"}
    )
    wandb_name: str = field(
        default="", metadata={"help": "Wandb run name. Format should be: \{num\}_\{run-name\}"}
    )
    model_name_or_path: str = field(
        default="klue/bert-base",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    encoder_base: Optional[str] = field(
        default="klue/bert-base",
        metadata={"help": "Base model of Encoder"},
    )
    q_encoder_path: Optional[str] = field(
        default="./encoders/p_encoder/",
        metadata={"help": "Name of q encoder to be saved"},
    )
    p_encoder_path: Optional[str] = field(
        default="./encoders/q_encoder/",
        metadata={"help": "Name of p encoder to be saved"},
    )
    p_encoder_ckpt: Optional[str] = field(
        default="./encoders/p_encoder/",
        metadata={"help": "Name of p encoder to load"},
    )
    q_encoder_ckpt: Optional[str] = field(
        default="./encoders/q_encoder",
        metadata={"help": "Name of p encoder to load"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default="../data/train_dataset",
        metadata={"help": "The name of the dataset to use."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={
            "help": "When splitting up a long document into chunks, how much stride to take between chunks."
        },
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    eval_retrieval: bool = field(
        default=True,
        metadata={"help": "Whether to run passage retrieval using sparse embedding."},
    )
    num_clusters: int = field(
        default=64, metadata={"help": "Define how many clusters to use for faiss."}
    )
    top_k_retrieval: int = field(
        default=10,
        metadata={
            "help": "Define how many top-k passages to retrieve based on similarity."
        },
    )
    use_faiss: bool = field(
        default=False, metadata={"help": "Whether to build with faiss"}
    )
    use_dense: bool = field(default=False, metadata={"help": "Whether to use DPR"})
    retriever: str = field(
        default="tfidf", metadata={"help": "type of retriever to use"}
    )
