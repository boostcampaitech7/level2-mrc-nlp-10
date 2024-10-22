from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TF_IDF_retrieval_arguments:
    data_route: Optional[str] = field(
        default = '/data/ephemeral/home/practice/data/train_dataset',
        metadata = {'help' : "데이터셋 위치입니다."},
    )
    test_data_route : Optional[str] = field(
        default = '/data/ephemeral/home/practice/data/test_dataset',
        metadata = {'help' : '테스트 데이터셋 위치입니다.'}
    )
    k : int = field(
        default = 30,
        metadata = {'help' : '비슷한 문서 중 몇개를 내보낼지를 결정합니다.'}
    )
    num_clusters : int = field(
        default = 128,
        metadata = {'help' : 'faiss 클러스터를 몇개로 할 지 정합니다.'}
    )
    wiki_route : str = field(
        default = '/data/ephemeral/data/wikipedia_documents.json',
        metadata = {'help' : '위키데이터의 경로입니다.'}
    )
    model_name : str = field(
        default = "klue/bert-base",
        metadata = {'help' : 'corpus를 나눌 토크나이저를 정합니다.'}
    )
    data_path : str = field(
        default = './retrieval_result',
        metadata = {'help' : 'retrieval_result를 저장할 경로입니다.'}
    )

#  ------------------------------------------------------------------

@dataclass
class BM25_retrieval_arguments:
    data_route: Optional[str] = field(
        default='/data/ephemeral/home/practice/data/train_dataset',
        metadata={'help': "데이터셋 위치입니다."}
    )
    test_data_route: Optional[str] = field(
        default='/data/ephemeral/home/practice/data/test_dataset',
        metadata={'help': '테스트 데이터셋 위치입니다.'}
    )
    k: int = field(
        default = 30,
        metadata={'help': '비슷한 문서 중 몇 개를 내보낼지를 결정합니다.'}
    )
    wiki_route: str = field(
        default = '/data/ephemeral/home/practice/data/wikipedia_documents.json',
        metadata={'help': '위키 데이터의 경로입니다.'}
    )
    data_path: str = field(
        default='./bm25_retrieval_result',
        metadata={'help': 'BM25 검색 결과를 저장할 경로입니다.'}
    )
    bm25_tokenizer: str = field(
        default="klue/roberta-small",
        metadata={'help': 'BM25 검색에서 사용할 토크나이저를 설정합니다.'}
    )
    model_name: str = field(
        default="klue/bert-base",
        metadata={'help': '토크나이저를 지정합니다. BM25에서도 동일하게 사용할 수 있습니다.'}
    )


#  ------------------------------------------------------------------

@dataclass
class Dense_search_retrieval_arguments:
    data_route : Optional[str] = field(
        default = '/data/ephemeral/home/practice/data/train_dataset',
        metadata = {'help' : "데이터셋 위치입니다."},
    )
    test_data_route : Optional[str] = field(
        default = '/data/ephemeral/home/practice/data/test_dataset',
        metadata = {'help' : '테스트 데이터셋 위치입니다.'}
    )
    
    data_path : str = field(
        default = './retrieval_result',
        metadata = {'help' : 'retrieval_result를 저장할 경로입니다.'}
    )
    output_dir : str = field(
        default = './Dense_embedding_retrieval_model_results',
        metadata = {'help' : 'DPR 모델의 저장 경로입니다.'}
    )

    model_name : Optional[str] = field(
        default = "klue/roberta-large",
        metadata = {'help': "모델이름 입니다."}
    )
    wiki_route : str = field(
        default = '/data/ephemeral/home/practice/data/wikipedia_documents.json',
        metadata = {'help' : '위키데이터의 경로입니다.'}
    )
    num_neg : int = field(
        default = 10,
        metadata = {'help': "네거티브 샘플링 몇개할건지를 고릅니다."}
    )
    num_train_epochs : int = field(
        default = 3,
        metadata = {'help': "에폭입니다."}
    )
    kfold : int = field(
        default = 3,
        metadata = {'help' : 'Kfold의 fold 수를 정합니다.'}
    )
    epoch_for_kfold : int = field(
        default = 2,
        metadata = {'help' : '교차검증을 통해 학습을 할 때의 epoch 수입니다.'}
    )
    per_device_train_batch_size : int = field(
        default = 8,
        metadata = {'help': "Dense embedding 모델의 배치사이즈입니다."}
    )
    per_device_eval_batch_size : int = field(
        default = 8,
        metadata = {'help' : 'eval과정에서의 batch size입니다.'}
    )
    learning_rate : float = field(
        default = 3e-5,
        metadata = {'help': "러닝레이트입니다."}
    )
    weight_decay : float = field(
        default = 0.01,
        metadata = {'help': "학습동안 적용할 weightdecay입니다."}
    )
    num_clusters : int = field(
        default = 128,
        metadata = {'help' : 'faiss 클러스터를 몇개로 할 지 정합니다.'}
    )
    top_k : int = field(
        default = 20,
        metadata = {'help' : '몇개의 후보를 faiss indexer에서 뽑을건지 정합니다.'}
    )
    use_wandb : bool = field(
        default = True,
        metadata = {'help' : 'wandb를 '}
    )
    wandb_key : str = field(
        default = "ea26fff0d932bc74bbfad9fd507b292c67444c02",
        metadata = {'help' : 'wandb API 키 입니다.'}
    )


# 용가리 : ea26fff0d932bc74bbfad9fd507b292c67444c02
# 토스트 : d1af552c8639b9bc38ead601ac46df6a86b16c97
#  ------------------------------------------------------------------
 

@dataclass
class Extraction_based_MRC_arguments:
    max_seq_length : int = field(
        default = 384,
        metadata = {'help' : '질문과 컨텍스트, special 토큰을 합한 문자열의 최대 길이입니다.'}
    )
    pad_to_max_length : bool = field(
        default = False,
        metadata = {'help' : '최대 길이의 문장을 기준으로 패딩을 수행합니다.'}
    )
    doc_stride : int = field(
        default = 128,
        metadata = {'help' : '긴 컨텍스트를 나눌 때 앞의 컨텍스트와 얼마나 겹치게 시퀀스를 구성할 것인지 정합니다.'}
    )
    model_name : str = field(
        default = "klue/roberta-large",
        metadata = {'help' : '훈련에 사용할 모델을 정의합니다.'}
    )
    model_path : str = field(
        default = './Extraction_based_MRC_outputs',
        metadata = {'help' : 'Extraction_model의 훈련 결과를 어디에 저장할지 정합니다.'}
    )
    retrieval_results_route : Optional[str] = field(
        default = '/data/ephemeral/home/practice/retrieval_results/TF-IDF_retrieval.csv',
        metadata = {'help' : 'retrieval을 통해 얻어낸 passage의 경로입니다.'}
    )
    output_dir : Optional[str] = field(
        default = "./Extraction_based_MRC_outputs",
        metadata = {'help' : 'Extraction model의 checkpoint ouptut 경로입니다.'}
    )
    preprocessing_num_workers : int = field(
        default = None,
        metadata = {'help' : '프로세스 몇개 쓸건지입니다. 높으면 빠르긴한데 메모리를 많이씁니다.'}
    )
    per_device_train_batch_size : int = field(
        default = 8,
        metadata = {'help': "Extraction_base 모델의 배치사이즈입니다."}
    )
    per_device_eval_batch_size : int = field(
        default = 8,
        metadata = {'help': "Extraction_base 모델의 배치사이즈입니다."}
    )
    num_train_epochs : int = field(
        default = 2,
        metadata = {'help' : '에폭을 정합니다.'}
    )
    n_best_size : int = field(
        default = 40,
        metadata = {'help' : '가능한 답변의 개수를 설정합니다.'}
    )
    max_answer_length : int = field(
        default = 30,
        metadata = {'help' : '답변의 최대 길이를 설정합니다.'}
    )
    data_route: Optional[str] = field(
        default = '/data/ephemeral/home/practice/data/train_dataset',
        metadata = {'help' :"데이터셋 위치입니다."},
    )
    test_data_route : Optional[str] = field(
        default = '/data/ephemeral/Jung/level2-mrc-nlp-10/data/test_dataset',
        metadata = {'help' : '테스트 데이터셋 위치입니다.'}
    )
    doc_stride : int = field(
        default = 128,
        metadata = {'help' : '문장이 너무 길어서 짤리면 앞에서 얼마나 겹쳐서 가져올건지를 정하는 숫자입니다.'}
    )
    num_proc : int = field(
        default = 1,
        metadata = {'help' : '프로세스 몇개 쓸건지입니다. 높으면 빠르긴한데 메모리를 많이씁니다.'}
    )
    load_from_cache_file : bool = field(
        default = True,
        metadata = {'help' : '캐싱해서 빨리하는겁니다. 빨라질 순 있는데 메모리를 잡아먹습니다.'}
    )
    use_wandb : bool = field(
        default = True,
        metadata = {'help' : 'wandb를 '}
    )
    wandb_key : str = field(
        default = "ea26fff0d932bc74bbfad9fd507b292c67444c02",
        metadata = {'help' : 'wandb API 키 입니다.'}
    )
    learning_rate : float = field(
        default = 3e-5,
        metadata = {'help' : '러닝레이트입니다.'}
    )
    kfold : int = field(
        default = 3,
        metadata = {'help' : 'Kfold의 fold 수를 정합니다.'}
    )
    epoch_for_kfold : int = field(
        default = 2,
        metadata = {'help' : '교차검증을 통해 학습을 할 때의 epoch 수입니다.'}
    )
#  ------------------------------------------------------------------


@dataclass
class Generation_based_MRC_arguments:
    data_route : Optional[str] = field(
        default = '/data/ephemeral/home/practice/data/train_dataset',
        metadata = {'help' : "훈련 데이터셋 위치입니다."},
    )
    test_data_route : Optional[str] = field(
        default = '/data/ephemeral/home/practice/data/test_dataset',
        metadata = {'help' : '테스트 데이터셋 위치입니다.'}
    )
    retrieval_results_route : Optional[str] = field(
        default = '/data/ephemeral/home/practice/retrieval_results/TF-IDF_retrieval.csv',
        metadata = {'help' : 'retrieval을 통해 얻어낸 passage의 경로입니다.'}
    )
    output_dir : Optional[str] = field(
        default = "./Generation_based_MRC_outputs",
        metadata = {'help' : 'Generation base mrc model의 checkpoint ouptut 경로입니다.'}
    )
    use_wandb : bool = field(
        default = True,
        metadata = {'help' : 'wandb를 사용할 지 여부입니다. 사용할거면 본인 API 키를 모델안에 넣어주세요.'}
    )
    wandb_key : str = field(
        default = "ea26fff0d932bc74bbfad9fd507b292c67444c02",
        metadata = {'help' : 'wandb API 키 입니다.'}
    )
    model_name : str = field(
        default = "paust/pko-t5-small",
        metadata = {'help' : '사용할 모델 이름입니다.'}
    )
    max_source_length: int = field(
        default = 384,
        metadata={'help': "소스 문장의 최대 길이입니다."}
    )
    max_target_length: int = field(
        default = 32,
        metadata={'help': "타겟 문장의 최대 길이입니다."}
    )
    padding: str = field(
        default = "max_length",
        metadata = {'help': "패딩 방식을 설정합니다 ('max_length' 또는 'longest')."}
    )
    preprocessing_num_workers: int = field(
        default = None,
        metadata = {'help': "데이터 전처리 시 사용되는 워커(worker) 수입니다."}
    )
    num_beams: int = field(
        default = 3,
        metadata = {'help': "빔 서치 시 사용할 빔의 개수입니다."}
    )
    per_device_train_batch_size : int = field(
        default = 8,
        metadata = {'help': "Generative_based_MRC 모델의 배치사이즈입니다."}
    )
    eval_batch_size: int = field(
        default = 8,
        metadata = {'help': "평가 시 사용할 배치 크기입니다."}
    )
    learning_rate: float = field(
        default = 1e-3,
        metadata = {'help': "학습률(learning rate)입니다."}
    )
    num_train_epochs : int = field(
        default = 5,
        metadata = {'help' : '에폭을 정합니다.'}
    )




    # ----------이 아래는 util_qa.py, trainer_qa.py를 쓰기 위한 arguments들입니다.------------------



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

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
