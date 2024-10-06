from dataclasses import dataclass, field
from typing import Optional

@dataclass
class MRC_train_dataset:
# argparse 설정
    data_route: Optional[str] = field(
        default = '/data/ephemeral/home/level2-mrc-nlp-10/data/train_dataset/',
        metadata = {'help' :"데이터셋 위치입니다."},
    )
    model_name: str = field(
        default = 'klue/bert-base',
        metadata = {'help' : '모델이름입니다. 기본은 버트입니다.'}
    )
    max_seq_length: int = field(
        default = 384,
        metadata = {'help' : 'context길이를 얼마나 가져갈건지를 정하는 수'}
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

@dataclass
class TF_IDF_retrieval_arguments:
    data_route: Optional[str] = field(
        default = '/data/ephemeral/home/level2-mrc-nlp-10/data/train_dataset/',
        metadata = {'help' : "데이터셋 위치입니다."},
    )
    test_data_route : Optional[str] = field(
        default = '/data/ephemeral/home/level2-mrc-nlp-10/data/test_dataset/',
        metadata = {'help' : '테스트 데이터셋 위치입니다.'}
    )
    k : int = field(
        default = 3,
        metadata = {'help' : '비슷한 문서 중 몇개를 내보낼지를 결정합니다.'}
    )
    num_cluster : int = field(
        default = 64,
        metadata = {'help' : 'faiss 클러스터를 몇개로 할 지 정합니다.'}
    )


@dataclass
class Dense_search_retrieval_arguments:
    data_route: Optional[str] = field(
        default = '/data/ephemeral/home/level2-mrc-nlp-10/data/train_dataset/',
        metadata = {'help' : "데이터셋 위치입니다."},
    )
    test_data_route : Optional[str] = field(
        default = '/data/ephemeral/home/level2-mrc-nlp-10/data/test_dataset/',
        metadata = {'help' : '테스트 데이터셋 위치입니다.'}
    )
    model_name: Optional[str] = field(
        default = "klue/roberta-small",
        metadata = {'help': "모델이름 입니다."}
    )
    num_neg: int = field(
        default = 3,
        metadata = {'help': "네거티브 샘플링 몇개할건지를 고릅니다."}
    )
    num_train_epochs: int = field(
        default = 5,
        metadata = {'help': "에폭입니다."}
    )
    per_device_train_batch_size: int = field(
        default = 8,
        metadata = {'help': "Dense embedding 모델의 배치사이즈입니다."}
    )
    learning_rate: float = field(
        default = 5e-5,
        metadata = {'help': "러닝레이트입니다."}
    )
    weight_decay: float = field(
        default = 0.01,
        metadata = {'help': "학습동안 적용할 weightdecay입니다."}
    )
    num_cluster : int = field(
        default = 64,
        metadata = {'help' : 'faiss 클러스터를 몇개로 할 지 정합니다.'}
    )
    top_k : int = field(
        default = 3,
        metadata = {'help' : '몇개의 후보를 faiss indexer에서 뽑을건지 정합니다.'}
    )
    use_wandb : bool = field(
        default = True,
        metadata = {'help' : 'wandb를 '}
    )
    
class Extraction_based_MRC_arguments:
    max_seq_length : int = field(
        default = 384,
        metadata = {'help' : '질문과 컨텍스트, special 토큰을 합한 문자열의 최대 길이입니다.'}
    )
    pad_to_max_length : bool = field(
        default = True,
        metadata = {'help' : '최대 길이의 문장을 기준으로 패딩을 수행합니다.'}
    )
    doc_stride : int = field(
        default = 128,
        metadata = {'help' : '긴 컨텍스트를 나눌 때 앞의 컨텍스트와 얼마나 겹치게 시퀀스를 구성할 것인지 정합니다.'}
    )
    # max_train_samples : int = field(
    #     default = 16,
    #     metadata = {'help' : 'train 배치사이즈를 정합니다.'}
    # )
    # max_val_samples : int = field(
    #     default = 16,
    #     metadata = {'help' : 'valid 배치사이즈르 정합니다.'}
    # )
    preprocessing_num_workers : int = field(
        default = 4,
        metadata = {'help' : '프로세스 몇개 쓸건지입니다. 높으면 빠르긴한데 메모리를 많이씁니다.'}
    )
    batch_size : int = field(
        default = 8,
        metadata = {'help' : '배치사이즈를 정합니다.'}
    )
    num_train_epochs : int = field(
        default = 5,
        metadata = {'help' : '에폭을 정합니다.'}
    )
    n_best_size : int = field(
        default = 20,
        metadata = {'help' : ''}
    )
    max_answer_length : int = field(
        default = 30,
        metadata = {'help' : ''}
    )