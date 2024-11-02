import torch
import torch.nn.functional as F
import os
import json
import pickle
from tqdm import tqdm
from datasets import Features, Value, DatasetDict, load_from_disk, Dataset
import wandb
from rank_bm25 import BM25Okapi
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import faiss
import transformers
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from arguments import BM25_retrieval_arguments

class BM25Search:
    def __init__(self):
        self.args = BM25_retrieval_arguments()
        print("Data route:", self.args.data_route)
        print("Test data route:", self.args.test_data_route)
        print("Wiki route:", self.args.wiki_route)
        print("Data path:", self.args.data_path)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.args.bm25_tokenizer)

        # 데이터셋 로드
        try:
            with open(self.args.wiki_route, 'r', encoding='utf-8') as f:
                wiki = json.load(f)
            print("File loaded successfully")
        except Exception as e:
            print("Error opening file:", e)  # 파일 열기 오류 로그


        self.contexts = list(dict.fromkeys([v['text'] for v in wiki.values()]))
        print(f"위키 컨텍스트의 수: {len(self.contexts)}")

        # BM25를 위한 토크나이저 적용 (BM25 기반 토크나이저 사용)
        self.tokenized_contexts = [self.tokenizer.tokenize(context) for context in self.contexts]
        self.bm25 = BM25Okapi(self.tokenized_contexts)

        # BM25 임베딩
        self.p_embedding = None

    def get_sparse_embedding(self):
        pickle_name = f"BM25_embedding.bin"
        if not os.path.exists(self.args.data_path):
            os.makedirs(self.args.data_path)
            print(f'{self.args.data_path} 폴더가 없어서 만듭니다.')
        emd_path = os.path.join(self.args.data_path, pickle_name)

        if os.path.isfile(emd_path):
            with open(emd_path, 'rb') as file:
                self.p_embedding = pickle.load(file)
            print('BM25 임베딩을 로드했습니다.')
        else:
            print('Passage embedding을 만듭니다.')
            self.p_embedding = self.tokenized_contexts  # 이미 토크나이즈된 컨텍스트 사용
            with open(emd_path, 'wb') as file:
                pickle.dump(self.p_embedding, file)
            print('임베딩을 피클 형태로 저장했습니다.')

    def get_relevant_doc_bulk_bm25(self, queries, k = None, mode = None):
        if k == None:
            k = self.args.k
        total_scores = []
        total_indices = []

        for query in queries:
            # BERT 토크나이저를 사용하여 쿼리 토크나이즈
            tokenized_query = self.tokenizer.tokenize(query)
            doc_scores = self.bm25.get_scores(tokenized_query)
            if mode == 'ensemble':
                total_scores.append(doc_scores[:k])
                continue
            sorted_scores_idx = np.argsort(doc_scores)[::-1]
            total_scores.append(doc_scores[sorted_scores_idx][:k].tolist())
            total_indices.append(sorted_scores_idx[:k].tolist())
        if mode == 'ensemble':
            return total_scores
        else:
            return total_scores, total_indices

    def search_query(self, mode = 'test'):
        if mode == 'test':
            test_dataset = load_from_disk(self.args.test_data_route)
        elif mode == 'eval':
            test_dataset = load_from_disk(self.args.data_route)
        queries = test_dataset['validation']['question']
        total = []

        # get_relevant_doc_bulk_bm25 메서드를 사용하여 문서 점수와 인덱스 가져오기
        doc_scores, doc_indices = self.get_relevant_doc_bulk_bm25(queries)

        total = []
        for idx, example in enumerate(tqdm(test_dataset['validation'], desc = "bm25 retrieval: ")):
            tmp = {
                "question": example["question"],
                "id": example["id"],
                "context": " ".join([self.contexts[pid] for pid in doc_indices[idx]]),
            }

            if "context" in example.keys() and "answers" in example.keys():
                tmp["original_context"] = example["context"]
                
                ground_truth_passage = example["context"]
                retrieved_passages = [self.contexts[pid] for pid in doc_indices[idx]]
                
                # 정답이 retrieved passages에 포함되는지 여부를 확인
                tmp["answers"] = ground_truth_passage in retrieved_passages  # True or False
            total.append(tmp)

        df = pd.DataFrame(total)
        self.df = df

        if mode == 'test':
            f = Features(
                {
                    "context": Value(dtype="string", id=None),
                    "id": Value(dtype="string", id=None),
                    "question": Value(dtype="string", id=None),
                }
            )
            datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
        if mode == 'eval':
            f = Features(
                {
                    "context": Value(dtype="string", id=None),
                    "id": Value(dtype="string", id=None),
                    "question": Value(dtype="string", id=None),
                    "original_context": Value(dtype="string", id=None),
                    "answers": Value(dtype="bool", id=None)  # answers를 bool로 설정
                }
            )

            datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
            cnt = 0
            for i in datasets['validation']['answers']:
                if i == True:
                    cnt += 1
            print('acc :',cnt / len(datasets['validation']['answers']))

        return datasets


