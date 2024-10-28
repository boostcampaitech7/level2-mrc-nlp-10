from arguments import TF_IDF_retrieval_arguments
from transformers import (AutoTokenizer,
                          )
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import os
import pickle
import faiss
import numpy as np
from datasets import (load_from_disk,
                      Features,
                      Value,
                      DatasetDict,
                      Dataset)
from tqdm import tqdm
import pandas as pd
from All_dataset import prepare_dataset

class TF_IDFRetrieval:
    def __init__(self):
        self.args = TF_IDF_retrieval_arguments()
        tokenize_fn = AutoTokenizer.from_pretrained(self.args.model_name).tokenize
        # 데이터셋 로드
        with open(self.args.wiki_route, 'r', encoding = 'utf-8') as f:
            wiki = json.load(f)
        
        self.contexts = list(
            dict.fromkeys([v['text'] for v in wiki.values()])
        )
        print(len(self.contexts))

        self.tfidfv = TfidfVectorizer(
            tokenizer = tokenize_fn, ngram_range = (1, 2), max_features = 50000)

        self.p_embedding = None
        self.indexer = None
        self.datas = prepare_dataset(self.args)

    def get_sparse_embedding(self):

        pickle_name = f"Sparse_embedding.bin"
        tfidfv_name = f"TFIDF_vec.bin"
        if not os.path.exists(self.args.data_path):
            os.makedirs(self.args.data_path)
            print(f'{self.args.data_path} 폴더가 없어서 만듭니다.')
        emd_path = os.path.join(self.args.data_path, pickle_name)
        tfidfv_path = os.path.join(self.args.data_path, tfidfv_name)

        if os.path.isfile(emd_path) and os.path.isfile(tfidfv_path):
            with open(emd_path, 'rb') as file:
                self.p_embedding = pickle.load(file)
            with open(tfidfv_path, 'rb') as file:
                self.tfidfv = pickle.load(file)
            print('tfidf를 로드했습니다.')
        
        else:
            print('Passage embedding을 만듭니다.')
            self.p_embedding = self.tfidfv.fit_transform(self.contexts)
            print('p_embedding.shape :', self.p_embedding.shape)
            with open(emd_path, 'wb') as file:
                pickle.dump(self.p_embedding, file)
            with open(tfidfv_path, 'wb') as file:
                pickle.dump(self.tfidfv, file)
            print('임베딩을 피클형태로 저장했습니다.')

    def build_faiss(self):
        num_clusters = self.args.num_clusters
        indexer_name = f"TFIDF_faiss_clusters{num_clusters}.index"
        indexer_path = os.path.join(self.args.data_path, indexer_name)
        
        if os.path.isfile(indexer_path):
            print("Faiss Indexer을 로드했습니다.")
            self.indexer = faiss.read_index(indexer_path)
        else:
            print('Faiss indexer을 만듭니다.')
            p_emb = self.p_embedding.astype(np.float32).toarray()
            emb_dim = p_emb.shape[-1]

            # FAISS IndexIVFScalarQuantizer 초기화
            quantizer = faiss.IndexFlatL2(emb_dim)  # 벡터 차원을 전달
            self.indexer = faiss.IndexIVFScalarQuantizer(
                quantizer, emb_dim, num_clusters, faiss.METRIC_L2
            )

            # 훈련 및 추가
            self.indexer.train(p_emb)
            self.indexer.add(p_emb)
            faiss.write_index(self.indexer, indexer_path)
            print('Faiss Indexer을 저장했습니다.')

    def get_relevant_doc_bulk_faiss(self, queries):
        k = self.args.k
        query_vecs = self.tfidfv.transform(queries)
        
        # 쿼리 벡터의 합을 확인하여 검색 불가능한 경우 처리
        if np.sum(query_vecs) == 0:
            raise ValueError('쿼리에 있는 단어가 임베딩과 일치하지 않습니다. 쿼리를 확인하세요.')

        q_embs = query_vecs.toarray().astype(np.float32)
        print(f'쿼리당 {k}개의 문서를 faiss indexer을 통해 찾습니다.')
        
        # 검색 수행
        D, I = self.indexer.search(q_embs, k)

        return D.tolist(), I.tolist()

    
    def get_relevant_doc_bulk(self, query):
        query_vec = self.tfidfv.transform(query)
        k = self.args.k
        result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()
        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
        return doc_scores, doc_indices
    
    def search_query_faiss(self, mode = 'test'):
        if mode == 'test':
            test_dataset = load_from_disk(self.args.test_data_route)
        elif mode == 'eval':
            test_dataset = load_from_disk(self.args.data_route)
        queries = test_dataset['validation']['question']
        total = []
        passages = self.datas.get_context()
        doc_scores, doc_indices = self.get_relevant_doc_bulk_faiss(queries)
        total = []
        for idx, example in enumerate(tqdm(test_dataset['validation'], desc = "Dense retrieval: ")):
            tmp = {
                "question": example["question"],
                "id": example["id"],
                "context": " ".join([passages[pid] for pid in doc_indices[idx]]),
            }

            if "context" in example.keys() and "answers" in example.keys():
                tmp["original_context"] = example["context"]
                
                ground_truth_passage = example["context"]
                retrieved_passages = [passages[pid] for pid in doc_indices[idx]]
                
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


    
    def search_query(self, mode = 'test'):
        if mode == 'test':
            test_dataset = load_from_disk(self.args.test_data_route)
        elif mode == 'eval':
            test_dataset = load_from_disk(self.args.data_route)

        
        queries = test_dataset['validation']['question']
        passages = self.datas.get_context()
        doc_scores, doc_indices = self.get_relevant_doc_bulk(queries)
        total = []
        for idx, example in enumerate(tqdm(test_dataset['validation'], desc = "Dense retrieval: ")):
            tmp = {
                "question": example["question"],
                "id": example["id"],
                "context": " ".join([passages[pid] for pid in doc_indices[idx]]),
            }

            if "context" in example.keys() and "answers" in example.keys():
                tmp["original_context"] = example["context"]
                
                ground_truth_passage = example["context"]
                retrieved_passages = [passages[pid] for pid in doc_indices[idx]]
                
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