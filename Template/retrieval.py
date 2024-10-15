from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import faiss
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import os
import transformers
import pickle
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from All_dataset import prepare_dataset
from transformers import DefaultDataCollator
from transformers import TrainingArguments, Trainer
import torch
import torch.nn.functional as F
import gc
import random
from tqdm import tqdm
from datasets import Features, Value, DatasetDict, load_from_disk, Dataset
import wandb
from rank_bm25 import BM25Okapi

from arguments import Dense_search_retrieval_arguments, TF_IDF_retrieval_arguments, BM25_retrieval_arguments
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)
random.seed(2024)


class TF_IDFSearch:
    def __init__(self):
        self.args = TF_IDF_retrieval_arguments()
        tokenize_fn = transformers.AutoTokenizer.from_pretrained(self.args.model_name).tokenize
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
    
    def search_query_faiss(self):
        test_dataset = load_from_disk(self.args.test_data_route)
        queries = test_dataset['validation']['question']
        total = []

        doc_scores, doc_indices = self.get_relevant_doc_bulk_faiss(queries)
        for idx, example in enumerate(
            tqdm(test_dataset['validation'], desc = 'Sparse retrieval: ')
        ):
            tmp = {
                'question' : example['question'],
                'id' : example['id'],
                'context' : ' '.join(
                    [self.contexts[pid] for pid in doc_indices[idx]]
                ),
            }
            if 'context' in example.keys() and 'answers' in example.keys():
                tmp['answers'] = example['answers']
            total.append(tmp)

        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
        df = pd.DataFrame(total)
        datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
        return datasets

    
    def search_query(self):
        test_dataset = load_from_disk(self.args.test_data_route)
        queries = test_dataset['validation']['question']
        total = []
        doc_scores, doc_indices = self.get_relevant_doc_bulk(queries)
        for idx, example in enumerate(
            tqdm(test_dataset['validation'], desc = 'Sparse retrieval: ')
        ):
            tmp = {
                'question' : example['question'],
                'id' : example['id'],
                'context' : ' '.join(
                    [self.contexts[pid] for pid in doc_indices[idx]]
                ),
            }
            if 'context' in example.keys() and 'answers' in example.keys():
                tmp['answers'] = example['answers']
            total.append(tmp)

        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
        df = pd.DataFrame(total)
        datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
        return datasets

# -------------------------------아래는 Dense Embedding을 사용하는 부분입니다.-----------------------------------------

import torch
import torch.nn.functional as F
import transformers

class Dense_embedding_retrieval_model(PreTrainedModel):
    def __init__(self):
        self.args = Dense_search_retrieval_arguments
        config = transformers.AutoConfig.from_pretrained(self.args.model_name)
        super(Dense_embedding_retrieval_model, self).__init__(config)
        model_name = self.args.model_name
        self.p_model = transformers.AutoModel.from_pretrained(model_name)
        # self.q_model = transformers.AutoModel.from_pretrained(model_name)
        self.q_model = self.p_model
        # self.p_linear = nn.Linear(768, 256)  # emb_dim은 p_model의 출력 크기
        # self.p_linear2 = nn.Linear(256, 768)
        # self.q_linear = nn.Linear(768, 256)
        # self.q_linear2 = nn.Linear(256, 768)
        # self.dropout = nn.Dropout(0.1)

    def forward(self, p_input_ids, p_attention_mask, p_token_type_ids,
                q_input_ids, q_attention_mask, q_token_type_ids, labels):
        
        p_input = {
            'input_ids': p_input_ids.view(self.args.per_device_train_batch_size * (self.args.num_neg+1),-1),
            'attention_mask': p_attention_mask.view(self.args.per_device_train_batch_size * (self.args.num_neg+1),-1),
            'token_type_ids': p_token_type_ids.view(self.args.per_device_train_batch_size * (self.args.num_neg+1),-1)
        }
        q_input = {
            'input_ids': q_input_ids,
            'attention_mask': q_attention_mask,
            'token_type_ids': q_token_type_ids
        }

        p_outputs = self.p_model(**p_input)
        q_outputs = self.q_model(**q_input)

        p_outputs = p_outputs.pooler_output  # (batch_size * (num_neg+1), emb_dim)
        q_outputs = q_outputs.pooler_output  # (batch_size, emb_dim)

        # p_outputs = self.p_linear(p_outputs)
        # p_outputs = self.dropout(p_outputs)
        # p_outputs = self.p_linear2(p_outputs)
        # p_outputs = self.dropout(p_outputs)

        # q_outputs = self.q_linear(q_outputs)
        # q_outputs = self.dropout(q_outputs)
        # q_outputs = self.q_linear2(q_outputs)
        # q_outputs = self.dropout(q_outputs)

        # Reshape p_outputs, q_outputs
        p_outputs = p_outputs.view(self.args.per_device_train_batch_size, self.args.num_neg + 1, -1)  # (batch_size, num_neg+1, emb_dim)
        q_outputs = q_outputs.view(self.args.per_device_train_batch_size, 1, -1)  # (batch_size, 1, emb_dim)

        # 유사도 계산 (cosine similarity or dot product)
        sim_scores = torch.matmul(q_outputs, p_outputs.transpose(1, 2)).squeeze(1)  # (batch_size, num_neg+1)

        # Negative Log Likelihood 적용
        log_probs = F.log_softmax(sim_scores, dim=-1)  # (batch_size, num_neg+1)
        loss = F.nll_loss(log_probs, labels)  # NLL Loss 계산

        return {
            'loss': loss,
            'output' : sim_scores
        }


class Dense_embedding_retrieval:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.args = Dense_search_retrieval_arguments()
        self.model = Dense_embedding_retrieval_model()
        self.model.to(self.device)
        if self.args.use_wandb:
            self.start_wandb()
        self.datas = prepare_dataset(self.args)
        self.contexts = self.datas.get_context()

    def compute_metrics(self,eval_preds):

        logits, labels = eval_preds  

        # logits에서 가장 높은 값을 가진 인덱스를 예측값으로 사용
        logits = torch.tensor(logits, dtype = torch.long)
        predictions = torch.argmax(logits, dim=1).detach().cpu().numpy()
        
        accuracy = accuracy_score(labels, predictions)  
        f1 = f1_score(labels, predictions, average='weighted') 

        return {'accuracy' : accuracy, 'f1' : f1 }

    def train(self):
        self.train_dataset = self.datas.get_dense_train_dataset()
        self.valid_dataset = self.datas.get_dense_valid_dataset()
        gc.collect()
        torch.cuda.empty_cache()
        data_collator = DefaultDataCollator()
        training_args = TrainingArguments(
            output_dir = './Dense_embedding_retrieval_model_results',
            num_train_epochs = self.args.num_train_epochs,
            per_device_train_batch_size = self.args.per_device_train_batch_size,
            per_device_eval_batch_size = self.args.per_device_train_batch_size,
            learning_rate = self.args.learning_rate,
            save_strategy = 'epoch',
            logging_steps = 30,
            fp16 = True, # FP16 (16-bit floating point)으로 수행하면 메모리 사용량이 줄어듭니다.
            evaluation_strategy = "epoch",  
            logging_dir = './logs',
            load_best_model_at_end = True,
            do_eval = True,
            weight_decay = 0.01,
        )
        if self.args.use_wandb:
            training_args.report_to = ["wandb"]
            training_args.run_name = "default"


        trainer = Trainer(
            model = self.model,
            args = training_args,
            train_dataset = self.train_dataset,
            eval_dataset = self.valid_dataset,
            compute_metrics = self.compute_metrics,
            data_collator = data_collator
            
        )
        trainer.train()
        torch.cuda.empty_cache()


    def build_faiss(self):
        num_clusters = self.args.num_clusters
        indexer_name = f"Dense_faiss_clusters{num_clusters}.index"
        indexer_path = os.path.join(self.args.data_path, indexer_name)
        print(indexer_path)
        if os.path.isfile(indexer_path):
            print("Faiss Indexer을 로드했습니다.")
            self.indexer = faiss.read_index(indexer_path)

        else:
            print("Building Faiss Indexer.")
            self.model.eval()
            self.passages = self.contexts
            tokenizer = transformers.AutoTokenizer.from_pretrained(self.args.model_name)

            with torch.no_grad():
                embeddings = []
                for passage in tqdm(self.passages):
                    inputs = tokenizer(passage, return_tensors='pt', padding=True, truncation=True).to(self.device)
                    embedding = self.model.p_model(**inputs).pooler_output.squeeze(0)  # passage 모델에서 출력
                    embeddings.append(embedding.cpu().numpy())

                embeddings = np.vstack(embeddings)

            emb_dim = embeddings.shape[1]
            quantizer = faiss.IndexFlatL2(emb_dim)

            # 클러스터 수가 0보다 큰지 확인
            if num_clusters > 0:
                self.indexer = faiss.IndexIVFScalarQuantizer(quantizer, emb_dim, num_clusters, faiss.METRIC_L2)
                self.indexer.train(embeddings)
                self.indexer.add(embeddings)
                faiss.write_index(self.indexer, indexer_path)
                print("Faiss Indexer Saved.")
            else:
                self.indexer = quantizer  # 클러스터 수가 0일 경우 flat index 사용
                self.indexer.add(embeddings)
                faiss.write_index(self.indexer, indexer_path)
                print("Faiss Indexer Saved.")

    def search_faiss(self):
        top_k = self.args.top_k
        self.passages = self.contexts
        self.model.eval()

        if not os.path.exists('retrieval_result'):
            os.makedirs('retrieval_result')

        with torch.no_grad():
            test_dataset, inputs = self.datas.get_dense_queries_for_search()  
            inputs = {key: value.to(self.device) for key, value in inputs.items()} 
            query_embeddings = self.model.q_model(**inputs).pooler_output.cpu().numpy() 

        print(f'faiss indxer을 통해 쿼리당{top_k}개의 문서를 찾습니다.')
        doc_scores, doc_indices = self.indexer.search(query_embeddings, top_k)  

        total = []
        for idx, example in enumerate(
            tqdm(test_dataset['validation'], desc = 'Sparse retrieval: ')
        ):
            tmp = {
                'question' : example['question'],
                'id' : example['id'],
                'context' : ' '.join(
                    [self.contexts[pid] for pid in doc_indices[idx]]
                ),
            }
            if 'context' in example.keys() and 'answers' in example.keys():
                tmp['answers'] = example['answers']
            total.append(tmp)

        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
        df = pd.DataFrame(total)
        datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
        return datasets

        

    def start_wandb(self):
        os.system("rm -rf /root/.cache/wandb")
        os.system("rm -rf /root/.config/wandb")
        os.system("rm -rf /root/.netrc")
        
        # WandB API 키 설정 (여러분의 API 키를 넣어주시면 됩니다)
        os.environ["WANDB_API_KEY"] = self.args.wandb_key
        wandb.init(project='Dense_embedding_retrieval')



# -------------------------------아래는 BM25Search 을 사용하는 부분입니다.-----------------------------------------


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

        # BM25를 위한 토크나이저 적용 (BERT 기반 토크나이저 사용)
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

    def get_relevant_doc_bulk_bm25(self, queries):
        k = self.args.k
        total_scores = []
        total_indices = []

        for query in queries:
            # BERT 토크나이저를 사용하여 쿼리 토크나이즈
            tokenized_query = self.tokenizer.tokenize(query)
            doc_scores = self.bm25.get_scores(tokenized_query)
            sorted_scores_idx = np.argsort(doc_scores)[::-1]
            total_scores.append(doc_scores[sorted_scores_idx][:k].tolist())
            total_indices.append(sorted_scores_idx[:k].tolist())

        return total_scores, total_indices

    def search_query_bm25(self):
        test_dataset = load_from_disk(self.args.test_data_route)
        queries = test_dataset['validation']['question']
        total = []

        # get_relevant_doc_bulk_bm25 메서드를 사용하여 문서 점수와 인덱스 가져오기
        doc_scores, doc_indices = self.get_relevant_doc_bulk_bm25(queries)

        for idx, example in enumerate(
            tqdm(test_dataset['validation'], desc='BM25 retrieval: ')
        ):
            tmp = {
                'question': example['question'],
                'id': example['id'],
                'context': ' '.join([self.contexts[pid] for pid in doc_indices[idx]]),
                # 'doc_scores': doc_scores[idx],  # 점수 추가
            }
            if 'context' in example.keys() and 'answers' in example.keys():
                tmp['answers'] = example['answers']
            total.append(tmp)

        # 데이터셋 생성
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
                # "doc_scores": Value(dtype="float32", id=None),  # 점수 필드 추가
            }
        )
        df = pd.DataFrame(total)
        datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
        return datasets


