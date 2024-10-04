from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_from_disk
import os
import transformers
import torch
from transformers import PreTrainedModel
from All_dataset import prepare_dataset
from transformers import DefaultDataCollator
from transformers import TrainingArguments, Trainer
import torch
import torch.nn.functional as F
import gc
import random
from tqdm import tqdm

torch.manual_seed(2023)
torch.cuda.manual_seed(2023)
np.random.seed(2023)
random.seed(2023)


class TF_IDFSearch:
    def __init__(self, args):
        self.args = args
        
        # 데이터셋 로드
        self.dataset = load_from_disk(args.data_route)
        self.corpus = list(set(self.dataset['train']['context'] + self.dataset['validation']['context']))
        
        # TF-IDF 벡터 생성
        self.tfidf_vectorizer = TfidfVectorizer()
        self.matrix = self.tfidf_vectorizer.fit_transform(self.corpus).toarray()

        # FAISS 인덱스 생성
        self.d = self.matrix.shape[1]  # 벡터의 차원 수
        self.index = faiss.IndexFlatL2(self.d)  # L2 거리 기준의 인덱스 생성
        self.index.add(np.array(self.matrix, dtype=np.float32))  # TF-IDF 벡터 추가

    def build_faiss(self):
        # FAISS 인덱서 이름과 경로 정의
        num_clusters = self.args.num_cluster
        indexer_name = f"faiss_TF-IDF_clusters{num_clusters}.index"
        indexer_path = os.path.join(os.getcwd(), indexer_name)
        
        if os.path.isfile(indexer_path):
            print("Load Saved Faiss Indexer.")
            self.indexer = faiss.read_index(indexer_path)

        else:
            print("Building Faiss Indexer.")
            # 데이터 벡터 생성
            p_emb = self.matrix.astype(np.float32)
            emb_dim = p_emb.shape[-1]

            # Faiss 인덱서 생성
            quantizer = faiss.IndexFlatL2(emb_dim)
            self.indexer = faiss.IndexIVFScalarQuantizer(
                quantizer, emb_dim, num_clusters, faiss.METRIC_L2
            )
            
            # 인덱서 학습 및 추가
            self.indexer.train(p_emb)
            self.indexer.add(p_emb)

            # 인덱서 저장
            faiss.write_index(self.indexer, indexer_path)
            print("Faiss Indexer Saved.")

    def search_query(self):
        self.test_dataset = load_from_disk(self.args.test_data_route)
        queries = self.test_dataset['validation']['question']
        k = self.args.k
        # 쿼리 벡터 생성
        query_vector = self.tfidf_vectorizer.transform(queries).toarray().astype(np.float32)
        
        # 유사한 문서 검색
        distances, indices = self.index.search(query_vector, k)

        # 검색 결과 저장
        results = []
        for i in range(len(queries)):  # 모든 쿼리에 대해 반복
            query_results = []
            for j in range(k):
                doc = self.corpus[indices[i][j]]  # 해당 쿼리의 상위 K 문서
                distance = distances[i][j]  # 해당 문서와의 거리
                query_results.append((doc, distance))
            results.append(query_results)

        return results


# -------------------------------아래는 Dense Embedding을 사용하는 부분입니다.-----------------------------------------

import torch
import torch.nn.functional as F
import transformers

class Dense_embedding_retrieval_model(PreTrainedModel):
    def __init__(self, args):
        config = transformers.AutoConfig.from_pretrained(args.model_name)
        super(Dense_embedding_retrieval_model, self).__init__(config)
        model_name = args.model_name
        self.args = args
        self.p_model = transformers.AutoModel.from_pretrained(model_name)
        self.q_model = transformers.AutoModel.from_pretrained(model_name)

    def forward(self, p_input_ids, p_attention_mask, p_token_type_ids,
                q_input_ids, q_attention_mask, q_token_type_ids, labels):
        
        p_input = {
            'input_ids': p_input_ids.reshape(self.args.per_device_train_batch_size * (self.args.num_neg+1),-1),
            'attention_mask': p_attention_mask.reshape(self.args.per_device_train_batch_size * (self.args.num_neg+1),-1),
            'token_type_ids': p_token_type_ids.reshape(self.args.per_device_train_batch_size * (self.args.num_neg+1),-1)
        }
        q_input = {
            'input_ids': q_input_ids,
            'attention_mask': q_attention_mask,
            'token_type_ids': q_token_type_ids
        }

        p_outputs = self.p_model(**p_input)
        q_outputs = self.q_model(**q_input)



        p_outputs = p_outputs.last_hidden_state[:, 0, :]  # (batch_size, emb_dim)
        q_outputs = q_outputs.last_hidden_state[:, 0, :]  # (batch_size, emb_dim)

        p_outputs = p_outputs.view(self.args.per_device_train_batch_size, self.args.num_neg+1, -1)  # (batch_size, num_neg+1, emb_dim)
        q_outputs = q_outputs.view(self.args.per_device_train_batch_size, 1, -1)  # (batch_size, 1, emb_dim)

        sim_scores = torch.bmm(q_outputs, p_outputs.transpose(1, 2)).squeeze(1)  # (batch_size, num_neg+1)
        sim_scores = F.log_softmax(sim_scores, dim=1)
        labels = torch.zeros(self.args.per_device_train_batch_size, dtype=torch.long, device=p_input_ids.device)

        loss = F.nll_loss(sim_scores, labels)

        return {
            'loss': loss,
            'output' : sim_scores
        }

def compute_metrics(eval_preds):

    logits, labels = eval_preds  

    # logits에서 가장 높은 값을 가진 인덱스를 예측값으로 사용
    logits = torch.tensor(logits, dtype = torch.long)
    predictions = torch.argmax(logits, dim=1).detach().cpu().numpy()
    labels = torch.zeros(len(predictions))
    labels = labels.detach().cpu().numpy()
    
    accuracy = accuracy_score(labels, predictions)  
    f1 = f1_score(labels, predictions, average='weighted') 

    return {'accuracy' : accuracy, 'f1' : f1 }

class Dense_embedding_retrieval:
    def __init__(self, args):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.args = args
        self.datas = prepare_dataset(args)
        self.train_dataset = self.datas.get_dense_train_dataset()
        self.valid_dataset = self.datas.get_dense_valid_dataset()
        self.model = Dense_embedding_retrieval_model(args)

    def train(self):
        torch.cuda.set_per_process_memory_fraction(0.8)  # GPU 메모리 사용 비율 조정
        torch.backends.cuda.matmul.allow_tf32 = True  # TF32 지원 활성화
        gc.collect()
        torch.cuda.empty_cache()
        data_collator = DefaultDataCollator()
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs = self.args.num_train_epochs,
            per_device_train_batch_size = self.args.per_device_train_batch_size,
            per_device_eval_batch_size=self.args.per_device_train_batch_size,
            learning_rate = self.args.learning_rate,
            save_strategy = 'epoch',
            logging_steps = 30,
            evaluation_strategy = "epoch",  
            logging_dir='./logs',
            load_best_model_at_end=True,
            do_eval = True,
            fp16 = True
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.valid_dataset,
            compute_metrics = compute_metrics,
            data_collator = data_collator
            
        )
        trainer.train()
        torch.cuda.empty_cache()


    def build_faiss(self):
        num_clusters = self.args.num_cluster
        indexer_name = f"faiss_dense_clusters{num_clusters}.index"
        indexer_path = os.path.join(os.getcwd(), indexer_name)
        
        if os.path.isfile(indexer_path):
            print("Load Saved Faiss Indexer.")
            self.indexer = faiss.read_index(indexer_path)

        else:
            print("Building Faiss Indexer.")
            self.model.eval()
            self.passages = self.datas.get_dense_faiss_corpus()
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
                # 인덱서 저장
                faiss.write_index(self.indexer, indexer_path)
                print("Faiss Indexer Saved.")

    def search(self):
        top_k = self.args.top_k
        self.passages = self.datas.get_dense_faiss_corpus()
        self.model.eval()
        with torch.no_grad():
            inputs = self.datas.get_dense_queries_for_search()  
            inputs = {key: value.to(self.device) for key, value in inputs.items()} 
            query_embeddings = self.model.q_model(**inputs).pooler_output.cpu().numpy() 

        distances, indices = self.indexer.search(query_embeddings, top_k)  

        results = []
        print('모든쿼리에 대해 탑 k를 찾습니당')
        for i in tqdm(range(query_embeddings.shape[0])):  # 모든 쿼리에 대해
            query_results = []
            for j in range(top_k):
                doc = self.passages[indices[i][j]]  # 상위 K 문서
                distance = distances[i][j]  # 해당 문서와의 거리
                query_results.append((doc, distance))
            results.append(query_results)

        return results 
