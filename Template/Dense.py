# 코드 실습시 같이 올린 Dense_test.ipynb 부터 실행하여 필요 파일 저장 후 retreival_test.ipynb 에서 실행하여 MRC 에 연결하시면 됩니다.

import os
import json
from datasets import load_from_disk, Dataset, DatasetDict, Features, Value
from transformers import AutoTokenizer, AutoModel, AutoConfig
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset as TorchDataset, DataLoaderㄴ
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Tuple
import random
import pickle

@dataclass
class DenseRetrievalArguments:
    data_route: Optional[str] = field(
        default='/data/ephemeral/template_2/data/train_dataset',
        metadata={'help': "데이터셋 위치입니다."},
    )
    test_data_route: Optional[str] = field(
        default='/data/ephemeral/data/test_dataset',
        metadata={'help': '테스트 데이터셋 위치입니다.'}
    )
    k: int = field(
        default=30,
        metadata={'help': '비슷한 문서 중 몇 개를 내보낼지를 결정합니다.'}
    )
    num_negatives: int = field(
        default=1,
        metadata={'help': '각 질문당 부정 샘플의 수'}
    )
    wiki_route: str = field(
        default='/data/ephemeral/data/wikipedia_documents.json',
        metadata={'help': '위키데이터의 경로입니다.'}
    )
    model_name: str = field(
        default="klue/bert-base",
        metadata={'help': '모델 이름입니다.'}
    )
    data_path: str = field(
        default='/data/ephemeral/retrieval_result',
        metadata={'help': 'retrieval_result를 저장할 경로입니다.'}
    )
    loss_function: str = field(
        default='contrastive',  # 'contrastive' 또는 'bce'
        metadata={'help': '손실 함수 선택: "contrastive" 또는 "bce"'}
    )
    margin: float = field(
        default=0.5,
        metadata={'help': 'Contrastive Loss의 마진 값'}
    )

def load_datasets(sample_size: Optional[int], data_path: str) -> Tuple[Dataset, Dataset, Dataset]:
    """
    주어진 경로에서 학습, 검증, 테스트 데이터셋을 로드합니다.
    중복된 문서를 제거하여 데이터셋의 품질을 향상시킵니다.
    """
    dataset = load_from_disk(data_path)
    train_dataset = dataset['train']
    validation_dataset = dataset['validation']
    test_dataset = dataset['validation']

    if sample_size:
        train_dataset = train_dataset.shuffle(seed=42).select(range(sample_size))
        print(f"학습 데이터셋 샘플 크기: {len(train_dataset)}")

    # 중복된 문서 제거 (예: 'context' 필드를 기준으로)
    def remove_duplicates(dataset):
        unique_contexts = set()
        def is_unique(example):
            if example['context'] in unique_contexts:
                return False
            unique_contexts.add(example['context'])
            return True
        return dataset.filter(is_unique)

    try:
        train_dataset = remove_duplicates(train_dataset)
        validation_dataset = remove_duplicates(validation_dataset)
        test_dataset = remove_duplicates(test_dataset)
        print(f"중복 제거 후 학습 데이터셋 크기: {len(train_dataset)}")
        print(f"중복 제거 후 검증 데이터셋 크기: {len(validation_dataset)}")
        print(f"중복 제거 후 테스트 데이터셋 크기: {len(test_dataset)}")
    except KeyError as e:
        print(f"중복 제거 중 오류 발생: {e}")
        print("데이터셋에 'context' 컬럼이 존재하는지 확인해주세요.")
        raise
    
    return train_dataset, validation_dataset, test_dataset


class DenseRetrievalModel(nn.Module):
    def __init__(self, model_name="bert-base-multilingual-cased"):
        super(DenseRetrievalModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = AutoConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.p_model = AutoModel.from_pretrained(model_name)
        self.q_model = AutoModel.from_pretrained(model_name)
        self.cos = nn.CosineSimilarity(dim=-1)

    def encode_passage(self, p_input_ids, p_attention_mask, p_token_type_ids):
        p_outputs = self.p_model(input_ids=p_input_ids,
                                 attention_mask=p_attention_mask,
                                 token_type_ids=p_token_type_ids)
        p_embeddings = p_outputs.last_hidden_state[:, 0, :]  # [CLS] 토큰
        return p_embeddings

    def encode_queries(self, q_input_ids, q_attention_mask, q_token_type_ids):
        q_outputs = self.q_model(input_ids=q_input_ids,
                                 attention_mask=q_attention_mask,
                                 token_type_ids=q_token_type_ids)
        q_embeddings = q_outputs.last_hidden_state[:, 0, :]  # [CLS] 토큰
        return q_embeddings

    def forward(self, p_input_ids, p_attention_mask, p_token_type_ids,
                q_input_ids, q_attention_mask, q_token_type_ids):
        p_embeddings = self.encode_passage(p_input_ids, p_attention_mask, p_token_type_ids)
        q_embeddings = self.encode_queries(q_input_ids, q_attention_mask, q_token_type_ids)
        return p_embeddings, q_embeddings

class RetrievalDataset(TorchDataset):
    def __init__(self, dataset, tokenizer, all_contexts, max_length=512, num_negatives=1):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.all_contexts = all_contexts
        self.num_negatives = num_negatives

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ex = self.dataset[idx]
        question = ex['question']
        positive_context = ex['context']

        # 긍정 샘플 인코딩
        q_encode = self.tokenizer(question, padding="max_length", truncation=True, max_length=128, return_tensors='pt')
        p_encode = self.tokenizer(positive_context, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')

        item = {
            'q_input_ids': q_encode['input_ids'].squeeze(),
            'q_attention_mask': q_encode['attention_mask'].squeeze(),
            'q_token_type_ids': q_encode.get('token_type_ids', torch.zeros_like(q_encode['input_ids'])).squeeze(),
            'p_input_ids': p_encode['input_ids'].squeeze(),
            'p_attention_mask': p_encode['attention_mask'].squeeze(),
            'p_token_type_ids': p_encode.get('token_type_ids', torch.zeros_like(p_encode['input_ids'])).squeeze(),
            'label': 1  # 긍정 샘플
        }

        # 부정 샘플 생성
        negative_context = random.choice(self.all_contexts)
        while negative_context == positive_context:
            negative_context = random.choice(self.all_contexts)

        n_encode = self.tokenizer(negative_context, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')

        item_neg = {
            'q_input_ids': q_encode['input_ids'].squeeze(),
            'q_attention_mask': q_encode['attention_mask'].squeeze(),
            'q_token_type_ids': q_encode.get('token_type_ids', torch.zeros_like(q_encode['input_ids'])).squeeze(),
            'p_input_ids': n_encode['input_ids'].squeeze(),
            'p_attention_mask': n_encode['attention_mask'].squeeze(),
            'p_token_type_ids': n_encode.get('token_type_ids', torch.zeros_like(n_encode['input_ids'])).squeeze(),
            'label': 0  # 부정 샘플
        }

        return item, item_neg

    @staticmethod
    def collate_fn(batch):
        q_input_ids = []
        q_attention_mask = []
        q_token_type_ids = []
        p_input_ids = []
        p_attention_mask = []
        p_token_type_ids = []
        labels = []

        for item, item_neg in batch:
            # 긍정 샘플
            q_input_ids.append(item['q_input_ids'])
            q_attention_mask.append(item['q_attention_mask'])
            q_token_type_ids.append(item['q_token_type_ids'])
            p_input_ids.append(item['p_input_ids'])
            p_attention_mask.append(item['p_attention_mask'])
            p_token_type_ids.append(item['p_token_type_ids'])
            labels.append(item['label'])

            # 부정 샘플
            q_input_ids.append(item_neg['q_input_ids'])
            q_attention_mask.append(item_neg['q_attention_mask'])
            q_token_type_ids.append(item_neg['q_token_type_ids'])
            p_input_ids.append(item_neg['p_input_ids'])
            p_attention_mask.append(item_neg['p_attention_mask'])
            p_token_type_ids.append(item_neg['p_token_type_ids'])
            labels.append(item_neg['label'])

        return {
            'q_input_ids': torch.stack(q_input_ids),
            'q_attention_mask': torch.stack(q_attention_mask),
            'q_token_type_ids': torch.stack(q_token_type_ids),
            'p_input_ids': torch.stack(p_input_ids),
            'p_attention_mask': torch.stack(p_attention_mask),
            'p_token_type_ids': torch.stack(p_token_type_ids),
            'label': torch.tensor(labels, dtype=torch.float)
        }

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, similarity, label):
        # label: 1 긍정, 0 은 부정
        loss = (1 - label) * torch.pow(similarity, 2) + \
               label * torch.pow(torch.clamp(self.margin - similarity, min=0.0), 2)
        return torch.mean(loss)

def create_dataloader(train_dataset, tokenizer, all_contexts, batch_size=16, shuffle=True, num_negatives=1):
    dataset = RetrievalDataset(train_dataset, tokenizer, all_contexts, num_negatives=num_negatives)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=RetrievalDataset.collate_fn)

def train_dense(model, dataloader, optimizer, scheduler, criterion, device, epochs=3):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            q_input_ids = batch['q_input_ids'].to(device)
            q_attention_mask = batch['q_attention_mask'].to(device)
            q_token_type_ids = batch['q_token_type_ids'].to(device)
            p_input_ids = batch['p_input_ids'].to(device)
            p_attention_mask = batch['p_attention_mask'].to(device)
            p_token_type_ids = batch['p_token_type_ids'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()

            # Forward pass
            p_embeddings, q_embeddings = model(p_input_ids, p_attention_mask, p_token_type_ids,
                                              q_input_ids, q_attention_mask, q_token_type_ids)
            # 유사도 계산
            sim_scores = torch.cosine_similarity(q_embeddings, p_embeddings)  # (batch_size*2, )

            if isinstance(criterion, ContrastiveLoss):
                loss = criterion(sim_scores, labels)
            elif isinstance(criterion, nn.BCEWithLogitsLoss):
                loss = criterion(sim_scores, labels)
            else:
                raise ValueError("Unsupported loss function")

            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")

def encode_documents(model, wiki_docs, tokenizer, device, batch_size=32):
    model.eval()
    doc_embeddings = []
    doc_ids = []
    contexts = []

    dataloader = DataLoader(wiki_docs, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding Documents"):
            # batch가 dict 형태로 반환됨을 가정
            if isinstance(batch, dict):
                contexts_batch = batch['text']
                doc_ids_batch = batch['id']
            else:
                raise TypeError(f"Unexpected batch type: {type(batch)}")

            # 토크나이징
            encodings = tokenizer(
                contexts_batch,
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            token_type_ids = encodings.get('token_type_ids', torch.zeros_like(input_ids)).to(device)

            # 임베딩 추출
            embeddings = model.encode_passage(input_ids, attention_mask, token_type_ids)
            embeddings = embeddings.cpu().numpy()

            doc_embeddings.append(embeddings)
            doc_ids.extend(doc_ids_batch)
            contexts.extend(contexts_batch)

    doc_embeddings = np.vstack(doc_embeddings)  # (num_docs, hidden_size)
    return doc_embeddings, doc_ids, contexts

def get_relevant_doc_bulk_dense(model, queries, doc_embeddings, contexts, tokenizer, device, k=20):
    model.eval()
    q_seqs = tokenizer(queries, padding="max_length", truncation=True, max_length=128, return_tensors='pt')
    q_input_ids = q_seqs['input_ids'].to(device)
    q_attention_mask = q_seqs['attention_mask'].to(device)
    q_token_type_ids = q_seqs.get('token_type_ids', torch.zeros_like(q_seqs['input_ids'])).to(device)

    with torch.no_grad():
        q_embeddings = model.encode_queries(q_input_ids, q_attention_mask, q_token_type_ids)
        q_embeddings = q_embeddings.cpu().numpy()  # (num_queries, hidden_size)

    # 코사인 유사도 계산
    similarities = cosine_similarity(q_embeddings, doc_embeddings)  # (num_queries, num_docs)

    # 상위 k개의 인덱스 추출
    topk_indices = np.argsort(-similarities, axis=1)[:, :k]  # (num_queries, k)
    topk_scores = np.take_along_axis(similarities, topk_indices, axis=1)

    # 상위 k개의 문서 추출
    relevant_docs = [[contexts[idx] for idx in query_indices] for query_indices in topk_indices]
    relevant_scores = [scores.tolist() for scores in topk_scores]

    return relevant_docs, relevant_scores

def search_and_create_dataset(model, test_dataset, doc_embeddings, contexts, tokenizer, device, k=20):
    queries = test_dataset['question']
    relevant_docs, relevant_scores = get_relevant_doc_bulk_dense(model, queries, doc_embeddings, contexts, tokenizer, device, k=k)

    total = []
    for idx, example in enumerate(tqdm(test_dataset, desc='Dense Retrieval')):
        tmp = {
            'question': example['question'],
            'id': example['id'],
            'context': ' '.join(relevant_docs[idx]),  # 관련 문서들을 하나의 문자열로 결합
        }
        # 'answers' 필드가 존재하는지 확인하고, 없으면 기본값 설정
        if 'answers' in example and example['answers']:
            tmp['answers'] = example['answers']
        else:
            tmp['answers'] = {'answer_start': [], 'text': []}  # 기본값 설정
        total.append(tmp)

    # 데이터셋 생성
    f = Features(
        {
            "context": Value(dtype="string", id=None),
            "id": Value(dtype="string", id=None),
            "question": Value(dtype="string", id=None),
            "answers": {
                "answer_start": [Value(dtype="int32", id=None)],
                "text": [Value(dtype="string", id=None)]
            }
        }
    )
    df = pd.DataFrame(total)
    print(f"DataFrame columns: {df.columns.tolist()}")  # 디버깅: 컬럼 확인
    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return datasets

class DenseRetrievalSearch:
    def __init__(self, model_class, model_path, tokenizer_name, embeddings_path, doc_ids_path, contexts_path, device='cpu'):
        self.device = device

        # 토크나이저 및 모델 로드
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = model_class(model_name=tokenizer_name)
        self.model.to(self.device)

        # 모델 가중치 로드
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 가중치 파일이 존재하지 않습니다: {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print("모델과 토크나이저가 로드되었습니다.")

        # 문서 임베딩 로드
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(f"임베딩 파일이 존재하지 않습니다: {embeddings_path}")
        self.doc_embeddings = np.load(embeddings_path)
        print(f"문서 임베딩이 {embeddings_path}에서 로드되었습니다. 임베딩 shape: {self.doc_embeddings.shape}")

        # 문서 ID 및 내용 로드
        with open(doc_ids_path, 'r', encoding='utf-8') as f:
            self.doc_ids = json.load(f)
        with open(contexts_path, 'r', encoding='utf-8') as f:
            self.contexts = json.load(f)
        print(f"문서 ID와 내용이 로드되었습니다. 총 문서 수: {len(self.doc_ids)}")

        # 문서 임베딩 정규화 (코사인 유사도 계산을 위해)
        self.doc_embeddings_normalized = self.doc_embeddings / np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True)
        print("문서 임베딩이 정규화되었습니다.")

    def get_dense_embedding(self):
        """
        Dense 임베딩을 저장하거나, 이미 존재하면 로드합니다.
        """
        pickle_name = f"Dense_embedding.npy"
        emd_path = os.path.join('/data/ephemeral/retrieval_result', pickle_name)

        if os.path.isfile(emd_path):
            self.doc_embeddings = np.load(emd_path)
            print('Dense 임베딩을 로드했습니다.')
        else:
            print('Passage embedding을 만듭니다.')
            pass  

    def encode_query(self, query):
        """
        쿼리를 임베딩으로 변환
        """
        inputs = self.tokenizer(query, return_tensors='pt', truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.encode_queries(inputs['input_ids'], inputs['attention_mask'], inputs.get('token_type_ids', torch.zeros_like(inputs['input_ids'])))
            query_embedding = outputs.cpu().numpy()
        # 정규화
        query_embedding_normalized = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        print("쿼리 임베딩이 생성되고 정규화되었습니다.")
        return query_embedding_normalized

    def search_query_dense(self, query, top_n=20):
        """
        Dense Retrieval을 사용하여 쿼리에 대한 상위 N개 문서 검색
        """
        query_embedding = self.encode_query(query)
        # 코사인 유사도 계산
        similarities = np.dot(self.doc_embeddings_normalized, query_embedding.T).flatten()
        # 상위 N개 문서 인덱스 추출 (중복 제거)
        top_indices = np.argsort(-similarities)[:top_n]

        results = [(index, similarities[index]) for index in top_indices]
        print(f"쿼리에 대한 상위 {top_n}개 문서를 Dense Retrieval으로 검색했습니다.")
        return results

    def get_document(self, index):
        """
        인덱스에 해당하는 문서 내용 반환
        """
        try:
            return self.contexts[index]
        except IndexError as e:
            print(f"문서 인덱스 조회 오류 {index}: {e}")
            return None

    def search_and_create_dataset_dense(self, test_data_route, k=20):
        """
        Dense Retrieval을 통해 검색한 결과를 DatasetDict 형태로 변환
        """
        # 테스트 데이터셋 로드
        test_dataset = load_from_disk(test_data_route)['validation']
        queries = test_dataset['question']
        print(f"테스트 데이터셋의 총 쿼리 수: {len(queries)}")

        total = []
        for idx, example in enumerate(tqdm(test_dataset, desc='Dense Retrieval')):
            query = example['question']
            search_results = self.search_query_dense(query, top_n=k)
            retrieved_contexts = [self.get_document(doc_idx) for doc_idx, _ in search_results]
            retrieved_contexts = [ctx for ctx in retrieved_contexts if ctx is not None]  # None 제거

            # 중복 문서 제거 (순서를 유지하면서)
            seen = set()
            unique_contexts = []
            for ctx in retrieved_contexts:
                if ctx not in seen:
                    unique_contexts.append(ctx)
                    seen.add(ctx)

            # 'context'는 관련 문서들을 하나의 문자열로 결합
            combined_context = ' ||| '.join(unique_contexts)  # 구분자 추가 가능

            result_entry = {
                'question': query,
                'id': example['id'],
                'context': combined_context
            }

            total.append(result_entry)

        # 데이터셋 생성
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
                # "answers": {
                #     "answer_start": [Value(dtype="int32", id=None)],
                #     "text": [Value(dtype="string", id=None)]
                # }
            }
        )
        df = pd.DataFrame(total)
        print(f"DataFrame columns: {df.columns.tolist()}")  # 디버깅: 컬럼 확인
        datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
        print("DatasetDict가 성공적으로 생성되었습니다.")
        return datasets
