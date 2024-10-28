from bm25 import BM25Search
from DPR import Dense_embedding_retrieval
from TFIDFRetrieval import TF_IDFRetrieval
import transformers
from tqdm import tqdm
import gc
import torch
import pandas as pd
import numpy as np
from datasets import (Features,
                      DatasetDict,
                      Value,
                      Dataset)
from All_dataset import prepare_dataset
from arguments import Dense_search_retrieval_arguments

class retrieval:
    def __init__(self, mode = 'Dense'):
        if mode == 'Dense':
            self.dense = Dense_embedding_retrieval()
        elif mode == 'TFIDF':
            self.tf = TF_IDFRetrieval()
        elif mode == 'BM25':
            self.bm = BM25Search()
        self.args = Dense_search_retrieval_arguments
        self.datas = prepare_dataset(self.args)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def sequential_reranking(self, mode = 'test'):
        print(f'BM25로 100개를 가져온 뒤 Dense retrieval로 {self.args.top_k}개의 문서를 찾습니다.')
        test_dataset, query_vec = self.datas.get_dense_queries_for_search(mode = mode)
        query_vec.to(self.device)
        queries = test_dataset['question']
        BMtotal = []
        passages = self.datas.get_context()
        BM25 = BM25Search()
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.best_checkpoint if self.besk_checkpoint != None else self.args.model_name)
        # get_relevant_doc_bulk_bm25 메서드를 사용하여 문서 점수와 인덱스 가져오기
        _, doc_indices = BM25.get_relevant_doc_bulk_bm25(queries, k = 100)
        self.doc_indices = doc_indices

        for idx, example in enumerate(
            tqdm(test_dataset, desc='BM25 retrieval: ')
        ):
            tmp = {
                'question': example['question'],
                'id': example['id'],
                'context': [passages[pid] for pid in doc_indices[idx]],
                # 'doc_scores': doc_scores[idx],  # 점수 추가
            }
            if "context" in example.keys():
                tmp["original_context"] = example["context"]
                ground_truth_passage = example["context"]
                # 정답이 retrieved passages에 포함되는지 여부를 확인
            BMtotal.append(tmp)
        self.bm = BMtotal
        del BM25 # OOM 방지
        gc.collect()

        self.dense.model.p_model.eval()
        self.dense.model.q_model.eval()
        k = self.args.top_k
        Densetotal = []
        with torch.no_grad():
            q_embs = self.model.q_model(**query_vec).cpu() # (examples, 512)
            for idx, example in enumerate(tqdm(BMtotal, desc='sequential reranking')):
                # context에 대한 배치 처리
                batch_input = tokenizer(
                    example['context'], padding = 'max_length', truncation = True, return_tensors = 'pt'
                ).to(self.device) # (100, 512)
                p_embs = self.dense.model.p_model(**batch_input).cpu() # (100, 768)
                # 유사도 계산
                sim_scores = torch.matmul(q_embs[idx].view(1,-1), p_embs.T) # (1,100)
                topk_indices = torch.argsort(sim_scores, dim=1, descending=True)[:, :k] # (1, k)
                # top-k 문서 가져오기
                tmp = {
                    'question': example['question'],
                    'id': example['id'],
                    'context': '[SEP]'.join([example['context'][pid] for pid in topk_indices[0]])
                    }
                if "original_context" in example.keys():
                    tmp["original_context"] = example["original_context"]
                    ground_truth_passage = example["original_context"]
                    retrieved_passages = [example['context'][pid] for pid in topk_indices[0]]
                    # 정답이 retrieved passages에 포함되는지 여부를 확인
                    tmp["answers"] = ground_truth_passage in retrieved_passages

                Densetotal.append(tmp)
        self.dense = Densetotal
        df = pd.DataFrame(Densetotal)

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
    
    def cross_reranking(self, mode = 'test', lamb_da = 1.1):
        print('BM25와 Dense retrieval의 score을 합한 뒤 sort하여 문서를 찾습니다.')
        print(f'현재 가중합은 {lamb_da}*Densescore + 1*BMscore입니다.')
        print('lamb_da = 뫄뫄 를 통해 람다를 바꿀 수 있습니다. (기본값 1.1)')
        test_dataset, query_vec = self.datas.get_dense_queries_for_search(mode = mode)
        passages = self.datas.get_context()
        BM25 = BM25Search()
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.args.model_name)

        with torch.no_grad():
            query_dataset = torch.utils.data.TensorDataset(query_vec['input_ids'], query_vec['attention_mask'], query_vec['token_type_ids'])
            batch_size = 16  # 원하는 배치 크기 설정
            query_loader = torch.utils.data.DataLoader(query_dataset, batch_size=batch_size)

            # 배치 처리
            q_embs_list = []
            for batch in query_loader:
                input_ids, attention_mask, token_type_ids = [t.to(self.device) for t in batch]
                
                # 배치 데이터를 모델에 넣어 q_embs 생성
                batch_query_vec = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}
                q_embs = self.dense.model.q_model(**batch_query_vec).cpu()

                # 각 배치의 임베딩 결과를 리스트에 추가
                q_embs_list.append(q_embs)

            # 배치 결과들을 모두 합침
            q_embs = torch.cat(q_embs_list, dim=0)
            p_embs = []
            batch_size = self.args.per_device_train_batch_size
            for i in tqdm(range(0, len(passages), batch_size)):
                batch_passages = passages[i:i + batch_size]
                tokenized_batch = tokenizer(batch_passages, padding='max_length', truncation=True, return_tensors='pt').to(self.device)
                p_emb = self.dense.model.p_model(**tokenized_batch).cpu().numpy()
                p_embs.append(p_emb)

            # 배치별 임베딩을 모두 연결
            p_embs = np.vstack(p_embs)
            p_embs = torch.tensor(np.array(p_embs)).squeeze()
        print('q_emb :', q_embs.size(), 'p_emb :', p_embs.size())

        dense_scores = torch.matmul(q_embs, torch.transpose(p_embs, 0, 1)) # (쿼리개수, 50000)
        del self.dense
        gc.collect() # OOM 방지

        queries = test_dataset['question']
        bm_scores =  torch.tensor(BM25.get_relevant_doc_bulk_bm25(queries, k = len(p_embs), mode = 'ensemble')) # (쿼리개수, 50000)
        total_scores = (lamb_da * dense_scores) + bm_scores # 가중합
        doc_indices = torch.argsort(total_scores, dim=1, descending=True)[:, :self.args.top_k]

        total = []
        for idx, example in enumerate(tqdm(test_dataset, desc = "cross reranking: ")):
            tmp = {
                "question": example["question"],
                "id": example["id"],
                "context": "[SEP]".join([passages[pid] for pid in doc_indices[idx]]),
            }

            if "context" in example.keys() and "answers" in example.keys():
                tmp["original_context"] = example["context"]
                
                ground_truth_passage = example["context"]
                retrieved_passages = [passages[pid] for pid in doc_indices[idx]]
                
                # 정답이 retrieved passages에 포함되는지 여부를 확인
                tmp["answers"] = ground_truth_passage in retrieved_passages  # True or False
            total.append(tmp)

        df = pd.DataFrame(total)

        if mode == 'test':
            f = Features(
                {
                    "context": Value(dtype="string", id=None),
                    "id": Value(dtype="string", id=None),
                    "question": Value(dtype="string", id=None),
                }
            )
            datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
        if mode == 'eval' or mode == 'making':
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


#