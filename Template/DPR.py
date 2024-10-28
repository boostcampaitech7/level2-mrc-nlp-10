from arguments import Dense_search_retrieval_arguments
from All_dataset import prepare_dataset
import torch
import torch.nn.functional as F
import transformers
from transformers import (PreTrainedModel,
                          AutoModel,
                          DefaultDataCollator,
                          TrainingArguments,
                          Trainer,
                          AutoTokenizer)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
from glob import glob
import os
import json
import faiss
from datasets import (DatasetDict,
                      Dataset,
                       Features,
                         Value,)
import wandb

class bert_model(PreTrainedModel):
    def __init__(self, model_name, config):
        self.args = Dense_search_retrieval_arguments
        super(bert_model, self).__init__(config)

        config = self.config

        self.bert = AutoModel.from_pretrained(model_name, config = config)

    def forward(self, input_ids,
                attention_mask=None, token_type_ids=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)

        pooled_output = outputs[1]
        return pooled_output
  
class Dense_embedding_retrieval_model(PreTrainedModel):
    def __init__(self, checkpoint = None):
        self.args = Dense_search_retrieval_arguments
        checkpoint = self.args.model_name if checkpoint == None else checkpoint
        config = transformers.AutoConfig.from_pretrained(checkpoint)
        super(Dense_embedding_retrieval_model, self).__init__(config)
        self.p_model = bert_model(checkpoint, config = config)
        # self.q_model = bert_model(checkpoint, config = config)
        self.q_model = self.p_model


    def forward(self, p_input_ids, p_attention_mask, p_token_type_ids,
                q_input_ids, q_attention_mask, q_token_type_ids, labels = None):
        
        p_input = {
            'input_ids': p_input_ids,
            'attention_mask': p_attention_mask,
            'token_type_ids': p_token_type_ids
        }
        q_input = {
            'input_ids': q_input_ids,
            'attention_mask': q_attention_mask,
            'token_type_ids': q_token_type_ids
        }

        p_outputs = self.p_model(**p_input)
        q_outputs = self.q_model(**q_input)

        sim_scores = torch.matmul(q_outputs, p_outputs.T) # (batch_size, batch_size)
        # Negative Log Likelihood 적용
        labels = torch.arange(0, len(p_outputs)).long().to(self.device)
        loss = F.cross_entropy(sim_scores, labels)

    
        return {
            'loss': loss,
            'output' : sim_scores
        }


class Dense_embedding_retrieval:
    def __init__(self, checkpoint = None):
        self.args = Dense_search_retrieval_arguments()
        self.model = Dense_embedding_retrieval_model()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.trainer = None
        if self.args.use_wandb:
            self.start_wandb()
        self.datas = prepare_dataset(self.args)
        self.besk_checkpoint = None
        self.output_dir = self.args.output_dir + '_' + self.args.model_name.split('/')[-1]


    def compute_metrics(self,eval_preds):

        logits, labels = eval_preds  
        logit_size = logits.shape[0]  # logits의 배치 크기
        num_classes = self.args.per_device_train_batch_size
        labels = torch.arange(num_classes).repeat((logit_size + num_classes - 1) // num_classes)[:logit_size].cpu().numpy()
        # label을 0 1 2 3 4 5 6 7 0 1 2 3 이런식으로
        predictions = torch.argmax(torch.tensor(logits), dim=1).detach().cpu().numpy()
        accuracy = accuracy_score(labels, predictions)  
        f1 = f1_score(labels, predictions, average='weighted') 

        return {'accuracy' : accuracy, 'f1' : f1 }


    def get_trainer(self):
        train_dataset = self.datas.get_dense_dataset(mode = 'train')
        valid_dataset = self.datas.get_dense_dataset(mode = 'valid')
        gc.collect()
        torch.cuda.empty_cache()
        data_collator = DefaultDataCollator()
        self.training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.args.num_train_epochs,
            gradient_accumulation_steps=1,
            overwrite_output_dir=True,
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            per_device_eval_batch_size=self.args.per_device_eval_batch_size,
            learning_rate=self.args.learning_rate,
            evaluation_strategy="epoch",
            # eval_steps=132,  # 평가 주기를 327로 설정
            save_strategy="epoch",
            # save_steps=132,  # 저장 주기를 327로 설정
            logging_steps=30,
            fp16=True,
            logging_dir='./logs',
            load_best_model_at_end=True,
            do_eval=True,
            weight_decay=0.01,
        )
        if self.args.use_wandb:
            self.training_args.report_to = ["wandb"]
            self.training_args.run_name = "default"


        self.trainer = Trainer(
            model = self.model,
            args = self.training_args,
            train_dataset = train_dataset,
            eval_dataset = valid_dataset,
            compute_metrics = self.compute_metrics,
            data_collator = data_collator   
        )
        torch.cuda.empty_cache()
        gc.collect()

    def load_model(self):
        self.training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            do_train=False,
            do_eval=True,
            per_device_eval_batch_size=self.args.per_device_eval_batch_size,  
            fp16=True
        )
        
        model_path = self.output_dir
        checkpoints = sorted(glob(model_path + '/checkpoint-*'), key=lambda x: int(x.split('-')[-1]), reverse=True)
        
        if not checkpoints:
            raise ValueError("Checkpoint 파일이 없습니다.")

        lastcheckpt = checkpoints[0]
        trainer_state_file = os.path.join(lastcheckpt, 'trainer_state.json')
        
        print('제일 마지막 checkpoint:', trainer_state_file)
        
        best_checkpoint = lastcheckpt  # 기본값으로 마지막 체크포인트
        if os.path.exists(trainer_state_file):
            with open(trainer_state_file, 'r') as f:
                trainer_state = json.load(f)
                best_checkpoint = trainer_state.get('best_model_checkpoint', lastcheckpt)
                print('best checkpoint:', best_checkpoint)
 
        # best model checkpoint 로드
        self.model = Dense_embedding_retrieval_model(best_checkpoint)  # 이 부분에서 로딩
        self.best_checkpoint = best_checkpoint
        self.model.to(self.device)
        print("bestmodel 체크포인트로 모델과 Trainer가 로드되었습니다.")


    def train(self):
        assert self.trainer is not None, "trainer가 없습니다. 로드하거나 get_trainer 하세요."
        self.trainer.train()
        torch.cuda.empty_cache()
    
    def train_kfold(self):
        kf = KFold(n_splits = self.args.kfold, shuffle = True, random_state = 42,)
        kfold_dataset = self.datas.get_dense_dataset(mode = 'train')

        self.training_args.num_train_epochs = self.args.epoch_for_kfold


        for fold, (train_idx, valid_idx) in enumerate(kf.split(kfold_dataset)):
            print(f'----------------- fold {fold + 1} -------------------------')
            self.get_trainer()
            self.training_args.num_train_epochs = self.args.epoch_for_kfold

            self.trainer.train_dataset = torch.utils.data.Subset(kfold_dataset, train_idx)  # Subset으로 변경
            self.trainer.eval_dataset = torch.utils.data.Subset(kfold_dataset, valid_idx)  # Subset으로 변경
            self.trainer.train()
            
        torch.cuda.empty_cache()


    def build_faiss(self):
        num_clusters = self.args.num_clusters
        indexer_name = f"Dense_faiss_clusters{num_clusters}_{self.args.model_name.split('/')[-1]}.index"
        
        # 데이터 경로와 인덱스 파일을 나눔
        dir_path = self.args.data_path
        indexer_path = os.path.join(dir_path, indexer_name)
        
        print(indexer_path)
        
        if os.path.isfile(indexer_path):
            print("Faiss Indexer을 로드했습니다.")
            self.indexer = faiss.read_index(indexer_path)

        else:
            print("Building Faiss Indexer.")
            
            # 디렉터리가 존재하지 않으면 생성
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)  # 경로는 파일이 아닌 디렉터리 경로여야 함
                print(f"'{dir_path}' 디렉터리가 생성되었습니다.")
            else:
                print(f"'{dir_path}' 디렉터리가 이미 존재합니다.")

            self.model.eval()
            passages = self.datas.get_context()[:1000]
            tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)

            with torch.no_grad():
                embeddings = []
                for passage in tqdm(passages):
                    inputs = tokenizer(passage, return_tensors='pt', padding='max_length', truncation=True).to(self.device)
                    embedding = self.model.p_model(**inputs).squeeze(0)  # passage 모델에서 출력
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

    def search_query_faiss(self):
        top_k = self.args.top_k
        passages = self.datas.get_context()
        self.model.eval()
        if not os.path.exists('retrieval_result'):
            os.makedirs('retrieval_result')

        with torch.no_grad():
            test_dataset, inputs = self.datas.get_dense_queries_for_search()  
            inputs = {key: value.to(self.device) for key, value in inputs.items()} 
            query_embeddings = self.model.q_model(**inputs).cpu().numpy() 

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
                    [passages[pid] for pid in doc_indices[idx]]
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
    
    def search_query(self, mode = 'test'):
        self.model.p_model.eval()
        self.model.q_model.eval()
        # 토크나이저 로드
        passages = self.datas.get_context()
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.args.model_name)
        
        # 테스트 데이터셋과 쿼리 벡터 가져오기
        test_dataset, query_vec = self.datas.get_dense_queries_for_search(mode = mode)

        with torch.no_grad():
            query_vec = query_vec.to(self.device)
            q_embs = self.model.q_model(**query_vec).cpu()
            # 패시지 데이터 가져오기
            p_embs = []
            batch_size = self.args.per_device_train_batch_size
            for i in tqdm(range(0, len(passages), batch_size)):
                batch_passages = passages[i:i + batch_size]
                tokenized_batch = tokenizer(batch_passages, padding='max_length', truncation=True, return_tensors='pt').to(self.device)
                p_emb = self.model.p_model(**tokenized_batch).cpu().numpy()
                p_embs.append(p_emb)

            # 배치별 임베딩을 모두 연결
            p_embs = np.vstack(p_embs)
            p_embs = torch.tensor(np.array(p_embs)).squeeze()
        print('q_emb :', q_embs.size(), 'p_emb :', p_embs.size())

        k = self.args.top_k
        sim_scores = torch.matmul(q_embs, torch.transpose(p_embs, 0, 1))

        doc_indices = torch.argsort(sim_scores, dim=1, descending=True)[:, :k]

        total = []
        for idx, example in enumerate(tqdm(test_dataset, desc = "Dense retrieval: ")):
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

        

    def start_wandb(self):
        os.system("rm -rf /root/.cache/wandb")
        os.system("rm -rf /root/.config/wandb")
        os.system("rm -rf /root/.netrc")
        
        # WandB API 키 설정 (여러분의 API 키를 넣어주시면 됩니다)
        os.environ["WANDB_API_KEY"] = self.args.wandb_key
        wandb.init(project='Dense_embedding_retrieval')