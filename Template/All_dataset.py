import pandas as pd
import torch.nn as nn
import os
import transformers
import numpy as np
from datasets import load_from_disk, Dataset
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset as torchDataset # datasets의 Class Dataset과
# torch.utils.data의 Dataset이 겹쳐 하나는 alias로 설정했습니다.
import json
from copy import deepcopy
import torch
import random


#  -----------------Dense Embedding을 위해 Dataset클래스를 정의하는 부분입니다------------------------------------------------------


class DenseDataset(torchDataset):
    def __init__(self, p_input_ids, p_attention_mask, p_token_type_ids, q_input_ids = None,
                  q_attention_mask = None, q_token_type_ids = None, labels = None):
        self.p_input_ids = p_input_ids
        self.p_attention_mask = p_attention_mask
        self.p_token_type_ids = p_token_type_ids
        self.q_input_ids = q_input_ids
        self.q_attention_mask = q_attention_mask
        self.q_token_type_ids = q_token_type_ids
        self.labels = labels

    def __len__(self):
        return len(self.p_input_ids)

    def __getitem__(self, idx):
        return {
            'p_input_ids': self.p_input_ids[idx],
            'p_attention_mask': self.p_attention_mask[idx],
            'p_token_type_ids': self.p_token_type_ids[idx],
            'q_input_ids': self.q_input_ids[idx],
            'q_attention_mask': self.q_attention_mask[idx],
            'q_token_type_ids': self.q_token_type_ids[idx],
            'labels' : self.labels[idx]
        }
    
#  --------------------------MRC 모델을 학습하기 위한 mrc_dataset을 선언하는 부분입니다-----------------------------------------------------------



class prepare_dataset:
    def __init__(self, args):
        self.args = args
        self.dataset = load_from_disk(self.args.data_route)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.args.model_name, trust_remote_code = True)

    def get_pure_dataset(self):
        return load_from_disk(self.args.data_route)

    def prepare_train_features(self, examples):
        # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
        # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
        tokenized_examples = self.tokenizer(
            examples['question'], 
            examples['context'] ,
            truncation ="only_second" ,
            max_length = self.args.max_seq_length,
            stride = self.args.doc_stride,
            return_overflowing_tokens = True,
            return_offsets_mapping = True,
            return_token_type_ids = True, # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
            padding = "max_length" ,
        )
        
        # 길이가 긴 context가 등장할 경우 truncate를 진행해야하므로, 해당 데이터셋을 찾을 수 있도록 mapping 가능한 값이 필요합니다.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = deepcopy(tokenized_examples["offset_mapping"])
        tokenized_examples["example_id"] = []
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            context_index = 1
            sequence_ids = tokenized_examples.sequence_ids(i)
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]


        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)  # cls index

            # sequence id를 설정합니다 (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # 하나의 example이 여러개의 span을 가질 수 있습니다.
            sample_index = sample_mapping[i]
            answers = examples['answers'][sample_index]

            # answer가 없을 경우 cls_index를 answer로 설정합니다(== example에서 정답이 없는 경우 존재할 수 있음).
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # text에서 정답의 Start/end character index
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # text에서 current span의 Start token index
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1

                # text에서 current span의 End token index
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

                # 정답이 span을 벗어났는지 확인합니다(정답이 없는 경우 CLS index로 label되어있음).
                if not (
                    offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char
                ):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # token_start_index 및 token_end_index를 answer의 끝으로 이동합니다.
                    # Note: answer가 마지막 단어인 경우 last offset을 따라갈 수 있습니다(edge case).
                    while (
                        token_start_index < len(offsets)
                        and offsets[token_start_index][0] <= start_char
                    ):
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    def get_mrc_train_dataset(self):
        column_names = self.dataset["train"].column_names
        train_dataset = self.dataset['train'].map(
            self.prepare_train_features,
            batched = True,
            num_proc = self.args.num_proc,
            remove_columns = column_names,
            load_from_cache_file = False
            )
        def remove_keys(example):
        # 각 샘플에서 'example_id'와 'offset_mapping' 제거
            example.pop("example_id", None)
            example.pop("offset_mapping", None)
            return example

            # 데이터셋에 일괄 적용
        train_dataset = train_dataset.map(remove_keys)
        return train_dataset
        
    def prepare_validation_features(self, examples):
        tokenized_examples = self.tokenizer(
            examples['question'],
            examples['context'],
            truncation = "only_second",
            max_length = self.args.max_seq_length,
            stride = self.args.doc_stride,
            return_overflowing_tokens = True,
            return_offsets_mapping = True,
            padding = "max_length",
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1

            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples
    
    def get_mrc_eval_dataset(self):
        column_names = self.dataset["validation"].column_names
        eval_exmaples = self.dataset['validation']
        eval_dataset = eval_exmaples.map(
            self.prepare_train_features,
            batched = True,
            num_proc = self.args.num_proc,
            remove_columns = column_names,
            load_from_cache_file = False,
        )
        return eval_dataset
    
    def get_mrc_test_dataset(self, test_dataset):
        column_names = test_dataset['validation'].column_names
        eval_examples = test_dataset['validation']

        eval_dataset = test_dataset.map(
            self.prepare_validation_features,
            batched = True,
            num_proc = self.args.num_proc,
            remove_columns = column_names,
            load_from_cache_file = False,
        )
        return eval_dataset, eval_examples

# -----------------------------------Dense Embedding을 위한 데이터셋을 선언하는 부분입니다--------------------------------------------------------


    def get_dense_train_dataset(self):
        training_dataset = self.dataset['train']
        num_neg = self.args.num_neg
        p_with_neg = []  # 패시지(긍정 + 부정 예시) 저장 리스트
        labels = []  # 정답 레이블 저장 리스트

        # 위키 데이터 로드
        with open(self.args.wiki_route, 'r', encoding='utf-8') as f:
            wiki = json.load(f)

        corpus = list(dict.fromkeys([v['text'] for v in wiki.values()]))

        tokenizer = transformers.AutoTokenizer.from_pretrained(self.args.model_name)

        for context in training_dataset['context']:
            while True:
                neg_idxs = np.random.choice(len(corpus), size=num_neg, replace=False)  # 부정 예시 랜덤 선택
                neg_samples = [corpus[i] for i in neg_idxs]

                # 부정 예시에 긍정 예시인 context가 없을 경우
                if context not in neg_samples:
                    p_neg = neg_samples + [context]  # 부정 예시와 긍정 예시를 합침
                    random.shuffle(p_neg)  # 긍정 예시와 부정 예시의 순서를 섞음

                    context_index = p_neg.index(context)  # 섞인 리스트에서 긍정 예시의 위치 찾기
                    labels.append(context_index)  # 위치를 정답 레이블에 추가
                    p_with_neg.extend(p_neg)  # 섞인 리스트를 전체 패시지에 추가
                    break  



        q_seqs = tokenizer(training_dataset['question'], padding='max_length', 
                        truncation=True, return_tensors='pt')
        p_seqs = tokenizer(p_with_neg, padding='max_length', truncation=True,
                        return_tensors='pt')
        max_len = p_seqs['input_ids'].size(-1)
        
        p_seqs['input_ids'] = p_seqs['input_ids'].view(-1, num_neg + 1, max_len)
        p_seqs['attention_mask'] = p_seqs['attention_mask'].view(-1, num_neg + 1, max_len)
        p_seqs['token_type_ids'] = p_seqs['token_type_ids'].view(-1, num_neg + 1, max_len)
        

        train_dataset = DenseDataset(
            p_seqs['input_ids'],
            p_seqs['attention_mask'],
            p_seqs['token_type_ids'],
            q_seqs['input_ids'],
            q_seqs['attention_mask'],
            q_seqs['token_type_ids'],
            labels
        )

        return train_dataset

            

    def get_dense_valid_dataset(self):
        validation_dataset = self.dataset['validation']
        num_neg = self.args.num_neg
        p_with_neg = []
        labels = []
        # 위키 데이터 로드
        with open(self.args.wiki_route, 'r', encoding='utf-8') as f:
            wiki = json.load(f)

        corpus = list(dict.fromkeys([v['text'] for v in wiki.values()]))

        tokenizer = transformers.AutoTokenizer.from_pretrained(self.args.model_name)

        for context in validation_dataset['context']:
            while True:
                neg_idxs = np.random.choice(len(corpus), size=num_neg, replace=False)  # 부정 예시 랜덤 선택
                neg_samples = [corpus[i] for i in neg_idxs]

                # 부정 예시에 긍정 예시인 context가 없을 경우
                if context not in neg_samples:
                    p_neg = neg_samples + [context]  # 부정 예시와 긍정 예시를 합침
                    random.shuffle(p_neg)  # 긍정 예시와 부정 예시의 순서를 섞음

                    context_index = p_neg.index(context)  # 섞인 리스트에서 긍정 예시의 위치 찾기
                    labels.append(context_index)  # 위치를 정답 레이블에 추가
                    p_with_neg.extend(p_neg)  # 섞인 리스트를 전체 패시지에 추가
                    break  # 반복문 종료 (새로운 부정 예시 선택이 필요 없음)


        q_seqs = tokenizer(validation_dataset['question'], padding='max_length', 
                        truncation=True, return_tensors='pt')
        p_seqs = tokenizer(p_with_neg, padding='max_length', truncation=True,
                        return_tensors='pt')

        max_len = p_seqs['input_ids'].size(-1)
        p_seqs['input_ids'] = p_seqs['input_ids'].view(-1, num_neg + 1, max_len)
        p_seqs['attention_mask'] = p_seqs['attention_mask'].view(-1, num_neg + 1, max_len)
        p_seqs['token_type_ids'] = p_seqs['token_type_ids'].view(-1, num_neg + 1, max_len)

        valid_dataset = DenseDataset(p_seqs['input_ids'], 
                                      p_seqs['attention_mask'],
                                      p_seqs['token_type_ids'],
                                      q_seqs['input_ids'], 
                                      q_seqs['attention_mask'],
                                      q_seqs['token_type_ids'],
                                      labels
        )

        return valid_dataset
    
    def get_dense_queries_for_search(self):
        test_dataset = load_from_disk(self.args.test_data_route)
        query_vectors = self.tokenizer(test_dataset['validation']['question'], padding='max_length', 
                        truncation=True, return_tensors='pt')
        return test_dataset, query_vectors

    def get_context(self):
        # 위키 데이터 로드
        with open(self.args.wiki_route, 'r', encoding='utf-8') as f:
            wiki = json.load(f)

        corpus = list(dict.fromkeys([v['text'] for v in wiki.values()]))
        return corpus

# ---------------------Generative based MRC 모델을 위한 데이터셋을 설정하는 부분입니다 -----------------


    def generative_MRC_preprocess_function(self, examples):
        tokenizer = self.tokenizer
        inputs = [f'question: {q}  context: {c}' for q, c in zip(examples['question'], examples['context'])]
        targets = [f'{a["text"][0]}' for a in examples['answers']]
        model_inputs = tokenizer(inputs,
                                  max_length = self.args.max_source_length,
                                  padding = self.args.padding,
                                  truncation = True,
                                  return_tensors='pt')

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer(): # context 문장과 target 문장의 인코딩이 다를 수 있기 때문에 이렇게 함
            labels = tokenizer(targets, max_length = self.args.max_target_length, padding = self.args.padding, truncation=True, return_tensors='pt')

        model_inputs["labels"] = labels["input_ids"]
        model_inputs["labels"][model_inputs["labels"] == tokenizer.pad_token_id] = -100
        model_inputs["example_id"] = []
        for i in range(len(model_inputs["labels"])):
            model_inputs["example_id"].append(examples["id"][i])
        return model_inputs
    
    def get_generative_MRC_train_dataset(self):
        col_names = self.dataset['train'].column_names
        train_dataset = self.dataset['train']
        train_dataset = train_dataset.map(
            self.generative_MRC_preprocess_function,
            batched = True,
            num_proc = self.args.preprocessing_num_workers,
            remove_columns = col_names,
            load_from_cache_file = False # 이전에 데이터셋에 대해 전처리를 수행한 결과가 로컬 캐시에 저장되어 있으면,
                                         # 다시 전처리 과정을 거치지 않고 그 캐시 파일을 불러옵니다.
        )
        return train_dataset
    
    def get_generative_MRC_valid_dataset(self):
        col_names = self.dataset['validation'].column_names
        eval_dataset = self.dataset['validation']
        eval_dataset = eval_dataset.map(
            self.generative_MRC_preprocess_function,
            batched = True,
            num_proc = self.args.preprocessing_num_workers,
            load_from_cache_file = False,
            remove_columns = col_names
        )
        return eval_dataset
    
