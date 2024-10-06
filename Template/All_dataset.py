import pandas as pd
import torch.nn as nn
import os
import transformers
import numpy as np
from datasets import load_from_disk
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset
import torch
import random


#  -----------------Dense Embedding을 위해 Dataset클래스를 정의하는 부분입니다------------------------------------------------------


class DenseDataset(Dataset):
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
        self.dataset = load_from_disk(args.data_route)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name, trust_remote_code = True)
        

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
        # token의 캐릭터 단위 position를 찾을 수 있도록 offset mapping을 사용합니다.
        # start_positions과 end_positions을 찾는데 도움을 줄 수 있습니다.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # 데이터셋에 "start position", "enc position" label을 부여합니다.
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

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
            load_from_cache_file = self.args.load_from_cache_file,  
            )
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
            self.prepare_validation_features,
            batched = True,
            num_proc = self.args.num_proc,
            remove_columns = column_names,
            load_from_cache_file = True,
        )
        return eval_dataset
    
# -----------------------------------Dense Embedding을 위한 데이터셋을 선언하는 부분입니다--------------------------------------------------------


    def get_dense_train_dataset(self):
        training_dataset = self.dataset['train']
        num_neg = self.args.num_neg
        p_with_neg = []
        labels = []
        corpus = list(set(self.dataset['train']['context']))
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.args.model_name)
        corpus_set = set(corpus)  # 집합으로 변환하여 검색 속도 향상

        for context in training_dataset['context']:
            while True:
                neg_idxs = np.random.choice(len(corpus), size=num_neg, replace=False)
                neg_samples = [corpus[i] for i in neg_idxs]

                if context not in corpus_set.intersection(neg_samples):  # 집합을 사용하여 확인
                    p_neg = neg_samples + [context]  # 긍정 예시 추가
                    random.shuffle(p_neg)

                    context_index = p_neg.index(context)  # context의 위치 찾기
                    labels.append(context_index)  # index 추가 (0 기반 인덱스)
                    p_with_neg.extend(p_neg)  # 섞인 리스트 추가
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
        corpus = list(set(self.dataset['train']['context']))
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.args.model_name)
        corpus_set = set(corpus)  # 집합으로 변환하여 검색 속도 향상

        for context in validation_dataset['context']:
            while True:
                neg_idxs = np.random.choice(len(corpus), size=num_neg, replace=False)
                neg_samples = [corpus[i] for i in neg_idxs]

                if context not in corpus_set.intersection(neg_samples):  # 집합을 사용하여 확인
                    p_neg = neg_samples + [context]  # 긍정 예시 추가
                    random.shuffle(p_neg)

                    context_index = p_neg.index(context)  # context의 위치 찾기
                    labels.append(context_index)  # index 추가 (0 기반 인덱스)
                    p_with_neg.extend(p_neg)  # 섞인 리스트 추가
                    break


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
    
    def get_dense_faiss_corpus(self):
        corpus = list(set(self.dataset['train']['context'] + self.dataset['validation']['context']))
        return corpus
    
    def get_dense_queries_for_search(self):
        test_dataset = load_from_disk(self.args.test_data_route)
        query_vectors = self.tokenizer(test_dataset['validation']['question'], padding='max_length', 
                        truncation=True, return_tensors='pt')
        return query_vectors

