import transformers
from transformers import TrainingArguments, EvalPrediction, DataCollatorWithPadding
from trainer_qa import QuestionAnsweringTrainer
from utils_qa import postprocess_qa_predictions
from arguments import Extraction_based_MRC_arguments
from All_dataset import prepare_dataset
from datasets import load_metric, Dataset, DatasetDict
import os
from glob import glob
import numpy as np
import nltk
import warnings
import json
import torch
import wandb
import pickle
from copy import deepcopy
import copy

nltk.download('punkt')
warnings.filterwarnings('ignore')

def set_seed(seed: int = 42):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Extraction_based_MRC:
    def __init__(self):
        self.args = Extraction_based_MRC_arguments()
        self.config = transformers.AutoConfig.from_pretrained(self.args.model_name)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.args.model_name)
        self.model = transformers.AutoModelForQuestionAnswering.from_pretrained(
            self.args.model_name,
            config=self.config,
            trust_remote_code=True
        )
        
        # 경량화 체크
        self.model.gradient_checkpointing_enable()
        
        self.datas = prepare_dataset(self.args)
        self.datasets = self.datas.get_pure_dataset()
        self.metric = load_metric("squad")
        self.trainer = None
        self.output_dir = os.path.join(self.args.model_path, self.args.model_name.split('/')[-1])

        if self.args.use_wandb:
            self.start_wandb()

    def load_model(self):
        self.training_args = TrainingArguments(
            output_dir=self.output_dir,
            do_train=False,
            do_eval=True,
            per_device_eval_batch_size=self.args.per_device_eval_batch_size,
        )
        model_path = self.args.output_dir
        checkpoints = sorted(
            glob(os.path.join(model_path, 'checkpoint-*')),
            key=lambda x: int(x.split('-')[-1]),
            reverse=True
        )
        if checkpoints:
            last_checkpoint = checkpoints[0]
        else:
            last_checkpoint = model_path

        trainer_state_file = os.path.join(last_checkpoint, 'trainer_state.json')
        print('Latest checkpoint:', trainer_state_file)
        if os.path.exists(trainer_state_file):
            with open(trainer_state_file, 'r') as f:
                trainer_state = json.load(f)
                best_checkpoint = trainer_state.get('best_model_checkpoint', None)
                print('Best checkpoint:', best_checkpoint)
                if best_checkpoint is None:
                    best_checkpoint = last_checkpoint
        else:
            best_checkpoint = last_checkpoint

        self.config = transformers.AutoConfig.from_pretrained(best_checkpoint)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(best_checkpoint)
        self.model = transformers.AutoModelForQuestionAnswering.from_pretrained(
            best_checkpoint,
            config=self.config
        )
        
        #경량화 체크용
        self.model.gradient_checkpointing_enable()

        self.trainer = QuestionAnsweringTrainer(
            model=self.model,
            args=self.training_args,
            tokenizer=self.tokenizer,
            post_process_function=self.post_processing_function,
            compute_metrics=self.compute_metrics
        )
        print("Model and Trainer loaded from the best checkpoint.")

    def post_processing_function(self, examples, features, predictions, training_args):
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=False,
            n_best_size=self.args.n_best_size,
            max_answer_length=self.args.max_answer_length,
            null_score_diff_threshold=0.0,
            output_dir=self.output_dir,
            is_world_process_zero=self.trainer.is_world_process_zero(),
        )

        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
        references = [{"id": ex["id"], "answers": ex['answers']} for ex in examples]

        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    def compute_metrics(self, p: EvalPrediction):
        return self.metric.compute(predictions=p.predictions, references=p.label_ids)

    def get_trainer(self, train_dataset=None, eval_dataset=None, eval_examples=None):
        self.training_args = TrainingArguments(
            output_dir=self.output_dir,
            learning_rate=3e-5,
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            per_device_eval_batch_size=self.args.per_device_eval_batch_size,
            num_train_epochs=self.args.num_train_epochs,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            report_to=["wandb"] if self.args.use_wandb else [],
            run_name=f"Extraction_MRC_{self.args.model_name}",
            logging_dir='./logs',
            logging_strategy="epoch",
            load_best_model_at_end=True,
            fp16=True,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
        )

        data_collator = DataCollatorWithPadding(
            self.tokenizer,
            pad_to_multiple_of=8 if self.training_args.fp16 else None
        )

        self.trainer = QuestionAnsweringTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            eval_examples=eval_examples,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            post_process_function=self.post_processing_function,
            compute_metrics=self.compute_metrics,
        )

    def train(self, train_dataset=None, eval_dataset=None, retrieved_train_dataset=None, retrieved_eval_dataset=None):
        set_seed(self.args.seed)
        if train_dataset is None:
            train_dataset = self.datasets['train']
            print('Using the default train dataset.')
        if eval_dataset is None:
            eval_dataset = self.datasets['validation']
            print('Using the default validation dataset.')

        if retrieved_train_dataset is not None:
            train_dataset = self.add_retrieved_contexts(train_dataset, retrieved_train_dataset)
        if retrieved_eval_dataset is not None:
            eval_dataset = self.add_retrieved_contexts(eval_dataset, retrieved_eval_dataset)

        train_dataset = self.datas.get_mrc_train_dataset(train_dataset)
        eval_dataset = self.datas.get_mrc_eval_dataset(eval_dataset)

        self.get_trainer(train_dataset=train_dataset, eval_dataset=eval_dataset)
        
        # 사용 안하는 gpu 캐시 초기화
        torch.cuda.empty_cache()
        
        self.trainer.train()
        
    def add_retrieved_contexts(self, dataset, retrieved_dataset):
        retrieved_contexts = {}
        for ex in retrieved_dataset:
            retrieved_contexts[ex['id']] = ex['context']

        def combine_contexts(examples):
            new_contexts = []
            
            # retrieved_contexts는 retrieved_dataset에서 미리 생성된 문맥 정보로 가정
            for idx, id_ in enumerate(examples['id']):
                # test_dataset에는 'context'가 없을 수 있으므로 기본값을 빈 문자열로 설정
                if 'context' in examples:
                    original_context = examples['context'][idx] if idx < len(examples['context']) else ''
                else:
                    original_context = ''

                retrieved_context = retrieved_contexts.get(id_, '')  # retrieved_context가 없을 때도 빈 문자열
                combined_context = original_context + ' [SEP] ' + retrieved_context if retrieved_context else original_context
                new_contexts.append(combined_context)
            
            examples['context'] = new_contexts
            return examples


        dataset = dataset.map(combine_contexts, batched=True)
        return dataset
    
    # def add_retrieved_contexts(self, dataset, retrieved_dataset):
    #     if not isinstance(dataset, Dataset):
    #         raise TypeError(f"Expected 'dataset' to be a Dataset, but got {type(dataset)}")
    #     if not isinstance(retrieved_dataset, Dataset):
    #         raise TypeError(f"Expected 'retrieved_dataset' to be a Dataset, but got {type(retrieved_dataset)}")

    #     print(f"Columns in dataset: {dataset.column_names}")
    #     if len(dataset) > 0:
    #         print(f"First example in dataset: {dataset[0]}")
    #     print(f"Columns in retrieved_dataset: {retrieved_dataset.column_names}")
    #     if len(retrieved_dataset) > 0:
    #         print(f"First example in retrieved_dataset: {retrieved_dataset[0]}")

    #     # 'context' 컬럼이 retrieved_dataset에 있는지 확인
    #     if 'context' not in retrieved_dataset.column_names:
    #         raise KeyError("The 'retrieved_dataset' does not contain a 'context' column.")
    #     if 'id' not in retrieved_dataset.column_names:
    #         raise KeyError("The 'retrieved_dataset' does not contain an 'id' column.")

    #     # retrieved_dataset에서 id별로 context를 매핑
    #     retrieved_contexts = {ex['id']: ex['context'] for ex in retrieved_dataset}

    #     def combine_contexts(examples):
    #         new_contexts = []
    #         for id_ in examples['id']:
    #             retrieved_context = retrieved_contexts.get(id_, '')
    #             new_contexts.append(retrieved_context)
    #         examples['context'] = new_contexts
    #         return examples

    #     # test_dataset에 'context' 추가
    #     dataset = dataset.map(combine_contexts, batched=True)
    #     return dataset




    def start_wandb(self):
        os.environ["WANDB_API_KEY"] = self.args.wandb_key
        wandb.init(project='Dense_embedding_retrieval')

    def inference(self, test_dataset):
        test_dataset = test_dataset['validation']
        inputs = [f'question: {test_dataset["question"][i]}  context: {test_dataset["context"][i]} </s>' for i in range(len(test_dataset))]
        sample = self.tokenizer(inputs,
                                  max_length = self.args.max_source_length,
                                  padding = self.args.padding, 
                                  truncation = True,
                                  return_tensors='pt')
        self.model.to('cpu')
        outputs = self.model.generate(**sample, 
                                    max_length=self.args.max_target_length, 
                                    num_beams=self.args.num_beams)

        preds = [self.tokenizer.decode(output, skip_special_tokens=True) for output in tqdm(outputs, desc = "Decoding")]

        print('output을 id : answers 형태의 json파일로 변환합니다. 결과는 model.generative_mrc_results로 확인할 수 있습니다.')
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in tqdm(preds, desc = "Sentence Tokenization")]
        self.generative_mrc_results = pd.DataFrame({'id' : test_dataset['id'], 'answers' : preds})
        result_dict = self.generative_mrc_results.set_index('id')['answers'].to_dict()
        if not os.path.exists("predict_result"):
            os.makedirs("predict_result")
            print("폴더 'predict_result'가 생성되었습니다.")

        with open(f"predict_result/Generative_mrc_output_{self.args.model_name.split('/')[-1]}.json", 'w', encoding='utf-8') as json_file:
            json.dump(result_dict, json_file, ensure_ascii=False, indent=4)


    def train_multiple_models(self, model_names, train_dataset=None, eval_dataset=None,
                            retrieved_train_dataset=None, retrieved_eval_dataset=None):
        for model_name in model_names:
            print(f"Training model: {model_name}")
            set_seed(42)
            # Update model, tokenizer, and config for each model
            self.config = transformers.AutoConfig.from_pretrained(model_name)
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

            self.model = transformers.AutoModelForQuestionAnswering.from_pretrained(
                model_name,
                config=self.config,
                trust_remote_code=True
            )

           
            is_roberta = "roberta" in model_name.lower()


            self.datas.tokenizer = self.tokenizer
            self.datas.include_token_type_ids = not is_roberta  # RoBERTa 계열이면 token_type_ids를 사용하지 않음


            model_dir_name = model_name.replace('/', '_')
            self.output_dir = os.path.join(self.args.model_path, model_dir_name)


            if train_dataset is None:
                train_dataset_current = self.datasets['train']
                print('Using the default train dataset.')
            else:
                train_dataset_current = train_dataset

            if eval_dataset is None:
                eval_dataset_current = self.datasets['validation']
                print('Using the default validation dataset.')
            else:
                eval_dataset_current = eval_dataset

            if retrieved_train_dataset is not None:
                train_dataset_augmented = self.add_retrieved_contexts(train_dataset_current, retrieved_train_dataset)
            else:
                train_dataset_augmented = train_dataset_current

            if retrieved_eval_dataset is not None:
                eval_dataset_augmented = self.add_retrieved_contexts(eval_dataset_current, retrieved_eval_dataset)
            else:
                eval_dataset_augmented = eval_dataset_current

            eval_examples = eval_dataset_augmented

            train_dataset_prepared = self.datas.get_mrc_train_dataset(train_dataset_augmented)
            eval_dataset_prepared = self.datas.get_mrc_eval_dataset(eval_dataset_augmented)
            self.get_trainer(
                train_dataset=train_dataset_prepared,
                eval_dataset=eval_dataset_prepared,
                eval_examples=eval_examples
            )

            self.trainer.train()
            self.trainer.save_model(self.output_dir)


    # def ensemble_inference(self, test_dataset, retrieved_dataset=None, model_paths=None):
    #     if model_paths is None:
    #         raise ValueError("You must provide a list of model paths for ensembling.")

    #     # test_dataset은 DatasetDict 또는 Dataset으로 가정
    #     if isinstance(test_dataset, DatasetDict):
    #         test_dataset = test_dataset['validation']  # 원하는 분할 이름으로 변경 가능
    #         print("Extracted 'validation' split from test_dataset.")
    #     elif isinstance(test_dataset, Dataset):
    #         print("Using Dataset object for test_dataset.")
    #     else:
    #         raise TypeError(f"test_dataset is of unexpected type: {type(test_dataset)}")

    #     # copy() 호출 제거
    #     # test_dataset = test_dataset.copy()
    #     print(f"Type of test_dataset: {type(test_dataset)}")

    #     if retrieved_dataset is not None:
    #         # retrieved_dataset이 DatasetDict인 경우 'validation' 분할을 사용
    #         if isinstance(retrieved_dataset, DatasetDict):
    #             retrieved_dataset = retrieved_dataset['validation']  # 원하는 분할 이름으로 변경 가능
    #             print("Extracted 'validation' split from retrieved_dataset.")
    #         elif isinstance(retrieved_dataset, Dataset):
    #             print("Using Dataset object for retrieved_dataset.")
    #         else:
    #             raise TypeError(f"retrieved_dataset is of unexpected type: {type(retrieved_dataset)}")

    #         # 'add_retrieved_contexts'는 Dataset 객체를 받도록 수정
    #         test_dataset = self.add_retrieved_contexts(test_dataset, retrieved_dataset)
    #         print("Added retrieved contexts.")

    #     # get_mrc_test_dataset은 Dataset을 처리하도록 가정
    #     test_dataset_prepared, test_examples = self.datas.get_mrc_test_dataset(test_dataset)

    #     print(f"Type of test_dataset_prepared: {type(test_dataset_prepared)}")
    #     print(f"Type of test_examples: {type(test_examples)}")

    #     all_start_logits = []
    #     all_end_logits = []

    #     for model_path in model_paths:
    #         print(f"Loading model from {model_path} for inference")
    #         model = transformers.AutoModelForQuestionAnswering.from_pretrained(
    #             model_path,
    #             trust_remote_code=True
    #         )
    #         tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

    #         # Inference를 위한 TrainingArguments 생성
    #         inference_args = TrainingArguments(
    #             output_dir=os.path.join(model_path, "inference"),
    #             per_device_eval_batch_size=self.args.per_device_eval_batch_size,
    #             do_predict=True,
    #             # 추가적인 인수 설정
    #             disable_tqdm=True,  # 로그를 깔끔하게 하기 위해 tqdm 비활성화
    #             remove_unused_columns=False,
    #         )

    #         # 각 모델에 대한 Trainer 준비
    #         trainer = QuestionAnsweringTrainer(
    #             model=model,
    #             args=inference_args,
    #             tokenizer=tokenizer,
    #             data_collator=DataCollatorWithPadding(
    #                 tokenizer,
    #                 pad_to_multiple_of=8 if getattr(inference_args, 'fp16', False) else None
    #             ),
    #             post_process_function=self.post_processing_function,
    #             compute_metrics=self.compute_metrics,
    #         )

    #         # 예측 수행
    #         predictions = trainer.predict(
    #             test_dataset=test_dataset_prepared,
    #             test_examples=test_examples,
    #         )
    #         start_logits, end_logits = predictions.predictions
    #         all_start_logits.append(start_logits)
    #         all_end_logits.append(end_logits)

    #         # 메모리 확보를 위해 모델과 트레이너 삭제
    #         del model
    #         del trainer
    #         torch.cuda.empty_cache()

    #     # 모든 모델의 로짓을 평균
    #     avg_start_logits = np.mean(all_start_logits, axis=0)
    #     avg_end_logits = np.mean(all_end_logits, axis=0)

    #     # 최종 예측 생성 (답변)
    #     final_predictions = postprocess_qa_predictions(
    #         examples=test_examples,
    #         features=test_dataset_prepared,
    #         predictions=(avg_start_logits, avg_end_logits),
    #         version_2_with_negative=False,
    #         n_best_size=self.args.n_best_size,
    #         max_answer_length=self.args.max_answer_length,
    #         null_score_diff_threshold=0.0,
    #         output_dir=self.output_dir,
    #         is_world_process_zero=True,  # 단일 프로세스로 가정
    #     )

    #     # 예측을 id와 답변 형태로 포맷
    #     formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions.items()]
    #     result_dict = {pred['id']: pred['prediction_text'] for pred in formatted_predictions}

    #     # JSON으로 저장
    #     if not os.path.exists("predict_result"):
    #         os.makedirs("predict_result")
    #         print("Created 'predict_result' directory.")

    #     output_file = f"predict_result/Extraction_mrc_ensemble_output.json"
    #     with open(output_file, 'w', encoding='utf-8') as json_file:
    #         json.dump(result_dict, json_file, ensure_ascii=False, indent=4)
    #     print(f"Results saved to {output_file}.")

    def ensemble_inference(self, test_dataset, retrieved_dataset=None, model_paths=None):
        if model_paths is None:
            raise ValueError("You must provide a list of model paths for ensembling.")

        # test_dataset 처리
        if isinstance(test_dataset, DatasetDict):
            test_dataset = test_dataset  # 원하는 분할 이름으로 변경 가능
            print("Extracted 'validation' split from test_dataset.")
        elif isinstance(test_dataset, Dataset):
            print("Using Dataset object for test_dataset.")
        else:
            raise TypeError(f"test_dataset is of unexpected type: {type(test_dataset)}")

        # retrieved_dataset 처리
        if retrieved_dataset is not None:
            if isinstance(retrieved_dataset, DatasetDict):
                retrieved_dataset = retrieved_dataset  # 원하는 분할 이름으로 변경 가능
                print("Extracted 'validation' split from retrieved_dataset.")
            elif isinstance(retrieved_dataset, Dataset):
                print("Using Dataset object for retrieved_dataset.")
            else:
                raise TypeError(f"retrieved_dataset is of unexpected type: {type(retrieved_dataset)}")

            # retrieved context 추가
            test_dataset = self.add_retrieved_contexts(test_dataset, retrieved_dataset)
            print("Added retrieved contexts.")

        # MRC 데이터셋 준비
        test_dataset_prepared, test_examples = self.datas.get_mrc_test_dataset(test_dataset)

        all_start_logits = []
        all_end_logits = []

        for model_path in model_paths:
            print(f"Loading model from {model_path} for inference")
            model = transformers.AutoModelForQuestionAnswering.from_pretrained(model_path, trust_remote_code=True)
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

            # Inference를 위한 TrainingArguments 생성
            inference_args = TrainingArguments(
                output_dir=os.path.join(model_path, "inference"),
                per_device_eval_batch_size=self.args.per_device_eval_batch_size,
                do_predict=True,
                disable_tqdm=True,  
                remove_unused_columns=False,
            )

            # Trainer 준비
            trainer = QuestionAnsweringTrainer(
                model=model,
                args=inference_args,
                tokenizer=tokenizer,
                data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if getattr(inference_args, 'fp16', False) else None),
                post_process_function=self.post_processing_function,
                compute_metrics=self.compute_metrics,
            )

            # 예측 수행
            predictions = trainer.predict(test_dataset=test_dataset_prepared, test_examples=test_examples)
            start_logits, end_logits = predictions.predictions
            all_start_logits.append(start_logits)
            all_end_logits.append(end_logits)

            # 메모리 확보
            del model
            del trainer
            torch.cuda.empty_cache()

        # 모든 모델의 로짓 평균 계산
        avg_start_logits = np.mean(all_start_logits, axis=0)
        avg_end_logits = np.mean(all_end_logits, axis=0)

        # 최종 예측 생성 (답변)
        final_predictions = postprocess_qa_predictions(
            examples=test_examples,
            features=test_dataset_prepared,
            predictions=(avg_start_logits, avg_end_logits),
            version_2_with_negative=False,
            n_best_size=self.args.n_best_size,
            max_answer_length=self.args.max_answer_length,
            null_score_diff_threshold=0.0,
            output_dir=self.output_dir,
            is_world_process_zero=True,
        )

        # 예측 결과를 id: answer 형식으로 저장
        self.extraction_mrc_results = pd.DataFrame.from_dict(final_predictions, orient='index', columns=['prediction_text'])
        result_dict = self.extraction_mrc_results['prediction_text'].to_dict()

        # 결과를 JSON 파일로 저장
        if not os.path.exists("predict_result"):
            os.makedirs("predict_result")
            print("Created 'predict_result' directory.")

        output_file = f"predict_result/Extraction_mrc_ensemble_output_{self.args.model_name.split('/')[-1]}.json"
        with open(output_file, 'w', encoding='utf-8') as json_file:
            json.dump(result_dict, json_file, ensure_ascii=False, indent=4)
        print(f"Results saved to {output_file}.")


