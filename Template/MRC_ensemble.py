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
import pandas as pd
from tqdm import tqdm

nltk.download('punkt')
warnings.filterwarnings('ignore')

def set_seed(seed: int = 42):
    import random
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
        self.model.gradient_checkpointing_enable()
        
        self.datas = prepare_dataset(self.args)
        self.datasets = self.datas.get_pure_dataset()
        self.metric = load_metric("squad")
        self.trainer = None
        self.output_dir = os.path.join(self.args.model_path, self.args.model_name.split('/')[-1])

        if self.args.use_wandb:
            self.start_wandb()

    def start_wandb(self):
        # WandB 설정
        os.environ["WANDB_API_KEY"] = self.args.wandb_key
        wandb.init(project='Dense_embedding_retrieval')
            

    def load_model(self):
        # 모델 체크포인트 로드
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
        print(f'Latest checkpoint: {trainer_state_file}')
        
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

        # 경량화 체크포인트 사용
        self.model.gradient_checkpointing_enable()

        self.trainer = QuestionAnsweringTrainer(
            model=self.model,
            args=self.training_args,
            tokenizer=self.tokenizer,
            post_process_function=self.post_processing_function,
            compute_metrics=self.compute_metrics
        )
        print("Model and Trainer loaded from the best checkpoint.")
        

    # def post_processing_function(self, examples, features, predictions, training_args):
    #     # self.trainer가 None인 경우 기본값으로 True 설정
    #     is_wzp = self.trainer.is_world_process_zero() if self.trainer else True
        
    #     predictions = postprocess_qa_predictions(
    #         examples=examples,
    #         features=features,
    #         predictions=predictions,
    #         version_2_with_negative=False,
    #         n_best_size=self.args.n_best_size,
    #         max_answer_length=self.args.max_answer_length,
    #         null_score_diff_threshold=0.0,
    #         output_dir=self.output_dir,
    #         is_world_process_zero=is_wzp,
    #     )

    #     # 'answers' 필드가 있는지 확인
    #     if 'answers' in examples.column_names:
    #         formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
    #         references = [{"id": ex["id"], "answers": ex['answers']} for ex in examples]
    #         return EvalPrediction(predictions=formatted_predictions, label_ids=references)
    #     else:
    #         # 'answers' 필드가 없는 경우, 단순히 예측만 반환
    #         return predictions
    
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
        print(examples)
        if training_args.do_predict:
            return formatted_predictions
        elif training_args.do_eval:
            references = [
                {"id": ex["id"], "answers": ex['answers']}
                for ex in examples
            ]

            return EvalPrediction(
                predictions=formatted_predictions, label_ids=references
            )


    def compute_metrics(self, p: EvalPrediction):
        if p.label_ids is not None:
            return self.metric.compute(predictions=p.predictions, references=p.label_ids)
        else:
            return {}
    
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

    # 앙상블 학습 메서드
    def train_multiple_models(self, model_names, train_dataset=None, eval_dataset=None):
        """
        여러 모델을 순차적으로 학습하고, 개별 모델을 저장하는 함수입니다.
        """
        for model_name in model_names:
            print(f"Training model: {model_name}")
            set_seed(42)
            # 모델별로 설정 업데이트
            self.config = transformers.AutoConfig.from_pretrained(model_name)
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            self.model = transformers.AutoModelForQuestionAnswering.from_pretrained(model_name, config=self.config, trust_remote_code=True)

            # RoBERTa 기반 모델일 경우 token_type_ids 사용하지 않도록 설정
            is_roberta = "roberta" in model_name.lower()
            self.datas.tokenizer = self.tokenizer
            self.datas.include_token_type_ids = not is_roberta

            # 모델별로 출력 경로 설정
            model_dir_name = model_name.replace('/', '_')
            self.output_dir = os.path.join(self.args.model_path, model_dir_name)

            # 데이터셋 준비
            if train_dataset is None:
                train_dataset_current = self.datasets['train']
            else:
                train_dataset_current = train_dataset

            if eval_dataset is None:
                eval_dataset_current = self.datasets['validation']
            else:
                eval_dataset_current = eval_dataset
                
            eval_examples = eval_dataset_current

            train_dataset_prepared = self.datas.get_mrc_train_dataset(train_dataset_current)
            eval_dataset_prepared = self.datas.get_mrc_eval_dataset(eval_dataset_current)
            
            self.get_trainer(
                train_dataset=train_dataset_prepared,
                eval_dataset=eval_dataset_prepared,
                eval_examples=eval_examples
            )
            
            # 학습 시작
            self.trainer.train()
            self.trainer.save_model(self.output_dir)
            print(f"Model {model_name} saved to {self.output_dir}")
            

    def ensemble_inference(self, test_dataset, model_paths=None):
        # Check for model paths
        if model_paths is None:
            raise ValueError("You must provide a list of model paths for ensembling.")

        # Ensure test_dataset is in the right format
        if isinstance(test_dataset, DatasetDict):
            test_dataset = test_dataset['validation']  # Extract 'validation' split if it's a DatasetDict
            print("Using 'validation' split from test_dataset.")
        elif isinstance(test_dataset, Dataset):
            print("Using Dataset object for test_dataset.")
        else:
            raise TypeError(f"test_dataset is of unexpected type: {type(test_dataset)}")

        # Check for None values in the dataset and ensure fields are strings
        test_dataset = test_dataset.filter(
            lambda x: x['context'] is not None 
                    and x['question'] is not None
                    and isinstance(x['context'], str)
                    and isinstance(x['question'], str)
        )
        print("Filtered dataset to remove None and non-string entries.")

        # The test_examples are directly taken from the test_dataset
        test_examples = test_dataset  # 'test_dataset' contains context, id, and question

        all_start_logits = []
        all_end_logits = []

        # Loop through each model for ensemble inference
        for model_path in model_paths:
            print(f"Loading model from {model_path} for inference")

            # 모델 로드 디버깅용 
            try:
                model = transformers.AutoModelForQuestionAnswering.from_pretrained(model_path, trust_remote_code=True)
                tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
            except Exception as e:
                print(f"Failed to load model from {model_path}. Error: {e}")
                continue

            test_dataset_prepared = test_dataset.map(
                lambda x: tokenizer(
                    x['question'], x['context'],
                    padding="max_length",
                    truncation=True,
                    max_length=self.args.max_seq_length,
                    return_offsets_mapping=True  # 추가
                ),
                batched=True
            )

            # Verify tokenization
            try:
                print("Sample tokenized example:", test_dataset_prepared[0])
            except Exception as e:
                print(f"Error printing tokenized example: {e}")
                continue

            # Define inference arguments
            inference_args = TrainingArguments(
                output_dir=os.path.join(model_path, "inference"),
                per_device_eval_batch_size=self.args.per_device_eval_batch_size,
                do_predict=True,
                disable_tqdm=False,
                remove_unused_columns=False,
            )

            # Set up the trainer for inference
            trainer = QuestionAnsweringTrainer(
                model=model,
                args=inference_args,
                tokenizer=tokenizer,
                data_collator = DataCollatorWithPadding(
                    self.tokenizer,
                    pad_to_multiple_of=8 if self.training_args.fp16 else None
                ),
                post_process_function=self.post_processing_function,
                compute_metrics=self.compute_metrics,
            )

            # Perform inference and collect the start and end logits
            try:
                predictions = trainer.predict(test_dataset=test_dataset_prepared, test_examples=test_examples)
                start_logits, end_logits = predictions.predictions
                all_start_logits.append(start_logits)
                all_end_logits.append(end_logits)
            except Exception as e:
                print(f"Error during prediction with model {model_path}: {e}")
                continue

            # Release memory used by the model and trainer
            del model
            del trainer
            torch.cuda.empty_cache()

        if not all_start_logits or not all_end_logits:
            raise RuntimeError("No logits were collected. Check if models are loaded correctly.")

        # Average the logits across all models for ensemble inference
        avg_start_logits = np.mean(all_start_logits, axis=0)
        avg_end_logits = np.mean(all_end_logits, axis=0)

        # Post-process the predictions using the averaged logits
        final_predictions = postprocess_qa_predictions(
            examples=test_examples,
            features=test_dataset_prepared,  # Process without offset_mapping
            predictions=(avg_start_logits, avg_end_logits),
            version_2_with_negative=False,
            n_best_size=self.args.n_best_size,
            max_answer_length=self.args.max_answer_length,
            null_score_diff_threshold=0.0,
            output_dir=self.output_dir,
            is_world_process_zero=True,
        )

        # Convert predictions to DataFrame and save results
        self.extraction_mrc_results = pd.DataFrame.from_dict(final_predictions, orient='index', columns=['prediction_text'])
        result_dict = self.extraction_mrc_results['prediction_text'].to_dict()

        # Save the results as a JSON file
        if not os.path.exists("predict_result"):
            os.makedirs("predict_result")
            print("Created 'predict_result' directory.")

        output_file = f"predict_result/Extraction_mrc_ensemble_output_{self.args.model_name.split('/')[-1]}.json"
        with open(output_file, 'w', encoding='utf-8') as json_file:
            json.dump(result_dict, json_file, ensure_ascii=False, indent=4)
        print(f"Results saved to {output_file}.")
