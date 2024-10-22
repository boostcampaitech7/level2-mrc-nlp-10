import transformers
from transformers import default_data_collator, TrainingArguments, EvalPrediction, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorWithPadding
from trainer_qa import QuestionAnsweringTrainer
from utils_qa import postprocess_qa_predictions
from arguments import Extraction_based_MRC_arguments, Generation_based_MRC_arguments
from All_dataset import prepare_dataset
from transformers import DataCollatorForSeq2Seq
import wandb
from datasets import load_metric, concatenate_datasets, Dataset
import os
from glob import glob
import numpy as np
import nltk
import pandas as pd
import torch
from sklearn.model_selection import KFold
nltk.download('punkt')
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import json

class CustomDataCollatorWithPadding(DataCollatorWithPadding):
    def __call__(self, features):
        # 'offset_mapping'과 'overflow_to_sample_mapping' 필드를 제거
        features = [
            {k: v for k, v in feature.items() if k not in ['offset_mapping', 'overflow_to_sample_mapping']}
            for feature in features
        ]
        return super().__call__(features)
    
    

class Extraction_based_MRC:
    def __init__(self):
        self.args = Extraction_based_MRC_arguments()
        self.config = transformers.AutoConfig.from_pretrained(self.args.model_name)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.args.model_name)
        self.model = transformers.AutoModelForQuestionAnswering.from_pretrained(self.args.model_name,
                                                            config = self.config,
                                                             trust_remote_code = True)
        self.datas = prepare_dataset(self.args)
        self.datasets = self.datas.get_pure_dataset()
        self.metric = load_metric("squad")
        self.trainer = None
        self.output_dir = self.args.model_path + '_' + self.args.model_name.split('/')[-1] # 모델 이름을 바탕으로 저장 경로가 생성됩니다.
        if self.args.use_wandb:
            self.start_wandb()

    def load_model(self):
        
        self.training_args = TrainingArguments(
            output_dir = self.output_dir,
            do_train = False,
            do_eval = True,
            per_device_eval_batch_size = self.args.per_device_eval_batch_size,  
            fp16 = True
        )
        model_path = self.output_dir
        print(model_path)
        
        checkpoints = sorted(glob(model_path + '/checkpoint-*'), key=lambda x: int(x.split('-')[-1]), reverse=True)
        lastcheckpt = checkpoints[0]
     
        trainer_state_file = os.path.join(lastcheckpt, 'trainer_state.json')
        print('제일 마지막 checkpoint :',trainer_state_file)
        if os.path.exists(trainer_state_file):
            with open(trainer_state_file, 'r') as f:
                trainer_state = json.load(f)
                best_checkpoint = trainer_state.get('best_model_checkpoint', None)
                print('best checkpoint :', best_checkpoint)
 
        
        # best model checkpoint 경로 가져오기
        best_checkpoint = trainer_state.get('best_model_checkpoint', None)
        self.config = transformers.AutoConfig.from_pretrained(best_checkpoint)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(best_checkpoint)
        self.model = transformers.AutoModelForQuestionAnswering.from_pretrained(best_checkpoint, config=self.config)
        # Trainer 인스턴스 생성
        self.trainer = QuestionAnsweringTrainer(
            model = self.model,
            args = self.training_args,
            post_process_function = self.post_processing_function,
            compute_metrics = self.compute_metrics
        )
        print("bestmodel 체크포인트로 모델과 Trainer가 로드되었습니다.")

    def post_processing_function(self, examples, features, predictions, training_args):
        # 모델의 output을 바탕으로 단어를 예측합니다. (model > trainer > post_processing_function > util_qa.py에 있는 postprocess_qa_predictions)
        predictions = postprocess_qa_predictions(
            examples = examples,
            features = features,
            predictions = predictions,
            version_2_with_negative = False,
            n_best_size = self.args.n_best_size,
            max_answer_length = self.args.max_answer_length,
            null_score_diff_threshold = 0.0,
            output_dir = self.output_dir,
            is_world_process_zero = self.trainer.is_world_process_zero(),
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
        return self.metric.compute(predictions = p.predictions, references = p.label_ids)
    
    def get_trainer(self, train_dataset = None, eval_dataset = None):
        self.training_args = TrainingArguments(
            output_dir = self.output_dir,
            learning_rate = 3e-5,
            per_device_train_batch_size = self.args.per_device_train_batch_size,
            per_device_eval_batch_size = self.args.per_device_train_batch_size,
            num_train_epochs = self.args.num_train_epochs,
            weight_decay = 0.01,
            evaluation_strategy = "epoch",  # 1 epoch마다 evaluation 수행
            save_strategy = "epoch",  # 1 epoch마다 모델 저장
            report_to = ["wandb"],  # wandb에 보고하도록 설정
            run_name = f"Extraction_MRC_{self.args.model_name}",  # wandb run name 설정
            logging_dir='./logs',  # 로그 저장 경로
            logging_strategy = "epoch",
            load_best_model_at_end = True,
            fp16 = True
        )

        data_collator = DataCollatorWithPadding(
            self.tokenizer, pad_to_multiple_of=8 if self.training_args.fp16 else None
        )

        self.trainer = QuestionAnsweringTrainer(
                model = self.model,
                args = self.training_args,
                train_dataset = train_dataset,
                eval_dataset = eval_dataset,
                eval_examples = self.datasets["validation"],
                tokenizer = self.tokenizer,
                data_collator = data_collator,
                post_process_function = self.post_processing_function,
                compute_metrics = self.compute_metrics,
            )

        if self.args.use_wandb:
            self.training_args.report_to = ["wandb"]
            self.training_args.run_name = "default"
            
    def train(self, train_dataset = None, eval_dataset = None):
        self.get_trainer()
        if train_dataset == None:
            train_dataset = self.datas.get_mrc_train_dataset()
            print('train_dataset을 넣지 않아 기존에 주어진 train dataset으로 학습합니다.')
        else:
            train_dataset = self.datas.get_mrc_train_dataset(train_dataset)

        if eval_dataset == None:
            eval_dataset = self.datas.get_mrc_eval_dataset() 
            print('eval_dataset을 넣지 않아 기존에 주어진 eval_dataset으로 평가합니다.')
        else:
            eval_dataset = self.datas.get_mrc_eval_dataset(eval_dataset)

        self.trainer.train_dataset = train_dataset
        self.trainer.eval_dataset = eval_dataset
        self.trainer.train()
        
        torch.cuda.empty_cache()
        

    def kfold_train(self, train_dataset = None, eval_dataset = None):
        output_dir = self.output_dir + '_kfold'
        self.training_args.output_dir = output_dir


        if train_dataset == None and eval_dataset == None:
            print('train_dataset을 넣지 않아 기존에 주어진 train dataset으로 학습합니다.')
            print('eval_dataset을 넣지 않아 기존에 주어진 eval_dataset으로 평가합니다.')
            train_dataset, eval_dataset = self.datasets['train'], self.datasets['validation']
            concat_data = concatenate_datasets([train_dataset, eval_dataset])
        else:
            concat_data = concatenate_datasets([train_dataset, eval_dataset])

        kf = KFold(n_splits = self.args.kfold, shuffle = True, random_state = 42,)
        self.get_trainer()
        self.training_args.num_train_epochs = self.args.epoch_for_kfold

        for fold, (train_idx, eval_idx) in enumerate(kf.split(concat_data)):
            print(f'--------------{fold+1} fold ----------------')
            train = Dataset.from_dict(concat_data[train_idx])
            eval = Dataset.from_dict(concat_data[eval_idx])
            eval_examples = eval
            self.trainer.train_dataset = self.datas.get_mrc_train_dataset(train)
            self.trainer.eval_dataset = self.datas.get_mrc_eval_dataset(eval)
            self.trainer.eval_examples = eval_examples
            print(eval_examples)
            print(self.datas.get_mrc_eval_dataset(eval))
            self.model.resize_token_embeddings(len(self.tokenizer))

            self.trainer.train()
        
        torch.cuda.empty_cache()

    def start_wandb(self):
        os.system("rm -rf /root/.cache/wandb")
        os.system("rm -rf /root/.config/wandb")
        os.system("rm -rf /root/.netrc")
        
        # WandB API 키 설정 (여러분의 API 키를 넣어주시면 됩니다)
        os.environ["WANDB_API_KEY"] = self.args.wandb_key
        wandb.init(project = 'Dense_embedding_retrieval')

    def inference(self, test_dataset):
        assert self.trainer is not None, "trainer가 없습니다. 로드하거나 train하세요."
        test_dataset, test_examples = self.datas.get_mrc_test_dataset(test_dataset)
        self.training_args.do_predict = True
        predictions = self.trainer.predict(
            test_dataset = test_dataset,
            test_examples = test_examples)        

        print('output을 id : answers 형태의 json파일로 내보냅니다. 결과는 model.extraction_mrc_results로 확인할 수 있습니다.')
        self.extraction_mrc_results = pd.DataFrame(predictions)
        result_dict = self.extraction_mrc_results.set_index('id')['prediction_text'].to_dict()
        if not os.path.exists("predict_result"):
            os.makedirs("predict_result")
            print("폴더 'predict_result'가 생성되었습니다.")

        with open(f"predict_result/Extraction_mrc_output_{self.args.model_name.split('/')[-1]}.json", 'w', encoding='utf-8') as json_file:
            json.dump(result_dict, json_file, ensure_ascii=False, indent=4)

class Generation_based_MRC:
    def __init__(self):
        self.args = Generation_based_MRC_arguments()
        self.config = transformers.AutoConfig.from_pretrained(self.args.model_name)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.args.model_name)
        self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(self.args.model_name,
                                                                        config = self.config,
                                                                        cache_dir = None)
        self.metric = load_metric('squad')
        self.datas = prepare_dataset(self.args)
        self.datasets = self.datas.get_pure_dataset()
        if self.args.use_wandb:
            self.start_wandb()
        self.output_dir = self.args.model_path + self.args.model_name.split('/')[-1] # 모델 이름을 바탕으로 저장 경로가 생성됩니다.

    def load_model(self):
        
        self.training_args = TrainingArguments(
            output_dir = self.output_dir,
            do_train = False,
            do_eval = True,
            per_device_eval_batch_size = self.args.per_device_eval_batch_size,  
        )
        model_path = self.output_dir
        lastcheckpt = glob(model_path + '/checkpoint-*')[-1]
    
        trainer_state_file = os.path.join(lastcheckpt, 'trainer_state.json')
        print('제일 마지막 checkpoint :',trainer_state_file)
        if os.path.exists(trainer_state_file):
            with open(trainer_state_file, 'r') as f:
                trainer_state = json.load(f)
                best_checkpoint = trainer_state.get('best_model_checkpoint', None)
                print('best checkpoint :', best_checkpoint)
        
        # best model checkpoint 경로 가져오기
        best_checkpoint = trainer_state.get('best_model_checkpoint', None)
        self.config = transformers.AutoConfig.from_pretrained(best_checkpoint)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(best_checkpoint)
        self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(best_checkpoint, config=self.config)
        # Trainer 인스턴스 생성
        self.trainer = Seq2SeqTrainer(
            model = self.model,
            args = self.training_args,
            post_process_function = self.post_processing_function,
            compute_metrics = self.compute_metrics
        )
        print("bestmodel 체크포인트로 모델과 Trainer가 로드되었습니다.")

    def postprocess_text(self, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)
        # [14,42,512,41,125,643,-100,-100,-100]
        # > [14,42,512,31,125,643,패딩,패딩,패딩] 
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = self.postprocess_text(decoded_preds, decoded_labels)

        formatted_predictions = [{"id": ex['id'], "prediction_text": decoded_preds[i]} for i, ex in enumerate(self.datasets["validation"])]
        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in self.datasets["validation"]]

        result = self.metric.compute(predictions=formatted_predictions, references=references)
        return result 
    
    def start_wandb(self):
        os.system("rm -rf /root/.cache/wandb")
        os.system("rm -rf /root/.config/wandb")
        os.system("rm -rf /root/.netrc")
        
        # WandB API 키 설정 (여러분의 API 키를 넣어주시면 됩니다)
        os.environ["WANDB_API_KEY"] = self.args.wandb_key
        wandb.init(project='Dense_embedding_retrieval')
    
    def train(self):

        train_dataset = self.datas.get_generative_MRC_train_dataset()
        eval_dataset = self.datas.get_generative_MRC_valid_dataset()
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            self.model,
            label_pad_token_id = self.tokenizer.pad_token_id,
            pad_to_multiple_of = 8,
        )
        training_args = Seq2SeqTrainingArguments(
            output_dir = self.output_dir,
            do_eval=True,
            per_device_train_batch_size = self.args.per_device_train_batch_size,
            per_device_eval_batch_size = self.args.per_device_train_batch_size,
            predict_with_generate = True,
            num_train_epochs = self.args.num_train_epochs,
            save_strategy = 'epoch',
            logging_steps = 30,
            evaluation_strategy = 'epoch',
            save_total_limit = 5, # 체크포인트 몇개 저장할건지 정하기
            logging_strategy = 'epoch',
            load_best_model_at_end = True,
            learning_rate = self.args.learning_rate,
            weight_decay = 0.01,
            remove_unused_columns = True,
            fp16 = True,
            )

        if self.args.use_wandb:
            training_args.report_to = ["wandb"]
            training_args.run_name = "default"

        trainer = Seq2SeqTrainer(
            model = self.model,
            tokenizer = self.tokenizer,
            args = training_args,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            compute_metrics = self.compute_metrics,
            data_collator = data_collator
        )
        trainer.train()

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