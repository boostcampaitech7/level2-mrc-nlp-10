import transformers
from transformers import default_data_collator, TrainingArguments, EvalPrediction, Seq2SeqTrainingArguments, Seq2SeqTrainer
from trainer_qa import QuestionAnsweringTrainer
from utils_qa import postprocess_qa_predictions
from arguments import Extraction_based_MRC_arguments, Generation_based_MRC_arguments
from All_dataset import prepare_dataset
from transformers import DataCollatorForSeq2Seq
import wandb
from datasets import load_metric
import os
import numpy as np
import nltk
import pandas as pd
nltk.download('punkt')
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import json

class Extraction_based_MRC:
    def __init__(self):
        self.args = Extraction_based_MRC_arguments()
        self.config = transformers.AutoConfig.from_pretrained(self.args.model_name)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.args.model_name)
        self.model = transformers.AutoModelForQuestionAnswering.from_pretrained(self.args.model_name,
                                                            config = self.config)
        self.datas = prepare_dataset(self.args)
        self.datasets = self.datas.get_pure_dataset()
        self.train_dataset = self.datas.get_mrc_train_dataset()
        self.eval_dataset = self.datas.get_mrc_eval_dataset()
        self.metric = load_metric("squad")
        self.trainer = None

        if self.args.use_wandb:
            self.start_wandb()

    def post_processing_function(self, examples, features, predictions, training_args):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples = examples,
            features = features,
            predictions = predictions,
            version_2_with_negative = False,
            n_best_size = self.args.n_best_size,
            max_answer_length = self.args.max_answer_length,
            null_score_diff_threshold = 0.0,
            output_dir = self.args.output_dir,
            is_world_process_zero = self.trainer.is_world_process_zero(),
        )
        
        # Format the result to the format the metric expects.
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in self.datasets["validation"]]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)
    
    def compute_metrics(self, p: EvalPrediction):
        return self.metric.compute(predictions = p.predictions, references = p.label_ids)
    
    def train(self):
        self.training_args = TrainingArguments(
            output_dir= self.args.output_dir,
            do_train = True, 
            do_eval = True, 
            learning_rate = 3e-5,
            per_device_train_batch_size = self.args.per_device_train_batch_size,
            per_device_eval_batch_size = self.args.per_device_train_batch_size,
            num_train_epochs = self.args.num_train_epochs,
            weight_decay = 0.01,
            evaluation_strategy = "epoch"
        )
        self.trainer = QuestionAnsweringTrainer(
                model = self.model,
                args = self.training_args,
                train_dataset = self.train_dataset,
                eval_dataset = self.eval_dataset,
                eval_examples = self.datasets["validation"],
                tokenizer = self.tokenizer,
                data_collator = default_data_collator,
                post_process_function = self.post_processing_function,
                compute_metrics = self.compute_metrics,
            )

        if self.args.use_wandb:
            self.training_args.report_to = ["wandb"]
            self.training_args.run_name = "default"

        self.trainer.train()
        self.trainer.evaluate()

    def start_wandb(self):
        os.system("rm -rf /root/.cache/wandb")
        os.system("rm -rf /root/.config/wandb")
        os.system("rm -rf /root/.netrc")
        
        # WandB API 키 설정 (여러분의 API 키를 넣어주시면 됩니다)
        os.environ["WANDB_API_KEY"] = self.args.wandb_key
        wandb.init(project='Dense_embedding_retrieval')

    def inference(self):
        eval_dataset, eval_examples = self.datas.get_mrc_test_dataset()

        if not self.trainer:
            self.training_args = TrainingArguments(
            output_dir= self.args.output_dir,
            do_train = True, 
            do_eval = True, 
            learning_rate = 3e-5,
            per_device_train_batch_size = self.args.per_device_train_batch_size,
            per_device_eval_batch_size = self.args.per_device_train_batch_size,
            num_train_epochs = self.args.num_train_epochs,
            weight_decay = 0.01,
            evaluation_strategy = "epoch"
        )

        self.trainer = QuestionAnsweringTrainer(
            model = self.model,
            args = self.training_args,
            train_dataset = self.train_dataset,
            eval_dataset = self.eval_dataset,
            eval_examples = self.datasets["validation"],
            tokenizer = self.tokenizer,
            data_collator = default_data_collator,
            post_process_function = self.post_processing_function,
            compute_metrics = self.compute_metrics,
            )
        
        predictions = self.trainer.predict(
            test_dataset = eval_dataset,
            test_examples = eval_examples)        
        
        print('output을 id : answers 형태의 json파일로 내보냅니다. 결과는 model.extraction_mrc_results로 확인할 수 있습니다.')
        self.extraction_mrc_results = pd.DataFrame({'id' : eval_examples['id'], 
                                                    'answers' : [item['prediction_text'] for item in predictions.predictions]})
        result_dict = self.extraction_mrc_results.set_index('id')['answers'].to_dict()

        with open('Extraction_mrc_output.json', 'w', encoding='utf-8') as json_file:
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
        # decoded_labels is for rouge metric, not used for f1/em metric

        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
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
            self.model
        )
        training_args = Seq2SeqTrainingArguments(
            output_dir = './generative_MRC_outputs',
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
            remove_unused_columns = True)

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

    def inference(self):
        test_dataset = self.datas.get_generative_MRC_test_dataset()
        inputs = [f'question: {test_dataset.iloc[i]["question"]}  context: {test_dataset.iloc[i]["context"]} </s>' for i in range(len(test_dataset))]
        
        sample = self.tokenizer(inputs, 
                                max_length = self.args.max_source_length, 
                                padding = self.args.padding, 
                                truncation = True, 
                                return_tensors = 'pt')

        # sample = {key: value.to("cuda:0") for key, value in sample.items()}
        self.model = self.model.to('cpu') # CUDA OOM이 떠서 CPU에서 작업하도록 바꿉니다 ㅠ
        outputs = self.model.generate(**sample, 
                                    max_length=self.args.max_target_length, 
                                    num_beams=self.args.num_beams)

        preds = [self.tokenizer.decode(output, skip_special_tokens=True) for output in tqdm(outputs, desc = "Decoding")]

        print('output을 id : answers 형태의 json파일로 변환합니다. 결과는 model.generative_mrc_results로 확인할 수 있습니다.')
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in tqdm(preds, desc = "Sentence Tokenization")]
        self.generative_mrc_results = pd.DataFrame({'id' : test_dataset['id'], 'answers' : preds})
        result_dict = self.generative_mrc_results.set_index('id')['answers'].to_dict()
        with open('Extraction_mrc_output.json', 'w', encoding='utf-8') as json_file:
            json.dump(result_dict, json_file, ensure_ascii=False, indent=4)