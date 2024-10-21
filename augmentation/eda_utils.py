import os
import pandas as pd
from datasets import Dataset

# CSV -> JSON 변환
def csv_to_json(csv_file_path, json_file_path):
    df = pd.read_csv(csv_file_path)
    df.to_json(json_file_path, orient='records', lines=True, force_ascii=False)

# JSON 데이터를 Hugging Face Dataset 형식으로 변환하고 저장
def create_augmentation_dataset(json_file_path, output_dir):
    # JSON 파일을 DataFrame으로 읽기
    df = pd.read_json(json_file_path, lines=True)
    
    # 인덱스 제거
    df.reset_index(drop=True, inplace=True)
    
    # Hugging Face Dataset으로 변환
    dataset = Dataset.from_pandas(df, preserve_index=False)
    
    # 디렉토리가 없으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Dataset 저장
    dataset.save_to_disk(output_dir)
    print(f"Dataset has been saved to {output_dir}")