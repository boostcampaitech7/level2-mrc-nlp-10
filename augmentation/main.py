from eda import eda
from eda_utils import csv_to_json, create_augmentation_dataset
import pandas as pd

# 데이터 증강 및 변환 실행
if __name__ == "__main__":
    csv_file_path = 'path_to_csv_file.csv'  # 변환할 CSV 파일 경로
    json_file_path = 'path_to_json_file.json'  # 생성할 JSON 파일 경로
    output_dir = 'path_to_output_directory'  # Hugging Face Dataset을 저장할 디렉토리

    # CSV -> JSON 변환
    csv_to_json(csv_file_path, json_file_path)
    
    # 증강 작업 수행
    df = pd.read_json(json_file_path, lines=True)
    augmented_df = eda(df)

    # 증강된 데이터를 JSON으로 저장
    augmented_df.to_json(json_file_path, orient='records', lines=True, force_ascii=False)

    # JSON 데이터를 Dataset으로 변환하고 저장
    create_augmentation_dataset(json_file_path, output_dir)