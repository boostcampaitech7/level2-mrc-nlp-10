# Open-Domain Question Answering
본 프로젝트는 MRC(기계독해)란 주어진 지문을 이해하고, 주어진 질의의 답변을 추론하는 태스크입니다.

<br><br>

## 1. Overview

### 🚩 Open-Domain Question Answering(ODQA) Project
`Context` 에서 `Question`에 대한 `Answer`을 찾는 Project입니다.
ODQA는 2 Step을 거쳐 답을 도출하게 됩니다.

<br>

**ODQA 2Steps**
1. `MRC` : `Context` 내에서 `Question`에 대한 `Answer`를 도출하는 방법 학습
(*MRC : Machine Reading Comprehension)
2. `Retrieval` : 문서를 검색하고 해당 문서 내에서 정답 탐색 후 도출

<br>

**모델 성능 평가 지표(Metrics)**
- EM (Exact Match) : 정답과 예측이 정확히 일치하는 샘플의 비율
- F1 score : 기준 값과 overlap 되는 정도를 F1으로 계산

<br><br>

## 2. 프로젝트 구성

### ⏰ 개발 기간
- 2024년 09월 30일(월) 10:00 ~ 2024년 10월 24일(목) 19:00
- 부스트캠프 AI Tech NLP 트랙 8-10주차
  
  |Title|Period|Days|Contents|
  |:---:|:---:|:---:|:---:|
  |필요 지식 학습|09.30 ~ 10.04|5 days|ODQA Task의 원리와 구조에 대해 이해|
  |데이터 분석 및 EDA|10.05 ~ 10.07|3 days|데이터 구조에 대한 이해|
  |Base Model 기반 구현|10.05 ~ 10.10|6 days|Base 모델 기반 Templet 구성|
  |Model 성능 테스트 및 고도화|10.11 ~ 10.24|13 days|실험 및 성능 개선|

<br>

### ✨ 분석 환경
- Upstage AI Stages 제공 V100 GPU Server 활용
- OS : Linux
- Language : Python
- Libraries(mainly used) : Pytorch, Hugging Face, Wandb etc.

<br>

### 📚 EDA Summary
**1. 결측치 및 중복값**
   - Context, Question, Answer, Answer_start 항목 중 결측치는 존재하지 않았음
   - 문서와 답변에서 중복 값이 있으나, Question과 Answer pair 측면에서 봤을 때 중복이라고 할 수 없음
     
**2. 단순 통계분석**
   - 문장의 길이가 길다고 해서 Token 수도 비례해서 많지 않을 수 있음을 확인 (의미 없는 단어의 나열일 가능성 있음)
     
**3. 단어 포함 여부 분석**
   - Extraction-based MRC를 EM 기준으로 수행하기 위해서는 Context 내 Answer가 포함되어 있어야한다는 생각을 기반으로 수행
   - 최소 1회, Train 데이터셋에는 평균 1.8회, Validation 데이터셋에서는 평균 1.7회 정답 단어가 포함되어 있음을 확인
     
**4. 언어 분석**
   
   ![언어별비율](https://github.com/user-attachments/assets/96f384e3-8c81-4dcf-8756-b548e9b6c6e5)
   <br>
   - 한글 외에도 다양한 언어가 섞여 있는 것을 확인할 수 있음
   - Train, Validation Context의 한글 차지 비율은 60~80%로 구성
   - Test 데이터에서 한글 비율이 30% 이하인 질문은 없으며, 600개의 질문 중 12개를 제외 하고 질문의 60% 이상이 한글로 구성되어 있음
   - 일부 문제는 번역된 문서에서 찾아와야 할 것으로 보이나, EM 방식에서도 가능한지는 논의가 필요해보임
     (ex. 디엔비엔푸 전투에서 보응우옌잡이 상대한 국가는?)
   - Text에서 일부 한글 비율이 낮은 데이터를 제거하는 것이 모델 성능에 어떠한 영향을 미치는지 실험해 볼 필요가 있다고 판단 됨


<br>

### 💾 Data Structure
<img src="https://github.com/user-attachments/assets/9d7008c1-8f0c-478f-9c83-c1731387e8c4" alt="데이터모식도" width="45%">


<br><br>


### 💡 구현 기능
- **MRC**
    - Extraction based MRC
    - Generation based MRC
- **Retrieval**
    - TF IDF Retrieval
    - BM25 Retrieval
    - Dense search Retrieval

<br>

> ⚙ How To Use
> 자세한 내용은 `Templet` > `README.md` 참고
> - MRC, Retrieval Arguments : `Templet` > `arguments.py`
> - Train, Validation, Test : `Templet` > `howtouse_MRC.ipynb`


<br><br>

## 3. 프로젝트 결과
|Idx  |Public EM|Public F1|Private EM|Private F1|MRC|Embedding|K-fold|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|1|60.83|72.02|59.72|71.47|klue/roberta-large <br>(uown97/klue-roberta-finetuned-korquad-v2)|Cross|X|
|2|60.42|73.37|59.72|70.00|klue/roberta-large <br>(uown97/klue-roberta-finetuned-korquad-v2)|Cross|3|

- 원본 데이터셋과 Fine-tuning된 Klue/Roberta-Large 모델 조합에서 가장 EM이 높게 나옴
- 문장의 특징을 추출하는 역할을 하는 Pre-trained Language Model(PLM)을 적용했을 때 이전 실험 대비 우수한 성과를 얻을 수 있었음
- K-fold를 했을때보다 안했을 때 조금 더 성능이 좋게 나옴


<br><br>


## 4. Team
<table>
    <tbody>
        <tr>
            <td align="center">
                <a href="https://github.com/Kimyongari">
                    <img src="https://github.com/Kimyongari.png" width="100px;" alt=""/><br />
                    <sub><b>Kimyongari</b></sub>
                </a><br />
                <sub>김용준</sub>
            </td>
            <td align="center">
                <a href="https://github.com/son0179">
                    <img src="https://github.com/son0179.png" width="100px;" alt=""/><br />
                    <sub><b>son0179</b></sub>
                </a><br />
                <sub>손익준</sub>
            </td>
            <td align="center">
                <a href="https://github.com/P-oong">
                    <img src="https://github.com/P-oong.png" width="100px;" alt=""/><br />
                    <sub><b>P-oong</b></sub>
                </a><br />
                <sub>이현풍</sub>
            </td>
            <td align="center">
                <a href="https://github.com/Aitoast">
                    <img src="https://github.com/Aitoast.png" width="100px;" alt=""/><br />
                    <sub><b>Aitoast</b></sub>
                </a><br />
                <sub>정석현</sub>
            </td>
            <td align="center">
                <a href="https://github.com/uzlnee">
                    <img src="https://github.com/uzlnee.png" width="100px;" alt=""/><br />
                    <sub><b>uzlnee</b></sub>
                </a><br />
                <sub>정유진</sub>
            </td>
            <td align="center">
                <a href="https://github.com/hayoung180">
                    <img src="https://github.com/hayoung180.png" width="100px;" alt=""/><br />
                    <sub><b>hayoung180</b></sub>
                </a><br />
                <sub>정하영</sub>
            </td>
        </tr>
    </tbody>
</table>

<br><br>

---

<br>

## Reference
![My Image](https://upload3.inven.co.kr/upload/2021/03/13/bbs/i013687510742.gif)
1. V. Karpukhin, B. Oğuz, S. Min, P. Lewis, L. Wu, S. Edunov, D. Chen, and W.-t. Yih, "Dense Passage Retrieval for Open-Domain Question Answering," *arXiv*, vol. 2004.04906, Sep. 2020. [Online]. Available: https://doi.org/10.48550/arXiv.2004.04906
2. A. Srivastava and A. Memon, "Toward Robust Evaluation: A Comprehensive Taxonomy of Datasets and Metrics for Open Domain Question Answering in the Era of Large Language Models," *IEEE Access*, vol. 12, pp. 117483-117503, 2024, doi: 10.1109/ACCESS.2024.3446854.
3. klue/bert-base (https://huggingface.co/klue/bert-base)
3-1. yjgwak/klue-bert-base-finetuned-squad-kor-v1 (https://huggingface.co/yjgwak/klue-bert-base-finetuned-squad-kor-v1)
4. klue/roberta-small (https://huggingface.co/klue/roberta-small)
5. klue/roberta-large (https://huggingface.co/klue/roberta-large)
5-1. FacebookAI/roberta-large (https://huggingface.co/FacebookAI/roberta-large)
5-2. uomnf97/klue-roberta-finetuned-korquad-v2 (https://huggingface.co/uomnf97/klue-roberta-finetuned-korquad-v2)
6. timpal0l/mdeberta-v3-base-squad2 (https://huggingface.co/timpal0l/mdeberta-v3-base-squad2)
7. monologg/kobigbird-bert-base (https://huggingface.co/monologg/kobigbird-bert-base)
8. monologg/koelectra-base-v3-finetuned-korquad (https://huggingface.co/monologg/koelectra-base-v3-finetuned-korquad)
9. monologg/kobert (https://huggingface.co/monologg/kobert)
10. N. Thakur, N. Reimers, A. Rücklé, A. Srivastava, and I. Gurevych, "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models," *arXiv*, vol. 2104.08663, Oct. 2021. [Online]. Available: https://doi.org/10.48550/arXiv.2104.08663
11. J. Wei and K. Zou, "EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks," *arXiv*, vol. 1901.11196, Aug. 2019. [Online]. Available: https://doi.org/10.48550/arXiv.1901.11196

