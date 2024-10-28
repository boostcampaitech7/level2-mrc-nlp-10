이번 프로젝트 템플릿입니다.

## 쓰는법
### 1. howtouse를 본다
### 2. arguments.py를 드가서 바꿀 부분을 고른다. 
(argument들이 정리도 안돼있고 개판이라 ctrl+f / command+f로 찾아서 바꾸셔야합니다 ㅎㅎ;;)

(Generation_based_MRC_arguments, 
Extraction_based_MRC_arguments,
Dense_search_retrieval_arguments, 
TF_IDF_retrieval_arguments 들을 치시고 찾으시면 금방 나올거에요.)

### 3. 바꿀 부분을 바꾼다.

retrieval을 선언하고 build_faiss()
retrieval_model.search()를 하면 test데이터셋에 대한 retrieval 결과가 나옵니다.
이를 mrcmodel.inference(retrieval_result)에 넣으면 predict_result 폴더에 json 파일이 생성됩니다.


## 이번에 내가 한 것
### 1. Dense retrieval 제작
어째서인지 q_model과 p_model을 따로 선언할경우 결과가 좋지않습니다. (gan 모델이 불안정한 이유와 비슷하다고 추정중)
q_model과 p_model의 가중치를 공유하여 학습을 진행합니다.
klue/roberta-base 기준 acc 67, large 기준 70이상 나옵니다.
전부 순수 epoch 3 훈련 기준입니다.
kfold를 사용하면 올라갈법도 하나, 제가 해봤을땐 오히려 떨어지거나 비슷했습니다. (튜닝하면 다를지도)

### 2. Dense retrieval의 kfold training
Kfold validation training을 수행합니다.
train 데이터만 분할하여 사용하며, val데이터는 남깁니다.

### 3. 모델 로드
model.load_model을 사용하면 output_dir 폴더의 체크포인트 숫자를 기준으로 정렬한뒤, 가장 마지막 체크포인트의 trainer.state 폴더에 들어가 best_checkpoint를 찾아 그 폴더의 내용을 로드합니다.
(하드코딩이긴 한데 이거말고 방법안떠오름)
근데 모델을 로드 후 sequential reranking을 하면 어째서인지 성능이 떨어집니다. (진짜 모르겠음)
cross reranking은 변함없이 잘 됩니다. 
이유는 좀 더 생각해보고 고쳐보겠습니다.

### 4. 모델 평가 기능 추가
model.search(mode = 'eval')
을 하면 validation 240개중 acc가 얼마나 나오는지 프린트하고 결과를 출력합니다.

### 5. Sequential reranking(이름몰라서 이렇게지음)
BM 25로 100개의 문서를 찾은 뒤, Dense retrieval의 Top_k만큼의 문서를 뽑아 출력합니다.
메모리가 아주 간당간당한데, del model과 gc.collect()로 손을 써두긴 했지만 잘 될지 모르겠습니다.
result = model.sequential_reranking(mode = 'test')로 사용합니다.
mode = 'eval'을 할 경우 validation 데이터 240개중 acc가 얼마나 나오는지 프린트하고 결과를 출력합니다. 
이거 되는거만 확인했고 지금알바가야돼서 성능을 못쟀네요 누가좀 재주시면감사하겠습니다


### 6. Cross reranking
BM 25로 문서 전체 (sample개수, 위키개수) score과 Dense retrieval로 문서 전체(sample개수, 위키개수) totalscore = score을 1.1 Densescore + 1 BM25score 로 구한 뒤 top_k개를 뽑습니다.
계수는 lamb_da = 1.1 기본값이고 수정해도됩니다. (lambda는 함수라 변수로못씀)
이게 메모리가 아주 간당간당할거같은데
일단 DPR로 추론해서 뽑고 모델 지우고 BM25로 뽑고 더해서 출력합니다.
result = model.cross_reranking(model = 'test')로 사용합니다.
model = 'eval'을 할 경우 validation 데이터 240개중 acc가 얼마나 나오는지 프린트하고 결과를 출력합니다.
small 모델 기준 dpr단독으로는 69.3, cross reranking은 92.몇인지기억안남 까지 성능이 올라갔었습니다.

# 여러분이 해 줬으면 하는 것
### 1. 결과 분석 (요게 좀 급함)
이번에 제작한 DPR 모델의 성능을 분석해주셨으면 좋겠습니다.
모델명 / 크기 별로 나눈 뒤에 kfold 별로도 나눠서 model.search(mode = 'eval')을 통해 acc를 기록해주시면 됩니다.
1. 그냥 train epoch 1 2 3 4 5 / 모델 크기별로
2. fold train fold 1 2 3 4 5 / epoch 1, 2 / 모델 크기별로
3. topk별로 5, 10, 20, 30 까지
위 결과가 잘 나오면 sequential / cross reranking의 설정 방향성을 잡을 수 있을 듯 싶습니다.

### 2. Extraction based MRC 모델의 train data 만들기 (난이도 하) (아직해결안됨)
멘토님께서 말씀하시길, ODQA에서 MRC 모델을 학습시킬 때 보통 데이터의 형태가 question : 뫄뫄 context : 진짜정답문서 + 리트리버 문서들 로 학습을 시킨다고 하셨습니다.
현재 MRC 모델은 train dataset에서의 context만 불러와서 학습을 하고 있습니다.
이를 멘토님께서 말씀주신 방법으로 학습시키도록 train / eval dataset을 만들어 주셨으면 좋겠습니다.

### 3. Generativee MRC Model (난이도 상) (보류)
현재 MRC.py에 Generative MRC model이 구현돼 있지만, 돌려본 결과, 성능이 좋지 않았습니다.
성능이 좋지 않은 이유는 다음과 같습니다.

1. 훈련 단계에서
question : 뫄뫄 context : 뫄뫄 로 입력이 들어오는데, extraction based mrc 모델에서는 context가 길면 잘라서 다음 input에 넣어주는 반면,
generative mrc model의 train data는 그러한 과정이 없습니다. 그래서 문서가 아무리 길어도 512자로 짤리더군요.
이게 아마 성능이 안나오는 이유일 듯 싶습니다.
그래서 train dataset이 길면 잘라서 다음 input에 넣고, answer이 알맞게 들어가는 전처리 부분을 만들어 주셨으면 좋겠습니다. (아마 매우 어렵겠지만)

2. Generative MRC Model은 기본적으로 파라미터가 커서 base 모델만 사용하더라도 OOM이 뜹니다.
그래서 small model을 사용할니다. 학습 단계에서도 validation 부분을 보면 성능이 좋지는 않았습니다 ㅠ

### 4. TFIDF retrieval / BM25 retrieval 의 eval 매트릭 구현
현재 Dense retrieval을 제외하고 BM25, TFIDF는 validation 240개의 데이터셋에 대해 문서를 떠올리고 얼마나 맞췄는지에 대한 acc를 구할 수 없는 상태입니다.
이를 구할 수 있도록 만들어주셨으면 좋겠습니다. (그래야 성능비교가 되니까)
구현은 제 Dense retrieval을 참고해서 만들어주셔도 되고 그냥 만들어보셔도 좋습니다.

### 5. Reader의 성능 향상
MRC 모델의 성능을 향상시켜보고싶은데 구조를 바꾸는건 어려울듯싶습니다.
데이터 전처리나 증강 등으로 성능을 향상시켜볼 수 있을 듯 싶습니다.
데이터 처리를 끝내셨다면 기존과 비교해서 결과롤 기록해주시면 감사하겠습니다.

### 6. 후처리에 대한 방향성 제시
위 과정이 전부 끝난다면, 시간상 모델을 바꿔서 결과 향상을 도모하는건 좀 어려울 듯 싶습니다.
결과 분석을 통해 가장 바람직한 모델 선정 및 훈련, topk 등을 정한 뒤에 마지막 결과물을 보고 처리해야할 부분을 처리하는 게 성능을 짜내는 데에 도움이 될 듯 싶습니다.

### 7. BM25
구현된 BM25는 get_sparse_embedding을 활용하고 있지 않은 것 같습니다.
한번 확인해주시고 고쳐주시면 감사하겠습니다.

### 8. 코드 이해안되는 부분 물어보기
코드 왜 이렇게 했는지 이해가 안되시면 물어봐주시면 좋겠습니다.
개념에 대해서는 잘 대답드리기 어려울 수도 있는데, 어떤 개념을 어떻게 구현하냐
이거는 왜이렇게 했냐 이런거는 잘 대답해드릴 수 있습니다 ㅋㅋ
진짜 사소한거도 괜찮습니다. (내가 여태 며칠동안 안된다찡찡댄건 다 사소한 이유때문이었음)


아마 난이도가 많이 높고 성능 기대값이 높지 않은 만큼, 제일 마지막의 수로 남겨둘 듯 싶습니다.
# 해결 된 것들
### 1. BM 25 리트리버 제작 (난이도 중) (석현님이 해결)
BM 25 리트리버는 tfidf에서 문서의 길이까지 고려하여(정규화) 문서를 찾아내는 알고리즘입니다.
retrieval에 class를 추가하고 arguments.py에 argument들을, 데이터를 불러오는 코드들은 All_dataset.py를 통해 넣으셔도 좋고,
독자적으로 코드를 만들어서 나중에 MRC 모델을 불러온 뒤 model.inference(BM25result)형태로 추론할 수도 있습니다.

### 2. Dense Embedding model 만들기 (난이도 중) (해결)
Dense Embedding Model은 이미 만들어졌지만, q_model과 p_model을 나눠서 학습하자니 메모리가 부족하여 두 가중치를 합치는 방향으로 학습하고 있습니다.
근데 마스터님께서 두 모델의 가중치를 전부 공유해버리면 의미가 없다고 말씀하셨기에 Head 부분을 linear모델로 나누어서 학습하였었는데
학습을 진행할수록 loss가 올라가고 accuracy가 떨어지는 기이한 현상이 발생중입니다.
이를 해결해주셨으면 좋겠습니다 ㅠ

### 3. Retrieval ensemble (난이도 중) (내가해결)
DPR 모델이 제대로 만들어 지면, BM 25로 한 백개의 문서를 찾고 그 중에서 DPR로 문서를 25개정도로 추리는 과정을 만들면 문맥과 단어를 모두 고려할 수 있습니다.
이를 통해 MRC 모델의 성능을 많이 올릴 수 있을 듯 싶습니다.


### 4. Cross validation (난이도 하) (내가해결)
현재 MRC 모델은 2epoch부터 validation loss가 상승합니다. 아마 오버피팅이 되고 있는 듯 싶습니다.
따라서 오버피팅에 효과적인 cross validation을 학습에 적용하면 효과를 볼 수 있을 것 같습니다.
이 부분은 난이도도 쉽고 금방 만들 수 있는 만큼 제가 직접 만들어보겠습니다.


#### 다른 분들이 안하시면 제가 차근차근 할 목록들입니다. 혹시라도 이 중 무언가 맡아주실 의향이 있다면 말씀 주시고 해주시면 감사하겠습니다 !!!
