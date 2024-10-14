이번 프로젝트 템플릿입니다.

## 쓰는법
### 1. howtouse를 본다
### 2. arguments.py를 드가서 바꿀 부분을 고른다. 
본인이 사용할 경로로 지정하셔야 아마 경로 에러가 발생하지 않을겁니다 !!!!
(argument들이 정리도 안돼있고 개판이라 ctrl+f / command+f로 찾아서 바꾸셔야합니다 ㅎㅎ;;)

(Generation_based_MRC_arguments, 
Extraction_based_MRC_arguments,
Dense_search_retrieval_arguments, 
TF_IDF_retrieval_arguments 들을 치시고 찾으시면 금방 나올거에요.)

### 3. 바꿀 부분을 바꾼다.

retrieval을 선언하고 build_faiss()
retrieval_model.search()를 하면 test데이터셋에 대한 retrieval 결과가 나옵니다.
이를 mrcmodel.inference(retrieval_result)에 넣으면 predict_result 폴더에 json 파일이 생성됩니다.


## 여러분이 해 줬으면 하는 것
### 1. BM 25 리트리버 제작 (난이도 중)
BM 25 리트리버는 tfidf에서 문서의 길이까지 고려하여(정규화) 문서를 찾아내는 알고리즘입니다.
retrieval에 class를 추가하고 arguments.py에 argument들을, 데이터를 불러오는 코드들은 All_dataset.py를 통해 넣으셔도 좋고,
독자적으로 코드를 만들어서 나중에 MRC 모델을 불러온 뒤 model.inference(BM25result)형태로 추론할 수도 있습니다.

### 2. Extraction based MRC 모델의 train data 만들기 (난이도 하)
멘토님께서 말씀하시길, ODQA에서 MRC 모델을 학습시킬 때 보통 데이터의 형태가 question : 뫄뫄 context : 진짜정답문서 + 리트리버 문서들 로 학습을 시킨다고 하셨습니다.
현재 MRC 모델은 train dataset에서의 context만 불러와서 학습을 하고 있습니다.
이를 멘토님께서 말씀주신 방법으로 학습시키도록 train / eval dataset을 만들어 주셨으면 좋겠습니다.

### 3. Dense Embedding model 만들기 (난이도 중)
Dense Embedding Model은 이미 만들어졌지만, q_model과 p_model을 나눠서 학습하자니 메모리가 부족하여 두 가중치를 합치는 방향으로 학습하고 있습니다.
근데 마스터님께서 두 모델의 가중치를 전부 공유해버리면 의미가 없다고 말씀하셨기에 Head 부분을 linear모델로 나누어서 학습하였었는데
학습을 진행할수록 loss가 올라가고 accuracy가 떨어지는 기이한 현상이 발생중입니다.
이를 해결해주셨으면 좋겠습니다 ㅠ

### 4. Retrieval ensemble (난이도 중)
DPR 모델이 제대로 만들어 지면, BM 25로 한 백개의 문서를 찾고 그 중에서 DPR로 문서를 25개정도로 추리는 과정을 만들면 문맥과 단어를 모두 고려할 수 있습니다.
이를 통해 MRC 모델의 성능을 많이 올릴 수 있을 듯 싶습니다.

### 5. Generativee MRC Model (난이도 상)
현재 MRC.py에 Generative MRC model이 구현돼 있지만, 돌려본 결과, 성능이 좋지 않았습니다.
성능이 좋지 않은 이유는 다음과 같습니다.
1. 훈련 단계에서
question : 뫄뫄 context : 뫄뫄 로 입력이 들어오는데, extraction based mrc 모델에서는 context가 길면 잘라서 다음 input에 넣어주는 반면,
generative mrc model의 train data는 그러한 과정이 없습니다. 그래서 문서가 아무리 길어도 512자로 짤리더군요.
이게 아마 성능이 안나오는 이유일 듯 싶습니다.
그래서 train dataset이 길면 잘라서 다음 input에 넣고, answer이 알맞게 들어가는 전처리 부분을 만들어 주셨으면 좋겠습니다. (아마 매우 어렵겠지만)

2. Generative MRC Model은 기본적으로 파라미터가 커서 base 모델만 사용하더라도 OOM이 뜹니다.
그래서 small model을 사용할니다. 학습 단계에서도 validation 부분을 보면 성능이 좋지는 않았습니다 ㅠ

아마 난이도가 많이 높고 성능 기대값이 높지 않은 만큼, 제일 마지막의 수로 남겨둘 듯 싶습니다.

### 6. Cross validation (난이도 하)
현재 MRC 모델은 2epoch부터 validation loss가 상승합니다. 아마 오버피팅이 되고 있는 듯 싶습니다.
따라서 오버피팅에 효과적인 cross validation을 학습에 적용하면 효과를 볼 수 있을 것 같습니다.
이 부분은 난이도도 쉽고 금방 만들 수 있는 만큼 제가 직접 만들어보겠습니다.

#### 다른 분들이 안하시면 제가 차근차근 할 목록들입니다. 혹시라도 이 중 무언가 맡아주실 의향이 있다면 말씀 주시고 해주시면 감사하겠습니다 !!!