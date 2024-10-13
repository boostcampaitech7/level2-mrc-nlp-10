이번 프로젝트 템플릿입니다.

## 쓰는법
### 1. howtouse를 본다
### 2. arguments.py를 드가서 바꿀 부분을 고른다. 
(argument들이 정리도 안돼있고 개판이라 ctrl+f / command+f로 찾아서 바꾸셔야합니다 ㅎㅎ;;)

(Generation_based_MRC_arguments, 
Extraction_based_MRC_arguments,
Dense_search_retrieval_arguments, 
TF_IDF_retrieval_arguments 들을 치시고 찾으시면 금방 나올거에요.)

### 3. 바꿀 부분을 바꾼다. (모델 small 모델이고 에폭도 낮습니다. 바꿔야됨)

retrieval을 선언하고 build_faiss()
retrieval_model.search()를 하면 test데이터셋에 대한 retrieval 결과가 나옵니다.
이를 mrcmodel.inference(retrieval_result)에 넣으면 predict_result 폴더에 json 파일이 생성됩니다.


### Note
Cross validation으로도 훈련방식을 바꿀 수 있는데 그건 eda좀 하고 해보겠습니다
