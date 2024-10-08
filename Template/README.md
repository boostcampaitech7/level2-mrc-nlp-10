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

### retrieval model은 어떻게 쓰나요 ?

```python

 model = 뫄뫄()
 model.train() (dense retrieval 사용시)
 model.build_faiss()
 model.search()

```

### MRC model은 어떻게 쓰나요 ?

```python

 model = 뫄뫄()
 model.train()
 model.inference()

```

### ** 주의점 **
Dense retrieval은 question embedding 모델과 context embedding 모델 두개를 불러오기 때문에 OOM이 자주 발생합니다.
Dense retrieval 모델이 가끔 accuracy가 24퍼를 벗어나지 못하는 local minima를 갈 때가 있는데 그냥 시드 바꾸고 다시 돌리시면 됩니다. (가끔은 그냥 다시 돌려도 됨)
Extraction MRC 모델이 훈련은 잘 되는데 val_loss가 No log가 뜹니다.. ㅠㅠ 아주 오랜시간 고쳐보려 노력했으나 실패했습니다

### Note
Cross validation으로도 훈련방식을 바꿀 수 있는데 그건 eda좀 하고 해보겠습니다
