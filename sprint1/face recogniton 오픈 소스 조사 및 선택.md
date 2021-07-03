# face recogniton 오픈 소스 조사 및 선택

이번 i-mind 프로젝트에서 아동의 얼굴 식별에 사용할 face recognition 오픈 소스를 선택하고자 한다. 
우선 github의 star수와 최근 업데이트 날짜, issue 등을 검토하여 python 기반 arcface를 지원하는 오픈소스 2개를 선정하였고
이 중 확장 가능성(튜닝), 정확도, 사용 편의성을 기준으로 최종 오픈소스를 선택하고자 한다. 

# 1. insightface

[https://github.com/deepinsight/insightface](https://github.com/deepinsight/insightface)

arcface 저자가 참여한 프로젝트의 official open source로 mxnet와 pytorch 두가지 버전의 소스코드와 정제된 데이터셋, pre-trained된 모델을 제공한다.

공식 open source이기 때문에 가장 정교한 모델 구현과 높은 정확도가 장점이다

### insightface를 선택하지 못한 이유

가장 정교한 모델 구현와 높은 정확도가 장점이지만 그럼에도 이번 프로젝트에서 insightface를 선택하지 않기로 했다. 가장 큰 이유로는 개발에 참고할 문서가 너무 적기 때문이다. 공식 readme도 매우 빈약하며, tutorial이나 demo도 존재하지 않아 코드를 이해하기 어려웠다. 또한 pytorch 버전이 최근에 official 버전이 나왔다고 하나 face recognition 외에도 앞에 진행되어야할 face detection, face alignment는 별도로 mxnet으로 구현되어 있기 때문에 파이프라인 구축에 어려움이 예상된다.

정리:

1. 참고 문서의 부족
2. 파이프라인 모델의 코드레벨이 통일되어 있지 않음

# 2. deepface

[https://github.com/serengil/deepface](https://github.com/serengil/deepface)

keras와 tensorflow 기반의 face recognition framework이다. face recognition 분야의 state-of-the-art 모델들 vgg-face, google FaceNet, OpenFace, DeepFace, DeepID, ArcFace, Dlib 등을 warpping하여 사용하기 쉽게 만들었다.

## deepface를 선택한 이유

### 1. 풍부한 개발 문서

readme만 읽어도 정리가 잘되어 있어서 파악이 갔으며, youtube를 통해 공식 tutorial도 확인할 수 있다.

가장 큰건 개발자가 블로그를 통해 opensource에 대한 설명과 의도 출저 등을 포스팅하고 있다는 점이었다.

[https://sefiks.com/](https://sefiks.com/page/9/)

### 2. 파이프라인 모델의 통일화

deepface는 face recognition의 파이프라인인 face detection, face alignment 과정도 같은 코드레밸로 구현하여 파이프라인을 구축하였다. 따라서 파이프라인 구축에 시간을 단축하여 핵심인 face representation에 집중할 수 있다. 또한 face detection 같은 부분도 retina face와 같은 sota 모델을 재현하여 파이프라인을 구축한 점이 인상적이다.

### 3. 준수한 정확도

deepface의 arcface는 LFW 데이터셋에서 99.40%의 정확도를 재현하였다. original arcface의 정확도인 99.83%에 비해 낮은 수치이긴 하지만 어짜피 아시아인과 아동에 대한 추가학습을 진행하야하기 때문에 실제 정확도 보다는 추가 학습할 때 더 쉽고 용이하게 할 수 있는지가 중요하여 deepface의 정확도도 충분히 경쟁력이 있다고 생각하였다.

### 4. 추가 학습 가능성

3번과 연결되는 부분이다. deepface는 선 학습된 모델을 사용하는데 이 선 학습된 모델을 학습시키는 code 또한 공개하고 있다

[https://github.com/leondgarse/Keras_insightface](https://github.com/leondgarse/Keras_insightface)

위 github에서 학습 환경과 데이터 정제 방법, 코드까지 자세히 설명하고 있고 arcface와 같은 loss함수도 keras로 재현하였기 때문에 추가 학습 시 가장 용이한 opensource라고 생각한다.

최종적으로 풍부한 개발 문서와 통일된 코드, 추가 학습 가능성이 열려있는 deepface opensource를 선택하기로 하였다.
