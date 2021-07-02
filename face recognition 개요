 ## Face recognition 시스템

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e749045b-24db-47cf-ae0e-8066deb09a5c/R1280x0.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e749045b-24db-47cf-ae0e-8066deb09a5c/R1280x0.png)

deep face recognition : A survey paper, face recognition system 대략적인 개요

1. face detection: 사진에서 얼굴만 추출하는 단계
2. face alignment: 추출한 사진을 정면 사진으로 조정해주는 전처리 단계
3. face processing: augumentation 처리 등 데이터 처리 과정
4. feature extraction: 데이터로 부터 얼굴 feature 추출 단계
5. loss function: 추출한 feature를 통해 loss함수로 학습

## face recognition 종류

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/20a78ad4-942d-4128-8bdf-d7ae0eaa0cd7/R1280x0.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/20a78ad4-942d-4128-8bdf-d7ae0eaa0cd7/R1280x0.png)

1. closed-set face recognition: train set에 test set label이 포함되어 있는 경우 (ex> 사내 회원 인증) classificaion 문제와 동일하게 해결 가능
2. open-set face recognition: train set에 test set label이 포함되어 있지 않은 경우, 분류 문제로 해결이 불가능, 얼굴 feature을 특정 백터 공간안에 임베딩 시키는 방향으로 학습(metric learning)

## DeepFace: Closing the Gap to Human-Level Performance in Face Verification(2014)

face alignment와 feature representation(특징 추출)단계를 개선 시켜 인간 레벨의 정확도로 verification성능을 올린 논문

1. face alignment: 얼굴에서 67개의 기준점을 잡아 3D모델로 변환 후 정면을 바라보도록 transformation 진행

    ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f3ed3189-4bc0-4388-a687-1e3d6d2cf802/R1280x0.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f3ed3189-4bc0-4388-a687-1e3d6d2cf802/R1280x0.png)

2. feature representation: 정렬된 이미지가 들어오기 때문에 local convolution이 가능, output이 어굴을 표현하는 representation 백터가 된다

    ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/12d278d2-1738-4973-ac1a-6f34522555d6/R1280x0.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/12d278d2-1738-4973-ac1a-6f34522555d6/R1280x0.png)

    ## FaceNet: A unified Embedding for Face Recognition and Clustering(2015)

    이미지를 128차원으로 임베딩하여 유클리드 공간에서 이미지간 거리를 통해 분류하는 모델 즉 얼굴 사진에서 그사람에 대한 특징 값을 구해주는 모델로 이 값들간의 거리를 통해 identification, clustrering 등을 할 수 있게 해준다.

    ### metric learning

    데이터에 적합한 거리 함수를 머신러닝 알고리즘으로 만드는 방식 ⇒ 데이터를 구분하기 쉽게 만들어주는 거리 함수, 기존 feature로는 분류가 쉽지 않았던 데이터를 class label별로 잘 구분할 수 있게 만드는 metric을 학습하는 것

    ### Triplet loss

    ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ce4c9932-07e7-496c-897b-9876b72edc3c/R1280x0.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ce4c9932-07e7-496c-897b-9876b72edc3c/R1280x0.png)

    어떤 기준점이 되는 한 사람(anchor)와 같은 사람(positive), 다른 사람(negative)가 있을 때 학습 시키는 미니 배치 안에서 anchor, positive, negativee들이 임베딩된 값들의 유클리드 거리를 구해 loss 함수를 만들어 학습

    LT=32N∑i=1M/3max(0,d(xi,xi,p)−d(xi,xi,n)+α) (triplet loss)

    LT를 최소화 하는 것은 positive와의 거리는 가까워지고, negative와의 거리는 멀어지도록 학습한다.

    ** 분석 필요
r
    어떤 기준점이 되는 한 사람(anchor)와 같은 사람(positive), 다른 사람(negative)가 있을 때 학습 시키는 미니 배치 안에서 anchor, positive, negativee들이 임베딩된 값들의 유클리드 거리를 구해 loss 함수를 만들어 학습

    LT=32N∑i=1M/3max(0,d(xi,xi,p)−d(xi,xi,n)+α) (triplet loss)

    LT를 최소화 하는 것은 positive와의 거리는 가까워지고, negative와의 거리는 멀어지도록 학습한다.

    ** 분석 필요

    논문에서는 hard positive(같은 사람이지만 다르게 보이는 사람)과 hard Negative(다른 사람이지만 닮은 사람)을 학습에 방해하는 요소로 두고 아래와 같이 제어, hard postive의 경우에는 전부 학습을 진행했지만 hard negative의 경우는 이 조건을 만족하는 경우에만 학습 진행

    ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/faccb683-29dc-45a2-bb53-cd4b5af3b053/R1280x0.jpeg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/faccb683-29dc-45a2-bb53-cd4b5af3b053/R1280x0.jpeg)

    ## VGG-Face Model

    triplet loss와 softmax loss를 사용, faceNet보다 적은 데이터로 비슷한 성능을 달성 vgg와 같은 가벼운 레이어로 구성했음에도 좋은 성능

    .

    ?? Open set은 임베딩으로 어캐 해결하지? 결국 라벨링이 새로 학습되어야 하는건가?
