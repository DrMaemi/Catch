
# Multi-person Real-time Action Recognition Based-on Human Skeleton

<p>

[원본 저장소 링크](https://github.com/felixchenfy/Realtime-Action-Recognition)<br>
모델 알고리즘과 데이터 학습 및 결과에 대한 내용을 참조하려면 위 링크를 참조하십시오.
</p>
<p>

본 저장소는 Tensorflow 2.0+ 환경에서 동작하도록 수정되었습니다.
</p>

![](https://github.com/felixchenfy/Data-Storage/raw/master/EECS-433-Pattern-Recognition/recog_actions.gif)

![](https://github.com/felixchenfy/Data-Storage/raw/master/EECS-433-Pattern-Recognition/recog_actions2.gif)

## 목차
<p>

- [1. 동작 환경](#1-동작-환경)
</p>

<br

## 1. 동작 환경
<p>

- Window 10 or Ubuntu
- CUDA
- Anaconda
</p>
<p>

가상환경 구축
```bash
conda create -n tf2-rar python=3.6.4 -y
conda activate tf2-rar
pip install tensorflow
```
</p>
<p>

tf2-rar 저장소 clone
```bash

git clone https://github.com/DrMaemi/tf2-rar.git
cd tf2-rar
pip install cython
pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
pip install -r requirements.txt
```
</p>
<p>

tf-pose-estimation 포즈 추출기 환경 구성
```bash
cd src
mkdir githubs
cd githubs
git clone https://github.com/DrMaemi/tf2-pose-estimation.git
mv tf2-pose-estimation tf-pose-estimation
cd tf-pose-estimation
```
```bash
sudo apt install swig
swig -python -c++ pafprocess.i && python setup.py build_ext --inplace
```
</p>
<p>

설치 확인
```bash
cd ../../
python run.py --model=mobilenet_thin --resize=432x368 --image=./images/p1.jpg
```
</p>

<p>

원본 저장소 저자는 **10프레임**의, **5명 이하**의 사람이 있는 비디오에서 모델을 동작시킬 것을 권장한다.
</p>