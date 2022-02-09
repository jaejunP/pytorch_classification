# pytorch_classification
파이토치를 사용한 CNN신경망 구성과 MNIST 분류문제 해결


파이토치 설명 & 모듈 및 라이브러리

파이토치 : define by run VS 텐서플로우, 케라스 : define and run

파이토치 패키지의 구성요소

torch : GQU지원기능을 갖춘 넘파이와 같은 라이브러리
import torch
x = torch.rand(5,3)는 x = numpy.rand(5,3)와 같음

torch.autograd : 자동 미분해서 포워드(순전파), 백워드(역전파) - 3Bblue1Brown의 비디오 참고

tgorch.optim : optimizer같은거 / SGD, RMSProp, Adam 등과 같은 표준 최적화 방법으로...torch.nn와 같이 사용

torch.nn : 최고의 유연성을 위해 설계된 자동 그래프와 깊이 통합된 신경 네트워크 라이브러리 / RNN, CNN, GAN, Relu, sigmoid, MSELoss 등(Contarnier, convolition layer~ normalization layers, transformer, linear, dropout, loss를 주로 사용함)

torch.utils : 편의를 위해 DataLoader, Trainer 및 기타 유틸리티 기능

+ torchvision : 여러 데이터셋을 미리 사용할 수 있도록 만들어 놓은 파이토치 패키지
(augmentaion :리사이즈, 플립, 로테이션 등 여러 바업으로 트랜스폼을 해서 데이터 agremantion에 사용)


https://pytorch.org/get-started/locally/에서 파이토치 설치 하는 법(사진참고)
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

conda create -n pytorch python=3.8 (가상환경 만들기)
conda activate pytorch (파이토치 활성화)
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch (파이토치 비전, 오디오, 쿠다툴킷 설치)
						      -c : channel

+ → 다른 버전을 사용하고 싶을 때 : ex) conda install pytorch=1.7.0 torchvision=0.8.0 cudatoolkit=10.2

import torch
torch.__version__ (버전확인)

파이토치 설치 끝
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

CUDA 설치 시작

<1>
먼저 기존 cuda 삭제(development, documentation, nishg nvtx, runtime, sample, visual studio intergation, nsight compute, nsight systems, nsight visual studio edition 삭제)
사진에 찍은 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA의 11.0 폴더 삭제하기

https://developer.nvidia.com/cuda-toolkit(접속)
→ download now → Archive of Previous CUDA Releases(https://developer.nvidia.com/cuda-toolkit-archive)
→ CUDA Toolkit 11.3 Downloads 다운받기 → windows & x86_64 & 10 & exe(local) / 설치는 밑에 visualstudio설치 후


<2>
https://my.visualstudio.com/Downloads?q=visual%20studio%202017&wt.mc_id=o~msft~vscom~older-downloads
에서 2번째꺼 다운받기(슬랙 확인 : vs Communitiy)(사진 확인)(Visual Studio Community 2017 (version 15.9))
→ cuda에서 사용될 파일이 있기때문에 설치하는거임

<3>
CUDA Toolkit 11.3 Downloads 실행하기!(모든 값 디폴트)

ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
<4>
cudnn 설치하기(슬랙으로도 다운 가능 : cudnn)

https://developer.nvidia.com/rdp/cudnn-archive 접속(cudnn검색 → cudnn archive) →
cudnn 8.2.1버전, for cuda 11 다운!(만약 저번에 다운받은 압축폴더 있으면 그걸로 사용하기)
→ 압축풀면 안의 파일들을 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3에 복붙(계속 덮어쓰기)(사진)
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
<5> 폴더 경로 바꾸기
새폴더(이름 : pytorch)를 만들고 (경로)   / cd (폴더경로)          /cd : change directory
prompt에서 cd C:\Users\11\Desktop\pytorch
경로 바꾸고

conda install jupyter
jupyter notebook


https://github.com/에서 계정 만들기
https://github.com/jaejunP/pytorch_classification
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
패키지 관리자 pip 혹은 conda 등으로 git이 설치 되어 있을 경우

Github repository에 저장되어 있는 코드를 다운받을 수 있다.

명령어 : git clone '주소명'
→ git clone https://github.com/jaejunP/pytorch_classification.git
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
확장자 : .ppy / .ipynb
