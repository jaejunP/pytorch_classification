#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
전체적인 구조

#첫 번째 레이어
합성곱(in_channel = 1. out_channel = 32, kernel_size = 3, stride = 1, padding= 1) + 활성화 함수(Activation function) ReLU
맥스풀링(kernel_size = 2, stride= 2)

#두 번째 레이어
합성곱(in_channel = 32. out_channel = 64, kernel_size = 3, stride = 1, padding= 1) + 활성화 함수(Activation function) ReLU
맥스풀링(kernel_size = 2, stride= 2)

#세 번째 레이어
batch_size x 7 x 7 x 64 => batch size = 3136 (특성맵을 펼치는 레이어)
Fully connected(전결합층 뉴런 10개) + 활성화 함수 softmax
"""


# In[3]:


#구현에 필요한 ㄴ라이브러리 불러오기
import torch
import torch.nn as nn
# as를 선언핳지 않았을 경우 torch.nn.Conv2d()
# as를 선언헀을 경우 nn.Conv2d()


# In[4]:


#임의의 텐서 생성(1, 1, 28, 28) → MNIST데이터를 받기 위한 공간
# (1채널, input이 1(한개)인 28 x 28데이터)
inputs = torch.Tensor(1, 1, 28, 28)
print("텐서의 크기: {} ".format(inputs.shape))


# In[5]:


#첫 번째 합성곱층 선언하기
conv1 = nn.Conv2d(1, 32, 3, padding = 1) #Conv2d 내의 인자값(input_channel, output_channel, 커널 사이즈, 패딩 ) / 2d는 2차원 이미지(MINIST는 2차원이므로)
print(conv1)


# In[29]:


#두 번째 합성곱층 선언하기
conv2 = nn.Conv2d(32, 64, 3, padding = 1)
print(conv2)


# In[30]:


#세 번째 합성곱층 선언하기(풀링 레이어)
pool = nn.MaxPool2d(2)   #간단히 선언하면 선언된 하나의 정수값을 통해 인자 2는 kernel_size, stride 값을 선언한 것
#MaxPool : 2차원에서 앞으로 받았던 모든 feauturemap을 모두 펼치겠다 / 
print(pool)

#pool = nn.MaxPool2d(kernel_size = 2, stride = 2, padding= 0, dilation=1, ceil_mode=False)로 해도 동일


# In[31]:


#선언 된 레이어를 연결하는 과정
#첫번째 레이어 지나가기
out = conv1(inputs)
print(out.shape)


# In[32]:


#컨벌루셔널 레이어를 거쳐 갈 경우 마지막에 pooling layer가 존재함.
out = pool(out)
print(out.shape)  #겹쳐진 이미지이므로 반으로 줄어듬


# In[33]:


#두 번째 레이어 지나가기
out = conv2(out)
print(out.shape)


# In[34]:


out = pool(out)
print(out.shape)


# In[35]:


#view()함수를 사용해서 텐서를 펼치는 작업(텐서를 통합한다)
out = out.view(out.size(0), -1)   #view를 통해서 전체 차원이 합쳐짐
print(out.shape)


# In[36]:


fc = nn.Linear(3136, 10)  #fully connected : 전체를 연결(기초적인 신경망) 통합된 텐서를 통해 input로 받고, 
                                # 0~9까지의 결과를 갖게 되기 때문에 10개의 output
out = fc(out)
print(out.shape)

#위까지 전체 모델 설정 과정


# In[84]:


#위에서 선언한 신경망을 통해서 MNIST(숫자 데이터셋) 분류 문제 해결하기
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transform
import torch.nn.init


# In[85]:


device = 'cuda' if torch.cuda.is_available() else 'cpu' 
#torch.cuda.is_available()함수는 False/True의 boolean형태로 나타나게 되는데, CUDA를 사용 가능할 경우 True를 반환하기 때문에
#if 조건을 만족하는 device = 'cuda'가 되고, 사용이 불가능할 경우 False를 반환하기 때문에 else 조건을 만족하는 device = 'cpu'가 됨.

#GPU 사용시에 고정시킬 시드를 선언
torch.manual_seed(777)

#만약 CPU가 아닌 GPU를 사용해서 훈련을 진행할 경우 고정 된 시드를 통해 GPU로 학습.
if device == 'cuda':
    torch.cuda.manual_seed_all(777)


# In[86]:


#학습에 사용되는 변수(파라미터들 : 학습률, 에폭, 배치사이즈 등) 선언
learning_rate = 0.001
training_epoch = 15
batch_size = 100


# In[89]:


#MNIST train, test 데이터뎃 을 torchvision,datasets를 통해 비교적으로 쉽게 가져오기

mnist_train = dsets.MNIST(root='MNIST_data/',
                         train = True,
                         transform=transform.ToTensor(),
                         download=True)

mnist_test = dsets.MNIST(root='MNIST_data/',
                       train=False,
                       transform=transform.ToTensor(),
                       download=True)


# In[90]:


#데이터로더를 이용하여 배치크기를 지정
data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)


# In[97]:


class CNN(torch.nn.Module): #CNN구조를 사용한 모델을 선언하기 위한 클래스
    def __init__(self):  #초기화 함수, 동시에 모델의 구조를 선언하는 함수
        super(CNN, self).__init__()
        
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                                          torch.nn.ReLU(), torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                                          torch.nn.ReLU(), torch.nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.fc = torch.nn.Linear(7 * 7 * 64, 10, bias=True)
        
        torch.nn.init.xavier_uniform_(self.fc.weight)
        
    def forward(self, x): #레이어를 거치며 계산되는 가중치를 전달하기 위해 사용하는 함수
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        
        return out
    


# In[98]:


model = CNN().to(device)   #.to(device) : 기계가 연산을 하기 위해서 선언
# model이라는 객체를 통해서 CNN클래스가 선언이되고, 이는 .to(device)를 통해 gpu에서 연산하게 됨


# In[99]:


#비용함수(cost function = loss function)은 크로스엔트로피로스로 선언
#옵티마이저는 Adam으로 설정
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# In[100]:


total_batch = len(data_loader)
print("총 배치의 수 : {}".format(total_batch))   
#MNIST dataset은 총 60,000개 데이터를 갖고 있기때문에 위의 선언한 batch_size=100으로 나누면 총 600 batch_size를 갖게됨


# In[102]:


for epoch in range(training_epoch):
    avg_cost = 0
    
    for X, Y in data_loader:
        X = X.to(device)
        Y = Y.to(device)
        
        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()
        
        avg_cost += cost / total_batch
        
    print(' [Epoch: {:>4}] cost = {:>.9}'.format(epoch+1, avg_cost))


# In[104]:


with torch.no_grad():
    X_test = mnist_test.test_data.view(len(mnist_test),1,28,28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)
    
    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float(). mean()
    print('Accuracy:', accuracy.item())


# In[ ]:




