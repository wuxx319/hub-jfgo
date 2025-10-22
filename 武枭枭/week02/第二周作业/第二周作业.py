#引入模块（向量计算，张量计算，张量函数模块,随机模块）
import random
import numpy as np
import torch
import torch.nn as nn
import math
import sys
#定义模型类，定义预测模型
class Trainmodel(nn.Module):
#初始化,定义初始化入参
    def __init__(self,input_size):
        super().__init__()#父类模型初始化#
        #定义网络层#
        self.layer=nn.Linear(input_size,5)
        self.loss=nn.CrossEntropyLoss()
    #定义目标模型函数#
    def forward(self,x,y=None):
    #定义预测值#
        y_pred=self.layer(x)
        if y is not None:
            return self.loss(y_pred,y)
        else:
            return y_pred
#定义训练函数和训练集
#训练函数是没有入参的#
def simple():
    x=np.random.random(5)
    y=np.argmax(x)
    return x,y
#通过训练函数创建训练集#
def train_simple(train_simple_num):#入参其实只有训练集的大小
    X=[]
    Y=[]
    for i in range(train_simple_num):
        x,y=simple()
        X.append(x)
        Y.append(y)
        #以上生成的结果为向量，需转化为张量进行计算操作#
    return torch.FloatTensor(X),torch.LongTensor(Y)
#定义测试函数集，关闭训练模式，定义测试样本数，创建测试集，关闭梯度计算，计算准确率#
def test_model(model):
    model.eval()
    test_num=100
    x,y=train_simple(test_num)
    correct,wrong=0,0
    with torch.no_grad():
        y_pred=model(x)
        for y_p,y_t in zip(y_pred,y):
            if torch.argmax(y_p)==int(y_t):
                correct+=1
            else:
                wrong+=1
        print(f"本轮测试共{test_num}个样本，正确：{correct}个，准确率：{correct/(correct+wrong)*100}%")
    return correct/(correct+wrong)
#定义训练流程,模型实例化，训练轮次，样本集大小，学习率，优化器选择，每轮训练次数，训练样本创建，预测值定义，训练日志
def main():
    model=Trainmodel(input_size=5)
    epoch=50
    total_simple_num=5000
    batch_size=20
    lr=0.01
    optim=torch.optim.Adam(model.parameters(),lr=lr)
    train_x, train_y = train_simple(total_simple_num)
    log=[]
#训练模式开启，损失值记录，
    for i in range(epoch):
        model.train()
        watch_loss = []
        for batch_index in range(total_simple_num//batch_size):
            x=train_x[batch_index*batch_size:(batch_index+1)*batch_size]
            y=train_y[batch_index*batch_size:(batch_index+1)*batch_size]
#梯度重置，梯度计算，反向传播，权重更新，注意此时的损失值为张量数据，需转化为数值
            optim.zero_grad()
            loss_value=model(x,y)
            loss_value.backward()
            optim.step()
            watch_loss.append(loss_value.item())
        print(f"第{i+1}轮训练损失值为：{np.mean(watch_loss)}")
        zhunquelv=test_model(model)
        log.append([zhunquelv,np.mean(watch_loss)])
#保存模型,打印权重
    torch.save(model.state_dict(),"model.pt")
    print(model.state_dict())
#定义预测模型，实例化，下载模型，加载参数
def true(model_path,ture_num):
    input_size=5
    model=Trainmodel(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        result=model.forward(torch.FloatTensor(ture_num))
    for m,n in zip(ture_num,result):
        print(f"输入{m},分类为：{torch.argmax(n)},概率分别为：{n}")
if __name__=="__main__":
    main()
    ture_num=[[1, 3, 4, 2, 1],
        [1, 4, 4, 2, 4],
        [2, 3, 1, 2, 4],
        [3, 3, 1, 2, 3],
        [2, 3, 2, 1, 4]]
    h=true("model.pt",ture_num)
