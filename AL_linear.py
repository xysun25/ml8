'''
1.线性模型
@date: 2021-10-14
@author: Alan
'''

import numpy as np

def forward(x,w):
    return w*x

def loss(x,y,w):
    y_pred = forward(x, w)
    error = (y_pred - y)**2
    return error

def gradient_desent(x_train,y_train):    
    i = 0      # 训练次数
    w = 0.0    # 初始化权重参数
    error = 0.0001
    list_w = []
    list_mse = []
    while True:
        loss_sum = 0.0
        k=0
        while (k<len(x_train)): # for x, y in zip(x_train, y_train):
            x = x_train[k]
            y = y_train[k]
            loss_val = loss(x,y,w)
            loss_sum += loss_val
            k = k + 1
        mse = loss_sum/len(x_train) # 均方根误差
        print('第'+str(i)+'次训练更新后均方根误差mse:', mse)
        if mse<error: # 收敛准则
            print('达到收敛准则，退出训练！')
            break
        w = w + 0.1  #更新权重
        i = i + 1    #更新迭代次数
        list_w.append(w)
        list_mse.append(mse)

    print('最优权重w为:',w)
    return w, list_w, list_mse


x_train = [1,2,3,4,5]
y_train = [2,4,6,8,10]

w, list_w, list_mse = gradient_desent(x_train,y_train)

import matplotlib.pyplot as plt
plt.plot(list_w,list_mse,color='#F92672')
plt.xlabel('w')
plt.ylabel('MSE')
plt.show()