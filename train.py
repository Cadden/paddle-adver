import paddle
from paddle.nn import Linear
import paddle.nn.functional as F
from paddle.vision.transforms import ToTensor, Compose, Normalize
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import cv2

from Model import ModelA, ModelB
from LinfPGDAttack import LinfPGDAttack

# 在Mnist数据集上进行PGD-steps100-restarts20-sourceA: 89.3%训练
def train1(model):
    #开启GPU
    use_gpu = True
    paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')

    model.train()
    epochs = config['max_num_training_steps']  #3

    # 用Adam作为优化函数
    optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters()) 
    
    for epoch in range(epochs):
        for batch_id, data in enumerate(train_loader()):
            x_data = data[0]
            # print(x_data.shape)
            
            y_data = data[1]
            predicts = model(x_data)
            loss = F.cross_entropy(predicts, y_data)
            # 计算损失
            acc = paddle.metric.accuracy(predicts, y_data)
            loss.backward()
            if batch_id % 300 == 0:
                print("nat epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id, loss.numpy(), acc.numpy()))

            attack = LinfPGDAttack(model, 
                       config['epsilon'],
                       config['k'],
                       config['a'],
                       config['random_start'],
                       x_data,
                       y_data)
            x_batch_adv = attack.perturb(x_data, y_data)
            x_batch_adv = paddle.to_tensor(x_batch_adv, dtype='float32')

            predicts1 = model(x_batch_adv)
            loss1 = F.cross_entropy(predicts1, y_data)
            # 计算损失
            acc1 = paddle.metric.accuracy(predicts1, y_data)
            loss1.backward()
            if batch_id % 300 == 0:
                print("adv epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id, loss1.numpy(), acc1.numpy()))

            optim.step()
            optim.clear_grad()
    #保存模型参数
    paddle.save(model.state_dict(), 'mnist-PGD-steps100-restarts20-sourceA.pdparams')


#在Mnist数据集上进行PGD-steps100-restarts20-sourceA: 95.7%训练
def train2(modelA, modelB, x):
    modelB.train()
    modelA.train()
    epochs = config['max_num_training_steps'] # 5
    optimA = paddle.optimizer.Adam(learning_rate=0.001, parameters=modelA.parameters())
    optimB = paddle.optimizer.Adam(learning_rate=0.001, parameters=modelB.parameters())
 
    # 用Adam作为优化函数
    for epoch in range(epochs):
        for batch_id, data in enumerate(train_loader()):
            x_data = data[0]
            y_data = data[1]

            # train modelB
            predictsB = modelB(x_data)
            lossB = F.cross_entropy(predictsB, y_data)
            # 计算损失
            accB = paddle.metric.accuracy(predictsB, y_data)
            lossB.backward()
            if batch_id % 300 == 0:
                print("modelB epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id, lossB.numpy(), accB.numpy()))

            optimB.step()
            optimB.clear_grad()

            predicts1 = modelA(x_data)
            loss1 = F.cross_entropy(predicts1, y_data)
            # 计算损失
            acc1 = paddle.metric.accuracy(predicts1, y_data)
            loss1.backward()
            if batch_id % 300 == 0:
                print("nat epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id, loss1.numpy(), acc1.numpy()))
           
            attack = LinfPGDAttack(modelB, 
                       config['epsilon'],
                       config['k'],
                       config['a'],
                       config['random_start'],
                       x_data,
                       y_data)
            x_batch_adv = attack.perturb(x_data, y_data)
            x_batch_adv = paddle.to_tensor(x_batch_adv, dtype='float32')

            predicts1 = modelA(x_batch_adv)
            loss1 = F.cross_entropy(predicts1, y_data)
            # 计算损失
            acc1 = paddle.metric.accuracy(predicts1, y_data)
            loss1.backward()
            if batch_id % 300 == 0:
                print("adv epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id, loss1.numpy(), acc1.numpy()))            

            optimA.step()
            optimA.clear_grad()
    #保存模型参数
    paddle.save(modelA.state_dict(), 'mnist-PGD-steps100-restarts20-sourceA-modelA-'+x+'.pdparams')
    paddle.save(modelB.state_dict(), 'mnist-PGD-steps100-restarts20-sourceA-modelB-'+x+'.pdparams')



  

if __name__=='__main__':
      # load dataset 
    transform = Compose([Normalize(mean=[127.5],
                               std=[127.5],
                               data_format='CHW')])
    # 使用transform对数据集做归一化
    print('download training data and load training data')
    train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
    print('load finished')

    # 加载配置文件
    with open('config.json') as config_file:
        config = json.load(config_file)

    #使用GPU将use_gpu变量设置成True
    use_gpu = True
    paddle.set_device('gpu') if use_gpu else paddle.set_device('cpu')

    
    # 加载训练数据
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=64, shuffle=True)

    #在Mnist数据集上进行PGD-steps100-restarts20-sourceA: 89.3%训练
    model = ModelA()
    train1(model)

     # 加载训练数据
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=64, shuffle=True)

    #在Mnist数据集上进行PGD-steps100-restarts20-sourceA: 95.7%训练
    modelA = ModelA()
    modelB = ModelA()
    train2(modelA, modelB, '1')

     # 加载训练数据
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=64, shuffle=True)

    #在Mnist数据集上进行PGD-steps40-restarts1-sourceB: 96.4%%训练
    config['k'] = 1
    config['max_num_training_steps'] = 40
    modelA = ModelA()
    modelB = ModelB()
    train2(modelA, modelB, '2')
