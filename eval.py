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


def test(model):
    model.eval()
    global acc_adv, acc_nat, num_exam
    batch_size = 64
    for batch_id, data in enumerate(test_loader()):
        x_data = data[0]
       
        num_exam = num_exam + 1
        # print(num,type(num))
        y_data = data[1]
        predicts = model(x_data)
        # 获取预测结果
        loss = F.cross_entropy(predicts, y_data)
        acc = paddle.metric.accuracy(predicts, y_data)
        # print(type(acc.numpy()), type(acc_nat))
        acc_nat = acc_nat + acc.numpy()
        if batch_id % 20 == 0:
            print("nat batch_id: {}, loss is: {}, acc is: {}".format(batch_id, loss.numpy(), acc.numpy()))

        predicts = model(x_data)
        attack = LinfPGDAttack(model, 
                       config['epsilon'],
                       config['k'],
                       config['a'],
                       config['random_start'],
                       x_data,
                       y_data)
        x_batch_adv = attack.perturb(x_data, y_data)
        x_batch_adv = paddle.to_tensor(x_batch_adv, dtype='float32')
        predicts = model(x_batch_adv)
        # 获取预测结果
        loss = F.cross_entropy(predicts, y_data)
        acc = paddle.metric.accuracy(predicts, y_data)
        acc_adv = acc_adv + acc.numpy()
        if batch_id % 20 == 0:
            print("adv batch_id: {}, loss is: {}, acc is: {}".format(batch_id, loss.numpy(), acc.numpy()))

def test2(modelA, modelB):
    modelA.eval()
    modelB.eval()
    global acc_adv, num_exam, acc_nat
    batch_size = 64
    for batch_id, data in enumerate(test_loader()):
        x_data = data[0]
        y_data = data[1]
        num_exam = num_exam + 1       
        predicts = modelA(x_data)
         # 获取预测结果
        loss = F.cross_entropy(predicts, y_data)
        acc = paddle.metric.accuracy(predicts, y_data)
        acc_nat = acc_nat + acc.numpy()
        if batch_id % 20 == 0:
            print("nat batch_id: {}, loss is: {}, acc is: {}".format(batch_id, loss.numpy(), acc.numpy())) 

        attack = LinfPGDAttack(modelB, 
                       config['epsilon'],
                       config['k'],
                       config['a'],
                       config['random_start'],
                       x_data,
                       y_data)
        x_batch_adv = attack.perturb(x_data, y_data)
        x_batch_adv = paddle.to_tensor(x_batch_adv, dtype='float32')
        predicts = modelA(x_batch_adv)
        # 获取预测结果
        loss = F.cross_entropy(predicts, y_data)
        acc = paddle.metric.accuracy(predicts, y_data)
        acc_adv = acc_adv + acc.numpy()
        if batch_id % 20 == 0:
            print("adv batch_id: {}, loss is: {}, acc is: {}".format(batch_id, loss.numpy(), acc.numpy()))

if __name__ == '__main__':

    transform = Compose([Normalize(mean=[127.5],
                               std=[127.5],
                               data_format='CHW')])

    test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)

    # 加载配置文件
    with open('config.json') as config_file:
        config = json.load(config_file)

    #在使用GPU机器时，可以将use_gpu变量设置成True
    use_gpu = True
    paddle.set_device('gpu') if use_gpu else paddle.set_device('cpu')

    # 加载测试数据集
    test_loader = paddle.io.DataLoader(test_dataset, places=paddle.CPUPlace(), batch_size=64)

    acc_nat = 0
    acc_adv = 0
    num_exam = 0

    # 测试Mnist数据集上进行PGD-steps100-restarts20-sourceA: 89.3%
    model = ModelA()
    params_dict = paddle.load('mnist-PGD-steps100-restarts20-sourceA.pdparams')
    model.set_state_dict(params_dict)
    test(model)
    print("nat  acc is: {}".format(acc_nat/num_exam))
    print("adv  acc is: {}".format(acc_adv/num_exam))
    


    # 测试Mnist数据集上进行PGD-steps100-restarts20-sourceA: 95.7%
    acc_nat = 0
    acc_adv = 0
    num_exam = 0

    modelA = ModelA()
    modelB = ModelA()
    params_dictA = paddle.load('mnist-PGD-steps100-restarts20-sourceA-modelA-1.pdparams')
    params_dictB = paddle.load('mnist-PGD-steps100-restarts20-sourceA-modelB-1.pdparams')
    modelA.set_state_dict(params_dictA)
    modelB.set_state_dict(params_dictB)
    test2(modelA, modelB)
    print("nat  acc is: {}".format(acc_nat/num_exam))
    print("adv  acc is: {}".format(acc_adv/num_exam))

    # 测试Mnist数据集上进行PGD-steps40-restarts1-sourceB: 96.4%%

    acc_nat = 0
    acc_adv = 0
    num_exam = 0
    
    modelA = ModelA()
    modelB = ModelB()
    params_dictA = paddle.load('mnist-PGD-steps100-restarts20-sourceA-modelA-2.pdparams')
    params_dictB = paddle.load('mnist-PGD-steps100-restarts20-sourceA-modelB-2.pdparams')
    modelA.set_state_dict(params_dictA)
    modelB.set_state_dict(params_dictB)
    test2(modelA, modelB)
    print("nat  acc is: {}".format(acc_nat/num_exam))
    print("adv  acc is: {}".format(acc_adv/num_exam))

