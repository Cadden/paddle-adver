Implemention of the paper "Towards Deep Learning Models Resistant to Adversarial Attacks" using PaddlePaddle


1. 复现论文 “Towards Deep Learning Models Resistant to Adversarial Attacks” 
2. 运行方式：
 1) 使用python train.py命令进行模型训练  
 2）使用python eval.py命令进行测试（该步骤可以直接运行，预训练的参数已经保存在环境中）

2. 在Mnist数据集上复现结果如下：
 PGD-steps100-restarts20-sourceA: 90.89% (论文中结果89.3%),
 PGD-steps100-restarts20-sourceA:  97.53% (论文中结果95.7%),   
 PGD-steps40-restarts1-sourceB: 98.47% (论文中结果96.4%)

 3.  nat:表示原始图像，avd表示生成的对抗样本图像
