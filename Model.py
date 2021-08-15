import paddle
from paddle.nn import Linear
import paddle.nn.functional as F


# 定义mnist数据识别网络结构
class ModelA(paddle.nn.Layer):
    def __init__(self):
        super(ModelA, self).__init__()
        
        self.conv1 = paddle.nn.Conv2D(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding='SAME')
        self.max_pool1 = paddle.nn.MaxPool2D(kernel_size=2, stride=2, padding='SAME')
        self.conv2 = paddle.nn.Conv2D(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding='SAME')
        self.max_pool2 = paddle.nn.MaxPool2D(kernel_size=2, stride=2, padding='SAME')
                
        self.linear1 = paddle.nn.Linear(in_features=7*7*64, out_features=1024)
        self.linear2 = paddle.nn.Linear(in_features=1024, out_features=10)
        self.soft = paddle.nn.Softmax()
        

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)

        x = paddle.flatten(x, start_axis=1,stop_axis=-1)
        x = self.linear1(x)
        x = F.relu(x)

        x = self.linear2(x)
       
        x = self.soft(x)
        return x


# 定义mnist数据识别网络结构
class ModelB(paddle.nn.Layer):
    def __init__(self):
        super(ModelB, self).__init__()
        self.drop1 = paddle.nn.Dropout(0.2)
        self.conv1 = paddle.nn.Conv2D(in_channels=1, out_channels=64, kernel_size=8, stride=1)
        
        self.conv2 = paddle.nn.Conv2D(in_channels=64, out_channels=128, kernel_size=6, stride=1)
        self.conv3 = paddle.nn.Conv2D(in_channels=128, out_channels=128, kernel_size=5, stride=1)
        self.drop2 = paddle.nn.Dropout(0.5)
        
        self.linear1 = paddle.nn.Linear(in_features=128*12*12, out_features=10)
        self.soft1 = paddle.nn.Softmax()
        

    def forward(self, x):
        x = self.drop1(x)
        x = self.conv1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = self.drop2(x)

        x = paddle.flatten(x, start_axis=1,stop_axis=-1)
        x = self.linear1(x)
       
        x = self.soft1(x)
        return x