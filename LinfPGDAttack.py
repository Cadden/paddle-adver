import paddle
from paddle.nn import Linear
import paddle.nn.functional as F
import numpy as np

class LinfPGDAttack:
  def __init__(self, model, epsilon, k, a, random_start, x_data, y_data):
    """Attack parameter initialization. The attack performs k steps of
       size a, while always staying within epsilon from the initial
       point."""
    self.model = model
    self.epsilon = epsilon
    self.k = k
    self.a = a
    self.rand = random_start
    self.x_data = x_data
    self.y_data = y_data
        

  def perturb(self, x_nat, y):
    """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_infinity norm."""
    if self.rand:
      x_nat = x_nat.numpy()
      x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
      x = np.clip(x, 0, 1) # ensure valid pixel range
    else:
      x = np.copy(x_nat)    
    

    for i in range(self.k):
      x_nat = paddle.to_tensor(x_nat)
      x_nat.stop_gradient = False
      loss = F.cross_entropy(self.model(x_nat), y)
      grad = paddle.grad(outputs=loss,inputs=x_nat)[0]

      grad = grad.numpy()

      x += self.a * np.sign(grad)
      # print('-------x:', x)
      x_nat = x_nat.numpy()

      x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon) 
      x = np.clip(x, 0, 1) # ensure valid pixel range

    return x