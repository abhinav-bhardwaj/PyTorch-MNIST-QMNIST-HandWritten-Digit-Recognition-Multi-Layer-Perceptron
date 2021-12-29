import torch.nn as nn
import torch.nn.functional as F

class MultiLayerPerceptron(nn.Module):
  def __init__(self, input_size=784, output_size=10, layers=[120, 84]):
    super().__init__()
    self.d1 = nn.Linear(input_size, layers[0])
    self.d2 = nn.Linear(layers[0], layers[1])
    self.d3 = nn.Linear(layers[1], output_size)

  def forward(self, X):
    X = F.relu(self.d1(X))
    X = F.relu(self.d2(X))
    X = self.d3(X)
    return F.log_softmax(X, dim=1)
