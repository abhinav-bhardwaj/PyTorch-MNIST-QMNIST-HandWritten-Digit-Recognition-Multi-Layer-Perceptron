from digit_classifier_model import MultiLayerPerceptron
from torch.utils.data import DataLoader
from torch import nn, optim, save, no_grad
import torch
from torchvision import datasets, transforms

LR = 1e-3
BS = 64
EPOCHS = 10

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), ])

trainset = datasets.MNIST(
    r'..\input\MNIST', download=True, train=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=BS, shuffle=True)

testset = datasets.MNIST(r'..\input\MNIST', download=True,
                         train=False, transform=transform)
testloader = DataLoader(testset, batch_size=BS, shuffle=True)

model = SudokuNet()
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

image, label = next(iter(trainloader))
print("Image Shape : ",image.shape)

train_losses = []
test_losses = []
train_correct = []
test_correct = []

for e in range(EPOCHS):
    train_cc = 0
    test_cc = 0
    for batch_id, (X_train, y_train) in enumerate(trainloader):
        batch_id += 1
        y_pred = model(X_train.view(X_train.shape[0], -1))
        loss = criterion(y_pred, y_train)

        predicted = torch.max(y_pred.data, 1)[1]
        batch_cc = (predicted == y_train).sum()
        train_cc += batch_cc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("X___ : ",X_train.shape[0])
    print(f'epoch: {e:2} batch: {batch_id:4} [{batch_id*len(X_train):6}/60000] Train loss: {loss.item():10.8f} Train accuracy: {train_cc.item()/len(trainloader.dataset):5.4f}%')
    train_losses.append(loss)
    train_correct.append(train_cc)

    with no_grad():
        for batch_id, (X_test, y_test) in enumerate(testloader):
            y_val = model(X_test.view(X_test.shape[0], -1))
            predicted = torch.max(y_val.data, 1)[1]
            test_cc += (predicted == y_test).sum()

    loss = criterion(y_val, y_test)
    test_losses.append(loss)
    test_correct.append(test_cc)

save(model, 'mnist_model.pt')
print("Model saved")
