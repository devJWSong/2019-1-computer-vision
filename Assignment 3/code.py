import torch
from torch.utils.data import dataloader
from torchvision import datasets
import torch.nn as nn
import torchvision

i_size = 28*28
h_size1 = 256
h_size2 = 100
n_classes = 10
task1_epochs = 100
task2_epochs = 50

learning_rates = [0.5, 0.1, 0.05, 0.01, 0.005]
momentums = [0.9, 0.5, 0.2, 0.1]
loss_functions = [0, 1] # 0: MSELoss, 1: CrossEntropyLoss
batch_sizes = [50, 100, 200, 500, 1000]


fc = torch.nn.Sequential(
    torch.nn.Linear(i_size, h_size1),
    torch.nn.ReLU(),
    torch.nn.Linear(h_size1, h_size2),
    torch.nn.ReLU(),
    torch.nn.Linear(h_size2, n_classes)
)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 2)
        self.fc1 = nn.Linear(16*7*7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16*7*7)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)

        return x


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

fc = fc.to(device)


def get_data(b_size):
    train_data = datasets.FashionMNIST(root="./", download=True, train=True,
                                       transform=torchvision.transforms.ToTensor())
    test_data = datasets.FashionMNIST(root="./", download=True, train=False,
                                      transform=torchvision.transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=b_size)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=b_size)

    return train_loader, test_loader


def get_optimizer(cnn, learning_rate, momentum):
    optimizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate, momentum=momentum)
    return optimizer


def get_criterion(loss_function):
    if loss_function == 0:
        return nn.MSELoss()
    elif loss_function == 1:
        return nn.CrossEntropyLoss()


def train(cnn, category, subject, train_loader, optimizer, criterion):
    for epoch in range(1, task2_epochs+1):
        step = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            step += 1

            if not (category == 'loss_function' and subject == 1):
                labels = nn.functional.one_hot(labels, n_classes)
                if torch.cuda.is_available():
                    labels = labels.type(torch.cuda.FloatTensor)
                else:
                    labels = labels.type(torch.FloatTensor)

            output = cnn.forward(images)

            loss = criterion(output, labels)
            if step % 10 == 0:
                print("[" + category + " %.4f, step %d, epoch %d] %f\n" % (subject, step, epoch, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return cnn


def test(cnn, category, subject, test_loader):
    cnn.eval()
    correct = 0.0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        output = cnn(images)
        prediction = output.data.max(1)[1]
        correct += float(prediction.eq(labels.data).sum())

    print("<" + category + " " + str(subject) + "> Test Accuracy: {:.2f}%"
          .format(100.0 * correct / float(len(test_loader.dataset))))
    print("\n")


def task1_fc():
    train_loader, test_loader = get_data(batch_sizes[1])

    optimizer = torch.optim.SGD(fc.parameters(), lr=learning_rates[3], momentum=momentums[0])
    criterion = nn.MSELoss()
    for epoch in range(1, task1_epochs + 1):
        step = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            # pre-process
            step += 1
            images = images.view(batch_sizes[1], i_size)
            one_hot = nn.functional.one_hot(labels, n_classes)
            if torch.cuda.is_available():
                one_hot = one_hot.type(torch.cuda.FloatTensor)
            else:
                one_hot = one_hot.type(torch.FloatTensor)
            # forward
            output = fc.forward(images)
            # calculate loss
            loss = criterion(output, one_hot)
            print("[Step %d, epoch %d] %f\n" % (step, epoch, loss.item()))
            # back-propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    fc.eval()
    correct = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        images = images.view(batch_sizes[1], i_size)
        output = fc(images)
        prediction = output.data.max(1)[1]
        correct += prediction.eq(labels.data).sum()

    print("Test Accuracy: {:.2f}%".format(100.0 * correct / len(test_loader.dataset)))


def task1_cnn():
    train_loader, test_loader = get_data(batch_sizes[1])

    cnn = CNN().to(device)
    optimizer = get_optimizer(cnn, learning_rates[3], momentums[0])
    criterion = nn.MSELoss()
    for epoch in range(1, task1_epochs + 1):
        step = 0
        for images, labels in train_loader:
            # pre-process
            images = images.to(device)
            labels = labels.to(device)
            step += 1
            labels = nn.functional.one_hot(labels, n_classes)
            if torch.cuda.is_available():
                labels = labels.type(torch.cuda.FloatTensor)
            else:
                labels = labels.type(torch.FloatTensor)
            # forward
            output = cnn.forward(images)
            # calculate loss
            loss = criterion(output, labels)
            print("[Step %d, epoch %d] %f\n" % (step, epoch, loss.item()))
            # back-propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    cnn.eval()
    correct = 0.0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        output = cnn(images)
        prediction = output.data.max(1)[1]
        correct += float(prediction.eq(labels.data).sum())

    print("Test Accuracy: {:.2f}%".format(100.0 * correct / float(len(test_loader.dataset))))


def task2():
    compare_batch()
    compare_learning_rate()
    compare_momentum()
    compare_loss_function()


def compare_batch():
    for b_size in batch_sizes:
        cnn = CNN().to(device)
        train_loader, test_loader = get_data(b_size)
        optimizer = get_optimizer(cnn, learning_rates[3], momentums[0])
        criterion = get_criterion(loss_functions[0])

        trained_cnn = train(cnn, "batch_size", b_size, train_loader, optimizer, criterion)
        test(trained_cnn, "batch_size", b_size, test_loader)


def compare_learning_rate():
    for lr in learning_rates:
        cnn = CNN().to(device)
        train_loader, test_loader = get_data(batch_sizes[1])
        optimizer = get_optimizer(cnn, lr, momentums[0])
        criterion = get_criterion(loss_functions[0])

        trained_cnn = train(cnn, "learning_rate", lr, train_loader, optimizer, criterion)
        test(trained_cnn, "learning_rate", lr, test_loader)


def compare_momentum():
    for mm in momentums:
        cnn = CNN().to(device)
        train_loader, test_loader = get_data(batch_sizes[1])
        optimizer = get_optimizer(cnn, learning_rates[3], mm)
        criterion = get_criterion(loss_functions[0])

        trained_cnn = train(cnn, "momentum", mm, train_loader, optimizer, criterion)
        test(trained_cnn, "momentum", mm, test_loader)


def compare_loss_function():
    for lf in loss_functions:
        cnn = CNN().to(device)
        train_loader, test_loader = get_data(batch_sizes[1])
        optimizer = get_optimizer(cnn, learning_rates[3], momentums[0])
        criterion = get_criterion(lf)

        trained_cnn = train(cnn, "loss_function", lf, train_loader, optimizer, criterion)
        test(trained_cnn, "loss_function", lf, test_loader)


task1_fc()
task1_cnn()
task2()
