import torchvision
import torch.nn as nn
import numpy as np
import torch
import torch.optim as optim

from src.LoadBatches import InputOutputGenerator
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pad, bias=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad, bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.block(x)


class BallTrackerNet(nn.Module):
    def __init__(self):
        super().__init__()
        layer_1 = ConvBlock(in_channels=9, out_channels=64, kernel_size=3, pad=1, bias=True)
        layer_2 = ConvBlock(in_channels=64, out_channels=64, kernel_size=3, pad=1, bias=True)
        layer_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        layer_4 = ConvBlock(in_channels=64, out_channels=128, kernel_size=3, pad=1, bias=True)
        layer_5 = ConvBlock(in_channels=128, out_channels=128, kernel_size=3, pad=1, bias=True)
        layer_6 = nn.MaxPool2d(kernel_size=2, stride=2)
        layer_7 = ConvBlock(in_channels=128, out_channels=256, kernel_size=3, pad=1, bias=True)
        layer_8 = ConvBlock(in_channels=256, out_channels=256, kernel_size=3, pad=1, bias=True)
        layer_9 = ConvBlock(in_channels=256, out_channels=256, kernel_size=3, pad=1, bias=True)
        layer_10 = nn.MaxPool2d(kernel_size=2, stride=2)
        layer_11 = ConvBlock(in_channels=256, out_channels=512, kernel_size=3, pad=1, bias=True)
        layer_12 = ConvBlock(in_channels=512, out_channels=512, kernel_size=3, pad=1, bias=True)
        layer_13 = ConvBlock(in_channels=512, out_channels=512, kernel_size=3, pad=1, bias=True)

        self.encoder = nn.Sequential(layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, layer_7, layer_8, layer_9,
                                     layer_10, layer_11, layer_12, layer_13)

        layer_14 = nn.Upsample(scale_factor=2)
        layer_15 = ConvBlock(in_channels=512, out_channels=256, kernel_size=3, pad=1, bias=True)
        layer_16 = ConvBlock(in_channels=256, out_channels=256, kernel_size=3, pad=1, bias=True)
        layer_17 = ConvBlock(in_channels=256, out_channels=256, kernel_size=3, pad=1, bias=True)
        layer_18 = nn.Upsample(scale_factor=2)
        layer_19 = ConvBlock(in_channels=256, out_channels=128, kernel_size=3, pad=1, bias=True)
        layer_20 = ConvBlock(in_channels=128, out_channels=128, kernel_size=3, pad=1, bias=True)
        layer_21 = nn.Upsample(scale_factor=2)
        layer_22 = ConvBlock(in_channels=128, out_channels=64, kernel_size=3, pad=1, bias=True)
        layer_23 = ConvBlock(in_channels=64, out_channels=64, kernel_size=3, pad=1, bias=True)
        layer_24 = ConvBlock(in_channels=64, out_channels=256, kernel_size=3, pad=1, bias=True)

        self.decoder = nn.Sequential(layer_14, layer_15, layer_16, layer_17, layer_18, layer_19, layer_20, layer_21,
                                     layer_22, layer_23, layer_24)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size(0)
        features = self.encoder(x)
        scores_map = self.decoder(features)
        output = scores_map.reshape(batch_size, 256, -1)
        # output = output.permute(0, 2, 1)
        if not self.training:
            output = self.softmax(output)
        return output


def accuracy(y_pred, y_true):
    correct = (y_pred == y_true).sum()
    acc = correct / len(y_pred[0]) * 100
    return acc


def show_result(inputs, labels, outputs):
    outputs = outputs.argmax(dim=1).detach().cpu().numpy()
    mask = outputs.reshape((360, 640))
    fig, ax = plt.subplots(1, 2, figsize=(20, 1 * 5))
    ax[0].imshow(inputs[0, :3, :, ].detach().cpu().numpy().transpose((1, 2, 0)))
    ax[0].set_title('Image')
    ax[1].imshow(labels[0].detach().cpu().numpy().reshape((360, 640)), cmap='gray')
    ax[1].set_title('gt')
    plt.show()
    plt.figure()
    plt.imshow(mask, cmap='gray')
    plt.title('Pred')
    plt.show()


def train(saved_state=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BallTrackerNet()
    if saved_state is not None:
        model.load_state_dict(torch.load(saved_state))
        print('Loaded saved state')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    batch_size = 1
    input_height, input_width = 360, 640
    output_height, output_width = 360, 640
    steps_per_epoch = 400
    epoch_num = 100
    model.train(True)
    for epoch in range(epoch_num):
        model.train()
        running_loss = 0.0
        running_acc = 0.0

        for i, data in enumerate(InputOutputGenerator('../dataset/Dataset/training_model2.csv', batch_size, 256,
                                                      input_height, input_width, output_height, output_width), 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            acc = accuracy(outputs.argmax(dim=1).detach().cpu().numpy(), labels.cpu().numpy())

            running_acc += acc
            if (i + 1) == steps_per_epoch or (i + 1) % 50 == 0:
                print('Epoch {} Step {} Loss: {:.4f} Acc: {:.4f}%'.format(epoch, i + 1, running_loss / (i + 1),
                                                                          running_acc / (i + 1)))
                break
        if epoch % 5 == 4:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            model.eval()
            with torch.no_grad():
                outputs = model(inputs)
                show_result(inputs, labels, outputs)

            PATH = './tracknet_weights.pth'
            torch.save(model.state_dict(), PATH)
            print('Saved state')
    PATH = './tracknet_weights.pth'
    torch.save(model.state_dict(), PATH)
    print('Finished Training')

    # Test set
    print('Start Testing')
    running_loss = 0.0
    running_acc = 0.0
    i = 0
    for i, data in enumerate(InputOutputGenerator('../dataset/Dataset/testing_model2.csv', batch_size, 256,
                                                  input_height, input_width, output_height, output_width), 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        model.train(False)
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            acc = accuracy(outputs.argmax(dim=1).detach().cpu().numpy(), labels.cpu().numpy())
            running_acc += acc * batch_size

            # print statistics
            running_loss += loss.item()
        if not i % 200:
            print('Test results Step {} Loss: {:.4f} Acc: {:.4f}%'.format(i + 1, running_loss / (i + 1),
                                                                          running_acc / (i + 1)))

    print('Test results Loss: {:.4f} Acc: {:.4f}%'.format(running_loss / (i + 1),
                                                          running_acc / (i + 1)))


train()
