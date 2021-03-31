import cv2
import torchvision
import torch.nn as nn
import numpy as np
import torch
import torch.optim as optim
import time
from src.load_batches import InputOutputGenerator
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from torch.optim.adadelta import Adadelta


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
        self._init_weights()

    def forward(self, x):
        batch_size = x.size(0)
        features = self.encoder(x)
        scores_map = self.decoder(features)
        output = scores_map.reshape(batch_size, 256, -1)
        # output = output.permute(0, 2, 1)
        if not self.training:
            output = self.softmax(output)
        return output

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.uniform_(module.weight, -0.05, 0.05)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


def accuracy(y_pred, y_true):
    correct = (y_pred == y_true).sum()
    acc = correct / len(y_pred[0]) * 100
    non_zero = (y_true > 0).sum()
    non_zero_correct = (y_pred[y_pred > 0] == y_true[y_pred > 0]).sum()
    if non_zero == 0:
        non_zero_acc = 0.0
    else:

        non_zero_acc = non_zero_correct / non_zero * 100
    return acc, non_zero_acc, non_zero_correct


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


def get_center_ball_dist(output, gt):
    gt = gt.reshape((360, 640))
    output = output.reshape((360, 640))

    # cv2 image must be numpy.uint8, convert numpy.int64 to numpy.uint8
    gt = gt.astype(np.uint8)
    output = output.astype(np.uint8)

    # reshape the image size as original input image
    heatmap = cv2.resize(output, (640, 360))

    # heatmap is converted into a binary image by threshold method.
    ret, y_true_heatmap = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)
    ret, heatmap = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)

    # find the circle in image with 2<=radius<=7
    y_true_circles = cv2.HoughCircles(y_true_heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2,
                                      minRadius=2, maxRadius=7)
    circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2,
                               maxRadius=7)
    x_true = None
    y_true = None
    if y_true_circles is not None:
        # if only one tennis be detected
        if len(y_true_circles) == 1:
            x_true = int(y_true_circles[0][0][0])
            y_true = int(y_true_circles[0][0][1])
            #print('true ', x_true, y_true)
    # check if there have any tennis be detected
    if circles is not None:
        # if only one tennis be detected
        if len(circles) == 1:
            x = int(circles[0][0][0])
            y = int(circles[0][0][1])
            #print('pred ', x, y)
            if x_true is not None and y_true is not None:
                dist = int(np.linalg.norm((x_true - x, y_true - y)))
            else:
                dist = -2
            return dist
    return -1


def train(model_saved_state=None, epochs_num=100, lr=1.0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BallTrackerNet()
    train_losses = []
    valid_losses = []
    train_acc = []
    valid_acc = []
    if model_saved_state is not None:
        saved_state = torch.load(model_saved_state)
        model.load_state_dict(saved_state['model_state'])
        train_losses = saved_state['train_loss']
        valid_losses = saved_state['valid_loss']
        train_acc = saved_state['train_acc']
        valid_acc = saved_state['valid_acc']
        print('Loaded saved state')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adadelta(model.parameters(), lr=1.0)
    batch_size = 1
    input_height, input_width = 360, 640
    output_height, output_width = 360, 640

    for epoch in range(epochs_num):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
                generator = InputOutputGenerator('../dataset/Dataset/training_model2.csv', batch_size, 256,
                                                 input_height, input_width, output_height, output_width)
                steps_per_epoch = 400

            else:
                model.train(False)  # Set model to evaluate mode
                generator = InputOutputGenerator('../dataset/Dataset/validation_model2.csv', batch_size, 256,
                                                 input_height, input_width, output_height, output_width)
                steps_per_epoch = 200
            print(f'Starting Epoch {epoch + 1} Phase {phase}')
            running_loss = 0.0
            running_acc = 0.0
            running_no_zero_acc = 0.0
            running_no_zero = 0
            min_dist = np.inf
            running_dist = 0.0
            count = 1
            n1 = 0
            n2 = 0
            for i, data in enumerate(generator, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # print statistics
                running_loss += loss.item()
                acc, non_zero_acc, non_zero = accuracy(outputs.argmax(dim=1).detach().cpu().numpy(), labels.cpu().numpy())
                dist = get_center_ball_dist(outputs.argmax(dim=1).detach().cpu().numpy(), labels.cpu().numpy())
                if dist in [-1, -2]:
                    dist = np.inf
                    if dist == -1:
                        n1 += 1
                    else:
                        n2 += 1
                else:
                    running_dist += dist
                    count += 1

                min_dist = min(dist, min_dist)
                running_acc += acc
                running_no_zero_acc += non_zero_acc
                running_no_zero += non_zero

                if (i + 1) % 100 == 0:
                    print('Phase {} Epoch {} Step {} Loss: {:.4f} Acc: {:.4f}%  Non zero acc: {:.4f}%  '
                          'Non zero: {}  Min Dist: {:.4f} Avg Dist {:.4f}'.format(phase, epoch + 1, i + 1, running_loss / (i + 1),
                                                                                  running_acc / (i + 1),
                                                                                  running_no_zero_acc / (i + 1),
                                                                                  running_no_zero,
                                                                                  min_dist, running_dist / count))
                    print(f'n1 = {n1}  n2 = {n2}')
                if (i + 1) == steps_per_epoch:
                    if phase == 'train':
                        train_losses.append(running_loss / (i + 1))
                        train_acc.append(running_no_zero_acc / (i + 1))
                    else:
                        valid_losses.append(running_loss / (i + 1))
                        valid_acc.append(running_no_zero_acc / (i + 1))
                    break
        if epoch % 5 == 4:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                show_result(inputs, labels, outputs)

            PATH = f'saved states/tracknet_weights_{lr}.pth'
            saved_state = dict(model_state=model.state_dict(), train_loss=train_losses, train_acc=train_acc,
                               valid_loss=valid_losses, valid_acc=valid_acc)
            torch.save(saved_state, PATH)
            print(f'*** Saved checkpoint ***')
    PATH = f'saved states/tracknet_weights_{lr}.pth'
    saved_state = dict(model_state=model.state_dict(), train_loss=train_losses, train_acc=train_acc,
                       valid_loss=valid_losses, valid_acc=valid_acc)
    torch.save(saved_state, PATH)
    print(f'*** Saved checkpoint ***')
    print('Finished Training')

    # Test set
    print('Start Testing')
    running_loss = 0.0
    running_acc = 0.0
    running_no_zero_acc = 0.0
    running_no_zero = 0
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
            acc, non_zero_acc, non_zero = accuracy(outputs.argmax(dim=1).detach().cpu().numpy(), labels.cpu().numpy())
            dist = get_center_ball_dist(outputs.argmax(dim=1).detach().cpu().numpy(), labels.cpu().numpy())
            running_acc += acc
            running_no_zero_acc += non_zero_acc
            running_no_zero += non_zero
            # print statistics
            running_loss += loss.item()
            if dist is None:
                dist = np.inf
        if not (i + 1) % 200:
            print(
                'Test results Step {} Loss: {:.4f} Acc: {:.4f}%  Non zero acc: {:.4f}% Non zero: {} Dist: {:.4f}'.format(
                    i,
                    running_loss / (
                            i + 1),
                    running_acc / (
                            i + 1),
                    running_no_zero_acc / (
                            i + 1),
                    running_no_zero, dist))
        if i % 1000 == 999:
            break

    print('Test results Loss: {:.4f} Acc: {:.4f}%  Non zero acc: {:.4f}% Non zero: {}'.format(running_loss / (i + 1),
                                                                                              running_acc / (i + 1),
                                                                                              running_no_zero_acc / (
                                                                                                      i + 1),
                                                                                              running_no_zero))


if __name__ == "__main__":

    start = time.time()
    for lr in [1.0]:
        s = time.time()
        print(f'Start training with LR = {lr}')
        train(epochs_num=100, lr=lr)
        print(f'End training with LR = {lr}, Time = {time.time() - s}')
    print(f'Finished all runs, Time = {time.time() - start}')
