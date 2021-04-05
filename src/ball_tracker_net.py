import cv2
import torchvision
import torch.nn as nn
import numpy as np
import torch
import torch.optim as optim
import time

from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.datasets import TrackNetDataset, get_dataloaders
from src.load_batches import InputOutputGenerator
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from torch.optim.adadelta import Adadelta

from src.trainer import plot_graph


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pad, bias=True, bn=True):
        super().__init__()
        if bn:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad, bias=bias),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad, bias=bias),
                nn.ReLU()
            )

    def forward(self, x):
        return self.block(x)


class BallTrackerNet(nn.Module):
    def __init__(self, bn=True):
        super().__init__()
        layer_1 = ConvBlock(in_channels=9, out_channels=64, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_2 = ConvBlock(in_channels=64, out_channels=64, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        layer_4 = ConvBlock(in_channels=64, out_channels=128, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_5 = ConvBlock(in_channels=128, out_channels=128, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_6 = nn.MaxPool2d(kernel_size=2, stride=2)
        layer_7 = ConvBlock(in_channels=128, out_channels=256, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_8 = ConvBlock(in_channels=256, out_channels=256, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_9 = ConvBlock(in_channels=256, out_channels=256, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_10 = nn.MaxPool2d(kernel_size=2, stride=2)
        layer_11 = ConvBlock(in_channels=256, out_channels=512, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_12 = ConvBlock(in_channels=512, out_channels=512, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_13 = ConvBlock(in_channels=512, out_channels=512, kernel_size=3, pad=1, bias=True, bn=bn)

        self.encoder = nn.Sequential(layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, layer_7, layer_8, layer_9,
                                     layer_10, layer_11, layer_12, layer_13)

        layer_14 = nn.Upsample(scale_factor=2)
        layer_15 = ConvBlock(in_channels=512, out_channels=256, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_16 = ConvBlock(in_channels=256, out_channels=256, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_17 = ConvBlock(in_channels=256, out_channels=256, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_18 = nn.Upsample(scale_factor=2)
        layer_19 = ConvBlock(in_channels=256, out_channels=128, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_20 = ConvBlock(in_channels=128, out_channels=128, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_21 = nn.Upsample(scale_factor=2)
        layer_22 = ConvBlock(in_channels=128, out_channels=64, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_23 = ConvBlock(in_channels=64, out_channels=64, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_24 = ConvBlock(in_channels=64, out_channels=256, kernel_size=3, pad=1, bias=True, bn=bn)

        self.decoder = nn.Sequential(layer_14, layer_15, layer_16, layer_17, layer_18, layer_19, layer_20, layer_21,
                                     layer_22, layer_23, layer_24)

        self.softmax = nn.Softmax(dim=1)
        self._init_weights()

    def forward(self, x, testing=False):
        batch_size = x.size(0)
        features = self.encoder(x)
        scores_map = self.decoder(features)
        output = scores_map.reshape(batch_size, 256, -1)
        # output = output.permute(0, 2, 1)
        if testing:
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

    def inference(self, frames: torch.Tensor):
        self.eval()
        with torch.no_grad():
            if len(frames.shape) == 3:
                frames = frames.unsqueeze(0)
            if next(self.parameters()).is_cuda:
                frames.cuda()
            output = self(frames, True)
            output = output.argmax(dim=1).detach().cpu().numpy()
            x, y = self.get_center_ball(output)
        return x, y

    def get_center_ball(self, output):
        output = output.reshape((360, 640))

        # cv2 image must be numpy.uint8, convert numpy.int64 to numpy.uint8
        output = output.astype(np.uint8)

        # reshape the image size as original input image
        heatmap = cv2.resize(output, (640, 360))

        # heatmap is converted into a binary image by threshold method.
        ret, heatmap = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)

        # find the circle in image with 2<=radius<=7
        circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2,
                                   maxRadius=7)
        # check if there have any tennis be detected
        if circles is not None:
            # if only one tennis be detected
            if len(circles) == 1:
                x = int(circles[0][0][0])
                y = int(circles[0][0][1])

                return x, y
        return None, None


def accuracy(y_pred, y_true):
    correct = (y_pred == y_true).sum()
    acc = correct / (len(y_pred[0]) * y_pred.shape[0]) * 100
    non_zero = (y_true > 0).sum()
    non_zero_correct = (y_pred[y_true > 0] == y_true[y_true > 0]).sum()
    if non_zero == 0:
        if non_zero_correct == 0:
            non_zero_acc = 100.0
        else:
            non_zero_acc = 0.0
    else:

        non_zero_acc = non_zero_correct / non_zero * 100
    return acc, non_zero_acc, non_zero_correct


def show_result(inputs, labels, outputs):
    outputs = outputs.argmax(dim=1).detach().cpu().numpy()
    mask = outputs[0].reshape((360, 640))
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


def get_center_ball_dist(output, x_true, y_true):
    dists = []
    for i in range(len(x_true)):
        if x_true[i] == -1:
            dists.append(-2)
            continue
        Rx = 640 / 1280
        Ry = 360 / 720
        x_true[i] *= Rx
        y_true[i] *= Ry
        cur_output = output[i].reshape((360, 640))

        # cv2 image must be numpy.uint8, convert numpy.int64 to numpy.uint8
        cur_output = cur_output.astype(np.uint8)

        # reshape the image size as original input image
        heatmap = cv2.resize(cur_output, (640, 360))

        # heatmap is converted into a binary image by threshold method.
        ret, heatmap = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)

        # find the circle in image with 2<=radius<=7
        circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2,
                                   maxRadius=7)
        # check if there have any tennis be detected
        if circles is not None:
            # if only one tennis be detected
            if len(circles) == 1:
                x = int(circles[0][0][0])
                y = int(circles[0][0][1])
                # print('pred ', x, y)
                dist = np.linalg.norm((x_true[i] - x, y_true[i] - y))
                dists.append(dist)
                continue
        dists.append(-1)
    return dists


def train(model_saved_state=None, epochs_num=100, lr=1.0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device {device}')
    model = BallTrackerNet(bn=True)
    train_losses = []
    valid_losses = []
    train_acc = []
    valid_acc = []
    total_epochs = 0
    if model_saved_state is not None:
        saved_state = torch.load(model_saved_state)
        model.load_state_dict(saved_state['model_state'])
        train_losses = saved_state['train_loss']
        valid_losses = saved_state['valid_loss']
        train_acc = saved_state['train_acc']
        valid_acc = saved_state['valid_acc']
        total_epochs = saved_state['epochs']
        print('Loaded saved state')
    model.to(device)
    batch_size = 2
    train_dl, valid_dl = get_dataloaders('../dataset/Dataset/training_model2.csv', root_dir=None, transform=None,
                                         batch_size=batch_size, dataset_type='tracknet', num_workers=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adadelta(model.parameters(), lr=lr)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, verbose=True,
                                     min_lr=0.0001)

    for epoch in range(epochs_num):
        start_time = time.time()
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
                dl = train_dl
                steps_per_epoch = 200

            else:
                model.train(False)  # Set model to evaluate mode
                dl = valid_dl
                steps_per_epoch = 100
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
            for i, data in enumerate(dl):
                torch.cuda.empty_cache()
                '''print(f'AllocMem (Mb): '
                      f'{torch.cuda.memory_allocated() / 1024 / 1024}')'''
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data['frames'], data['gt']
                inputs = inputs.to(device)

                labels = labels.to(device)

                x_true = data['x_true']
                y_true = data['y_true']

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                if phase == 'train':
                    outputs = model(inputs)

                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                # print statistics
                running_loss += loss.item() * batch_size

                acc, non_zero_acc, non_zero = accuracy(outputs.argmax(dim=1).detach().cpu().numpy(),
                                                       labels.cpu().numpy())
                dists = get_center_ball_dist(outputs.argmax(dim=1).detach().cpu().numpy(), x_true, y_true)
                for j, dist in enumerate(dists.copy()):
                    if dist in [-1, -2]:
                        if dist == -1:
                            n1 += 1
                        else:
                            n2 += 1
                        dists[j] = np.inf
                    else:
                        running_dist += dist
                        count += 1

                min_dist = min(*dists, min_dist)
                running_acc += acc
                running_no_zero_acc += non_zero_acc * batch_size
                running_no_zero += non_zero * batch_size

                if (i + 1) % 100 == 0:
                    print('Phase {} Epoch {} Step {} Loss: {:.4f} Acc: {:.4f}%  Non zero acc: {:.4f}%  '
                          'Non zero: {}  Min Dist: {:.4f} Avg Dist {:.4f}'.format(phase, epoch + 1, i + 1,
                                                                                  running_loss / ((i + 1) * batch_size),
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
                        lr_scheduler.step(valid_losses[-1])
                    break
        total_epochs += 1
        print('Last Epoch time : {:.4f} min'.format((time.time() - start_time) / 60))
        if epoch % 5 == 4:
            inputs, labels = data['frames'], data['gt']
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                show_result(inputs, labels, outputs)

            PATH = f'saved states/tracknet_weights_lr_{lr}_epochs_{total_epochs}.pth'
            saved_state = dict(model_state=model.state_dict(), train_loss=train_losses, train_acc=train_acc,
                               valid_loss=valid_losses, valid_acc=valid_acc, epochs=total_epochs)
            torch.save(saved_state, PATH)
            print(f'*** Saved checkpoint ***')
    PATH = f'saved states/tracknet_weights_lr_{lr}_epochs_{total_epochs}.pth'
    saved_state = dict(model_state=model.state_dict(), train_loss=train_losses, train_acc=train_acc,
                       valid_loss=valid_losses, valid_acc=valid_acc, epochs=total_epochs)
    torch.save(saved_state, PATH)
    print(f'*** Saved checkpoint ***')
    print('Finished Training')
    plot_graph(train_losses, valid_losses, 'loss', f'../report/tracknet_losses_{total_epochs}_epochs.png')
    plot_graph(train_acc, valid_acc, 'acc', f'../report/tracknet_acc_{total_epochs}_epochs.png')

    '''# Test set
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
                                                                                              running_no_zero))'''


if __name__ == "__main__":
    '''state = torch.load('saved states/tracknet_weights_lr_1.0_epochs_115.pth')
    plot_graph(state['train_loss'], state['valid_loss'], 'loss', '../report/tracknet_losses_115_epochs.png')
    plot_graph(state['train_acc'], state['valid_acc'], 'acc', '../report/tracknet_acc_115_epochs.png')'''
    start = time.time()
    for lr in [1.0]:
        s = time.time()
        print(f'Start training with LR = {lr}')
        train(epochs_num=130, lr=lr)
        print(f'End training with LR = {lr}, Time = {time.time() - s}')
    print(f'Finished all runs, Time = {time.time() - start}')
