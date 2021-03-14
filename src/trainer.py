import torch
import torch.nn as nn
from torchvision.transforms import ToTensor

from src.datasets import create_train_valid_test_datasets
from src.shot_recognition import LSTM_model

from src.utils import get_dtype
import time
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from skimage import io, transform
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import cv2


class Trainer:
    def __init__(self, model, train_dl, valid_dl, lr=0.001, reg=0.003):
        # Using cuda if possible
        self.dtype = get_dtype()

        # Model
        self.model = model

        # Dataset and data loaders
        self.train_dl = train_dl
        self.valid_dl = valid_dl

        # Optimizer and schedule
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=reg)
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.3, patience=3, verbose=True,
                                              min_lr=1e-8)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Extras
        self.softmax = nn.Softmax(dim=1)
        self.saved_state_name = 'saved_state'
        print(f'Learning rate = {lr}')

    def train(self, epochs=1):
        start = time.time()

        self.model.type(self.dtype)

        train_loss, valid_loss = [], []
        train_acc, valid_acc = [], []

        best_acc = 0.0

        for epoch in range(1, epochs + 1):
            print('Epoch {}/{}'.format(epoch, epochs))
            print('-' * 10)
            flag = True
            for phase in ['train', 'valid']:
                if phase == 'train':
                    self.model.train(True)  # Set training mode = true
                    dataloader = self.train_dl
                else:
                    self.model.train(False)  # Set model to evaluate mode
                    dataloader = self.valid_dl

                running_loss = 0.0
                running_acc = 0.0

                step = 0

                # iterate over data
                for sample_batched in dataloader:
                    x = sample_batched['features'].type(self.dtype)
                    y = sample_batched['gt'].type(self.dtype)
                    step += 1

                    # forward pass
                    if phase == 'train':
                        # zero the gradients
                        self.optimizer.zero_grad()
                        outputs = self.model(x)

                        loss = self.loss_fn(outputs, y.long())

                        # the backward pass frees the graph memory, so there is no
                        # need for torch.no_grad in this training pass
                        loss.backward()
                        self.optimizer.step()
                        # scheduler.step()

                    else:
                        with torch.no_grad():
                            outputs = self.model(x)
                            loss = self.loss_fn(outputs, y.long())

                    # stats - whatever is the phase

                    y_pred = np.argmax(self.softmax(outputs).detach().cpu().numpy(), axis=1)
                    acc = accuracy_score(y.detach().cpu().numpy(), y_pred)

                    running_acc += acc * dataloader.batch_size
                    running_loss += loss.item() * dataloader.batch_size

                    if step % 300 == 0:
                        # clear_output(wait=True)
                        print(f'Current step: {step}  Loss: {loss.item()}  Acc: {acc} '
                              f'AllocMem (Mb): '
                              f'{torch.cuda.memory_allocated() / 1024 / 1024} '
                              f'Prediction: {y_pred}  real: {y}')

                        # print(torch.cuda.memory_summary())

                epoch_loss = running_loss / len(dataloader.dataset)
                epoch_acc = running_acc / len(dataloader.dataset)

                print('Epoch {}/{}'.format(epoch, epochs))
                print('-' * 10)
                print('{} Loss: {:.4f} Acc: {}'.format(phase, epoch_loss, epoch_acc))
                print('-' * 10)

                train_loss.append(epoch_loss) if phase == 'train' else valid_loss.append(epoch_loss)
                train_acc.append(epoch_acc) if phase == 'train' else valid_acc.append(epoch_acc)

                if phase == 'valid':
                    self.lr_scheduler.step(epoch_loss)

        time_elapsed = time.time() - start
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        saved_state = dict(model_state=self.model.state_dict(), train_loss=train_loss, train_acc=train_acc,
                           valid_loss=valid_loss, valid_acc=valid_acc)
        torch.save(saved_state, 'saved states/' + self.saved_state_name)
        print(f'*** Saved checkpoint ***')
        # print(f'Finding best threshold:')
        # find_best_threshold(model, valid_dl)

        plot_graph(train_loss, valid_loss, 'loss', f'../report/losses.png')
        plot_graph(train_acc, valid_acc, 'accuracy', f'../report/accuracy.png')

        return train_loss, valid_loss, train_acc, valid_acc


def plot_graph(train_data, valid_data, data_type, destination):
    plt.figure(figsize=(10, 8))
    plt.plot(train_data, label=f'Train {data_type}')
    plt.plot(valid_data, label=f'Valid {data_type}')
    plt.legend()
    plt.savefig(destination)
    plt.show()


def evaluate_performance(model, test_dl):
    is_cuda = next(model.parameters()).is_cuda
    dtype = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor
    softmax = nn.Softmax(dim=1)

    model.train(False)
    acc = 0
    for sample_batched in test_dl:
        x = sample_batched['features'].type(dtype)
        y = sample_batched['gt'].type(dtype)

        with torch.no_grad():
            outputs = model(x)
            outputs = softmax(outputs)
            y_pred = torch.argmax(outputs, 1)
            acc += accuracy_score(y.detach().cpu().numpy(), y_pred.detach().cpu().numpy())

    accuracy = acc / len(test_dl.dataset)
    print(f'Test accuracy = {accuracy}')


def train():
    dtype = get_dtype()
    batch_size = 1

    train_ds, valid_ds, test_ds = create_train_valid_test_datasets('../dataset/THETIS/VIDEO_RGB/THETIS_data.csv',
                                                                   '../dataset/THETIS/VIDEO_RGB/',
                                                                   )
    print(f'Train size : {len(train_ds)}, Validation size : {len(valid_ds)}')
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

    for lr in [0.001, 0.00003, 0.00005, 0.0001]:
        model = LSTM_model(3, dtype=dtype)
        model.type(dtype)
        trainer = Trainer(model, train_dl, valid_dl, lr=lr)
        trainer.train(30)
        print('Test accuracy')
        evaluate_performance(model, test_dl)


if __name__ == "__main__":
    train()
