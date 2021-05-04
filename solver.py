import os
import time
import datetime

import torch
import torch.optim as optim
import torch.nn as nn
from model import Classifier

class Solver(object):
    def __init__(self,config, loader):

        # Loader
        self.loader = loader

        # Model configuration
        self.model = config.model

        # Training configuration
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.momentum = config.momentum
        self.EPOCH = config.EPOCH

        # Directories
        self.model_save_path = config.model_save_path

        # Steps
        self.model_save_step = config.model_save_step

        # Miscellaneous
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'GPU Availability: {torch.cuda.is_available()}')
        if torch.cuda.is_available():
            print(torch.cuda.get_device_properties(self.device))

        # Build model
        self.build_model()

    def build_model(self):
        self.net = Classifier(self.model).build_model().to(self.device)
        print('Net built')
        self.print_network(self.net)
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum)
        self.criterion = nn.CrossEntropyLoss()

    def print_network(self, net):
        num = 0
        for param in net.parameters():
            num += param.numel()
        print('The number of parameters is :{}'.format(num))

    def train(self):
        self.net.train()
        print('Start training...')
        start_time = time.time()

        for epoch in range(self.EPOCH):
            for batch_idx, (img, target) in enumerate(self.loader):
                img = img.to(self.device)
                target = target.to(self.device)
                preds = self.net(img)
                loss = self.criterion(preds, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                print(f'Elapsed time: {et}, Batch:{batch_idx}, epoch:{epoch+1}/{self.EPOCH}, Loss:{loss}')

            if (epoch+1) % self.model_save_step == 0:
                save_path = os.path.join(self.model_save_path, f'{epoch+1}.ckpt')
                torch.save(self.net.state_dict(), save_path)
                print(f'Save model check point into {save_path}')

    def test(self):
        self.model.eval()
        for batch_idx, (img, target) in enumerate(self.loader):
            img = img.to(self.device)
            preds = self.model(img)

