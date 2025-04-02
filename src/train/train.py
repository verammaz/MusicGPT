import os
import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader
from collections import defaultdict
from ..utils.general_utils import CfgNode as CN


class Trainer():

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'auto'
        # training parameters
        C.epochs = 10
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0

        return C
    
    def __init__(self, config, model, dataloader):
        self.config = config
        self.dataloader = dataloader
        self.model = model
        self.optimizer = model.configure_optimizers(config)
        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("Train", self.device)

        self.n_examples = 0
        self.n_iter = 0
        self.n_epoch = 0
        self.best_loss = np.inf
        self.loss_improved = False

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)
            

    def run(self):
        model = self.model
        config = self.config

        model.train()

        for epoch in range(config.epochs):

            for batch, encodings in enumerate(self.dataloader):
                self.optimizer.zero_grad()

                for k, v in encodings.items():
                    encodings[k] = v.to(self.device)
                
                self.loss, logits, *_ = self.model(**encodings) 
                model.zero_grad(set_to_none=True)
                self.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                self.optimizer.step()
                
                self.n_examples += self.dataloader.batch_size

                self.trigger_callbacks('on_batch_end')
                
                if batch % 200 == 0:
                    print(f'epoch: {epoch + 1}, batch: {batch + 1}, loss: {self.loss.item()}')
                
                self.n_iter += 1

            self.n_epoch += 1
            
              


