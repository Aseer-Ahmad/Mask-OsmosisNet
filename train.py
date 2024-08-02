#train.py
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam, SGD
import torch

import os

class ModelTrainer():

    def __init__(self, output_dir, optimizer, scheduler, lr, weight_decay, train_batch, test_batch):
        
        self.output_dir= output_dir,
        self.optimizer= optimizer,
        self.scheduler= scheduler,
        self.learning_rate= lr,
        self.weight_decay= weight_decay, 
        self.train_batch_size = train_batch,
        self.test_batch_size = test_batch

    def getOptimizer(self, model):

        if self.optimizer == "Adam":
            opt = Adam(model.parameters(), lr=self.lr, weight_decay = self.weight_decay)

        elif self.optimizer == "SGD":
            opt = SGD(model.parameters(), lr=self.lr, momentum = self.weight_decay)
              
        return opt

    def getScheduler(self):
        pass

    def calcLoss(self):
        pass

    def validate(self):
        pass

    def loadCheckpoint(self, model, optimizer, path):
        
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return model, optimizer


    def saveCheckpoint(self, model, optimizer):
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(self.output_dir, "ckp.pt"))
        

    def getDataloaders(self):

        train_dataloader = DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True)
        test_dataloader  = DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=True)

        return (train_dataloader, test_dataloader)

    def train(self, epochs, resume_checkpoint_file, save_every ,train_dataset ,test_dataseet):
        
        train_dataloader, test_dataloader = self.getDataloaders()
        optimizer = self.getOptimizer()
        scheduler = self.getScheduler()

        if resume_checkpoint_file != None:
            model, optimizer = self.loadCheckpoint(model, optimizer, resume_checkpoint_file)
        
        model.train()

        print("\nmodel, opt, schdl loaded")
        print("\nbeginning training ...")

        for epoch in range(epochs):

            running_loss = 0.0
            
            for i, x in enumerate(train_dataloader): 
                
                

                print(f'Epoch {epoch}/{epochs} , Step {i}/{len(train_dataloader)} ')

            epoch_loss = running_loss / train_dataset.__len__()
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, epoch_loss))

