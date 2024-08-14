#train.py
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam, SGD
import torch

import os

class ModelTrainer():

    def __init__(self, output_dir, optimizer, scheduler, lr, weight_decay, train_batch_size, test_batch_size):
        
        self.output_dir= output_dir,
        self.optimizer= optimizer,
        self.scheduler= scheduler,
        self.learning_rate= lr,
        self.weight_decay= weight_decay, 
        self.train_batch_size = train_batch_size,
        self.test_batch_size = test_batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        

    def getDataloaders(self, train_dataset, test_dataset):

        train_dataloader = DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=True)
        test_dataloader  = DataLoader(test_dataset, batch_size=self.test_batch_size, shuffle=True)

        return (train_dataloader, test_dataloader)

    def train(self, model, epochs, resume_checkpoint_file, save_every , val_every, train_dataset ,test_dataset):
        
        train_dataloader, test_dataloader = self.getDataloaders(train_dataset, test_dataset)
        print(f"train and test dataloaders created")
        print(f"train batches  : {len(train_dataloader)}")
        print(f"test  batches  : {len(test_dataloader)}")

        optimizer = self.getOptimizer()
        scheduler = self.getScheduler()
        print(f"optimizer : {self.optimizer}, scheduler : {self.scheduler} loaded")

        if resume_checkpoint_file != None:
            print(f"loading checkpoint file : {resume_checkpoint_file}")
            model, optimizer = self.loadCheckpoint(model, optimizer, resume_checkpoint_file)
            print(f"model, opt, schdl loaded from checkpoint")

        model.to(self.device)
        model.train()

        print("\nbeginning training ...")

        for epoch in range(epochs):

            running_loss = 0.0
            
            for i, x in enumerate(train_dataloader): 
                
                print(x.shape)
                

                print(f'Epoch {epoch}/{epochs} , Step {i}/{len(train_dataloader)} ')

            epoch_loss = running_loss / train_dataset.__len__()
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, epoch_loss))

