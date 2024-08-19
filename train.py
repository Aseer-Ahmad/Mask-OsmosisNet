#train.py
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, LambdaLR
import torch
import torch.nn as nn

import os

from InpaintingSolver.bi_cg import OsmosisInpainting

class MaskLoss(nn.Module):
    """
    Inverse variance loss 
    """
    def __init__(self):
        super(MaskLoss, self).__init__()

    def forward(self, X, eps = 1e-6):
        return torch.mean(1. / (torch.var(X, dim=(2,3)) + eps) )

class ModelTrainer():

    def __init__(self, output_dir, optimizer, scheduler, lr, weight_decay, train_batch_size, test_batch_size):
        
        self.output_dir= output_dir
        self.optimizer= optimizer
        self.scheduler= scheduler
        self.lr= lr
        self.weight_decay= weight_decay
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def getOptimizer(self, model):
        
        opt = None
        
        if self.optimizer == "Adam":
            opt = Adam(model.parameters(), lr=self.lr, weight_decay = self.weight_decay)

        elif self.optimizer == "SGD":
            opt = SGD(model.parameters(), lr=self.lr, momentum = self.weight_decay)
              
        return opt

    def getScheduler(self, optim):
        
        schdl = None

        if self.scheduler == "exp":
            schdl = ExponentialLR(optim, gamma=0.9)
        
        elif self.scheduler == "multiStep":
            schdl = MultiStepLR(optim, milestones=[30,80], gamma=0.1)

        elif self.scheduler == "multiStep":
            lambda1 = lambda epoch: epoch // 30
            lambda2 = lambda epoch: 0.95 ** epoch
            schdl = LambdaLR(optim, lr_lambda=[lambda1, lambda2])

        return schdl
    

    def loadCheckpoint(self, model, optimizer, path):
        
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return model, optimizer


    def saveCheckpoint(self, model, optimizer, epoch):
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(self.output_dir, f"ckp_epoch_{str(epoch)}.pt"))
        

    def getDataloaders(self, train_dataset, test_dataset):

        train_dataloader = DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=True)
        test_dataloader  = DataLoader(test_dataset, batch_size=self.test_batch_size, shuffle=True)

        print(f"train and test dataloaders created")
        print(f"total train batches  : {len(train_dataloader)}")
        print(f"total test  batches  : {len(test_dataloader)}")

        return train_dataloader, test_dataloader

    def validate(self, model, test_dataset):
        pass

    def train(self, model, epochs, resume_checkpoint_file, save_every , val_every, train_dataset ,test_dataset):
        
        train_dataloader, test_dataloader = self.getDataloaders(train_dataset, test_dataset)

        optimizer = self.getOptimizer(model)
        scheduler = self.getScheduler(optimizer)
        print(f"optimizer : {self.optimizer}, scheduler : {self.scheduler} loaded")

        if resume_checkpoint_file != None:
            print(f"loading checkpoint file : {resume_checkpoint_file}")
            model, optimizer = self.loadCheckpoint(model, optimizer, resume_checkpoint_file)
            print(f"model, opt, schdl loaded from checkpoint")


        print(f"optimizer : {optimizer}")
        print(f"scheduler : {scheduler}")

        model = model.double()
        model.to(self.device)
        model.train()

        loss = MaskLoss()

        print("\nbeginning training ...")

        for epoch in range(epochs):

            running_loss = 0.0
            
            for i, X in enumerate(train_dataloader): 
                
                optimizer.zero_grad()

                # output = model(X)                
                # print(output.shape)

                # loss(output)
                # loss.backward()

                # print(f"\nOsmosis solver for input : {X.shape}")
                # osmosis = OsmosisInpainting(None, X, None, None, offset=1, tau=10, device = self.device, apply_canny=False)
                # osmosis.calculateWeights(False, False, False)
                # osmosis.solveBatch(100, save_batch = False, verbose = False)
            
                loss.backward()
                optimizer.step()


                if (i+1) % save_every == 0:
                    print("saving checkpoint")
                    self.saveCheckpoint(model, optimizer, epoch)

                if (i+1) % val_every == 0:
                    print("validating on test dataset")
                    self.validate(model, test_dataset)

                print(f'Epoch {epoch}/{epochs} , Step {i}/{len(train_dataloader)} ')

            scheduler.step()

            epoch_loss = running_loss / train_dataset.__len__()
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, epoch_loss))

