#train.py
from torch.utils.data import DataLoader

class ModelTrainer():

    def __init__(self, output_dir, optimizer, scheduler, lr, weight_decay, train_batch, test_batch):
        
        self.output_dir= output_dir,
        self.optimizer= optimizer,
        self.scheduler= scheduler,
        self.learning_rate= lr,
        self.weight_decay= weight_decay, 
        self.train_batch_size = train_batch,
        self.test_batch_size = test_batch

    def getOptimizer(self):
        if self.optimizer == "Adam":
            pass
        elif self.optimizer == "SGD":
            pass

    def getScheduler(self):
        pass

    def calcLoss(self):
        pass

    def validate(self):
        pass

    def loadCheckpoint(self):
        pass

    def saveCheckpoint(self):
        pass

    def getDataloaders(self):

        train_dataloader = DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True)
        test_dataloader  = DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=True)

        return (train_dataloader, test_dataloader)

    def train(self, epochs, resume_checkpoint_file, save_every ,train_dataset ,test_dataseet):
        
        
        train_dataloader, test_dataloader = self.getDataloaders()
        optimizer = self.getOptimizer()

        for epoch in range(epochs):

            running_loss = 0.0
            
            for i, x in enumerate(train_dataloader): 
                
                optimizer.zero_grad()

                print(f'Epoch {epoch}/{num_epochs} , Step {i}/{len(train_dataloader)} ')
                # print(f'accuracy : { metrics_dict["accuracy"] } precision : { metrics_dict["precision"] } recall : { metrics_dict["recall"] } f1 : { metrics_dict["f1"] }')

            epoch_loss = running_loss / train_dataset.__len__()
            # writer.add_scalar('Loss/train', epoch_loss, epoch)
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, epoch_loss))

