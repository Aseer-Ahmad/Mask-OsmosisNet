#train.py
from torch.utils.data import DataLoader

class ModelTrainer():

    def __init__(self):
        pass

    def getOptimizer():
        pass

    def getScheduler():
        pass

    def calcLoss():
        pass

    def validate():
        pass

    def loadCheckpoint():
        pass

    def saveCheckpoint():
        pass

    def getDataloaders(self):
        train_dataloader = DataLoader(self.train_dataset, batch_size=train_batch_size, shuffle=True)
        test_dataloader  = DataLoader(self.test_dataset, batch_size=test_batch_size, shuffle=True)


    def train(epochs, resume_checkpoint_file, save_every ,train_dataset ,test_dataseet):
        
        for epoch in range(num_epochs):

            running_loss = 0.0
            
            for i, x in enumerate(train_dataloader): 
                
                optimizer.zero_grad()

                print(f'Epoch {epoch}/{num_epochs} , Step {i}/{len(train_dataloader)} ')
                # print(f'accuracy : { metrics_dict["accuracy"] } precision : { metrics_dict["precision"] } recall : { metrics_dict["recall"] } f1 : { metrics_dict["f1"] }')

            epoch_loss = running_loss / train_dataset.__len__()
            # writer.add_scalar('Loss/train', epoch_loss, epoch)
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, epoch_loss))

