from datetime import datetime

import torch
import numpy as np
from torch.nn import CrossEntropyLoss, MSELoss, L1Loss
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import Data_generator
from net import SimpleMLP
from visualize_utils import make_meshgrid, predict_proba_on_mesh_tensor, plot_predictions

import os

# detect the current working directory and print it


from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, model, lr, runname, optimizer="Adam", criterion="CrossEntropyLoss"):
        self.model = model
        # fixme select loss
        if criterion == "CrossEntropyLoss":
            self.criterion = CrossEntropyLoss()
        elif criterion == "MSELoss":
            self.criterion = MSELoss()
        elif criterion == "L1Loss":
            self.criterion = L1Loss()
        else: 
            print("Unsupported loss function")
            exit
        
        if optimizer == "Adadelta":
            self.optimizer = optim.Adadelta(self.model.parameters(), lr=lr)
        elif optimizer == "Adagrad":
            self.optimizer = optim.Adagrad(self.model.parameters(), lr=lr)
        elif optimizer == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        else: 
            print("Unsupported optimizer")
            exit
        
        cuda = torch.cuda.is_available()
        self.device = torch.device("coda:0" if cuda else "cpu")
        
        self.runname = runname
        
        #self.experiment_name = datetime.now().strftime("%y%m%d_%H%M%S")

        self.writer = SummaryWriter(self.runname + '/log')


    def fit(self, train_dataloader, n_epochs):
        self.model.train()
        for epoch in range(n_epochs):
            print("epoch: ", epoch)
            epoch_loss = 0
            for i, (x_batch, y_batch) in enumerate(train_dataloader):
                y_batch = torch.tensor(y_batch, dtype=torch.long)
                x_batch = torch.tensor(x_batch)
                #if(epoch == 0) and (i == 0):
                    #self.writer.add_graph(self.model, x_batch)
                self.optimizer.zero_grad()
                output = self.model(x_batch)
                loss = self.criterion(output, y_batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                self.writer.add_scalar('training_loss', epoch_loss / len(train_dataloader), epoch * len(train_dataloader) + i)
                if epoch < 3 and (i + 1) % 10 == 1: 
                    print("save image", epoch, " ", i)
                    train_dataset = train_dataloader.dataset
                    X_train, X_test, y_train, y_test = get_data_from_datasets(train_dataset, train_dataset)
                    xx, yy = make_meshgrid(X_train, X_test, y_train, y_test)
                    Z = predict_proba_on_mesh_tensor(self, xx, yy)
                    plot_title = self.runname + "/pictures/nn_predictions_{}_{}.png".format(str(epoch).rjust(3, '0'), str(i).rjust(3, '0'))
                    plot_predictions(xx, yy, Z, X_train=X_train, X_test=X_test,
                                     y_train=y_train, y_test=y_test,
                                     title=plot_title)
                elif epoch >= 3 and (epoch + 1) % 5 == 1 and i == len(train_dataloader) - 1:
                    print("save image", epoch, " ", i)
                    train_dataset = train_dataloader.dataset
                    X_train, X_test, y_train, y_test = get_data_from_datasets(train_dataset, train_dataset)
                    xx, yy = make_meshgrid(X_train, X_test, y_train, y_test)
                    Z = predict_proba_on_mesh_tensor(self, xx, yy)
                    plot_title = self.runname + "/pictures/nn_predictions_{}_{}.png".format(str(epoch).rjust(3, '0'), str(i).rjust(3, '0'))
                    plot_predictions(xx, yy, Z, X_train=X_train, X_test=X_test,
                                     y_train=y_train, y_test=y_test,
                                     title=plot_title)
                

    def predict(self, test_dataloader):
        all_outputs = torch.tensor([], dtype=torch.long)
        self.model.eval()
        with torch.no_grad():
            for i, (x_batch, y_batch) in enumerate(test_dataloader):
                output_batch = self.model(x_batch)
                _, predicted = torch.max(output_batch.data, 1)
                print(predicted)
                all_outputs = torch.cat((all_outputs, predicted), 0)
        return all_outputs

    def predict_proba(self, test_dataloader):
        all_outputs = torch.tensor([], dtype=torch.long)
        self.model.eval()
        with torch.no_grad():
            for i, (x_batch, y_batch) in enumerate(test_dataloader):
                output_batch = self.model(x_batch)
                all_outputs = torch.cat((all_outputs, output_batch), 0)
        return all_outputs

    def predict_proba_tensor(self, T):
        self.model.eval()
        with torch.no_grad():
            output = self.model(T)
        return output



def get_data_from_datasets(train_dataset, test_dataset):
    X_train = train_dataset.X.astype(np.float32)
    X_test = test_dataset.X.astype(np.float32)

    y_train = train_dataset.y.astype(np.int)
    y_test = test_dataset.y.astype(np.int)

    return X_train, X_test, y_train, y_test



if __name__ == "__main__":
    path = os.getcwd()
    path = path + "/pictures"
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)
        
    
    layers_list_example = [(2, 4, 'relu'), (4, 6, 'sigmoid'), (6, 2, 'tanh')]
    model = SimpleMLP(layers_list_example)

    trainer = Trainer(model, lr=0.01)
    print(trainer.device)

    train_dataset = Data_generator(n_samples=5000, shuffle=True, random_state=0, data_type="circles")
    test_dataset = Data_generator(n_samples=1000, shuffle=True, random_state=0, data_type="circles")

    print(train_dataset)


    train_dataloader = DataLoader(train_dataset, batch_size=50, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=50, shuffle=True)

    trainer.fit(train_dataloader, n_epochs=100)


    test_predicion_proba = trainer.predict_proba(test_dataloader)
    
    torch.save(model, '/models')
    
