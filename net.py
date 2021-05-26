"""
Архитектура сети
"""
import torch
from torch import nn
from pathlib import Path

class SimpleMLP(nn.Module):
    def __init__(self, layers_list):
        super(SimpleMLP, self).__init__()
        self.layers_list = layers_list
        layers = []
        for i, layer in enumerate(layers_list):
            layers.append(torch.nn.Linear(layer[0], layer[1]))
            if (layer[2] == 'sigmoid'):
                layers.append(torch.nn.Sigmoid())
            elif (layer[2] == 'relu'):
                layers.append(torch.nn.ReLU())
            elif (layer[2] == 'tanh'):
                layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Softmax())
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
    def save_model(self, path): 
        torch.save(self.state_dict(), Path(path))
    
    def load_model(self, path): 
        self.load_state_dict(torch.load(path))
    
    def desc(self):
        for i, layer in enumerate(self.layers_list):
            print(f"Layer {i+1} has {layer[0]} inputs, {layer[1]} outputs and {layer[2]} activation function")

if __name__ == "__main__":
    layers_list_example = [(2, 4, 'relu'), (4, 7, 'sigmoid'), (7, 2, 'tanh')]
    example_net = SimpleMLP(layers_list_example)
    example_net.desc()





