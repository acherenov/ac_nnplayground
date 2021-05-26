"""
Генерация синтетических данных
"""
from sklearn import datasets
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class Data_generator(Dataset):
    def __init__(self, data_type, n_samples, shuffle, random_state=0, centers = 2, factor = 0.8, noise = 0.1):
        if data_type == "blobs":            
            self.X, self.y = datasets.make_blobs(n_samples=n_samples, shuffle=shuffle,
                                               random_state=random_state, centers = centers)
        elif data_type == "circles":
            self.X, self.y = datasets.make_circles(n_samples=n_samples, shuffle=shuffle, noise=noise,
                                               random_state=random_state,
                                               factor=factor)
        else:
            self.X, self.y = datasets.make_moons(n_samples=n_samples, shuffle=shuffle,
                                               noise=noise, random_state=random_state)
        st_sc = StandardScaler()
        self.X = st_sc.fit_transform(self.X)
        self.X, self.y = self.X.astype(np.float32), self.y.astype(np.int)
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(np.array(self.y[idx]))

    def plot_data(self):
        plt.figure(figsize=(8, 8))
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y)
        plt.show()


if __name__ == "__main__":
    dataset = Data_generator(n_samples=5000, shuffle=True, random_state=0, data_type="circles")
    dataset.plot_data()