"""
============= You don't need to modify this file for the assignment ===========
This file contains a custom dataset class for the digits dataset.
The dataset consists of 1797 8x8 images of digits from 0 to 9.
It is split into a training set of 1500 samples and a test set of 297 samples.
It is also normalized by default to have mean=0 and standard deviation=1.0.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
# for the digits dataset
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
# to create a custom torch dataset
from torch.utils.data import Dataset

class DigitsDataset(Dataset):
    def __init__(self, train=True, normalize=True):
        split = 1500
        self.normalize = normalize
        data, target = load_digits(return_X_y=True)
        data = torch.tensor(data, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.int64)

        if normalize:
            # normalize the data using torch -> mean=0, standard deviation=1.0
            # we use epsilon=1e-7 to avoid division by zero
            self.mean = data[0:split, :].mean(dim=0)
            self.std = data[0:split, :].std(dim=0)
            data = (data - self.mean) / (self.std + 1e-7)

        self.X = data[0:split, :] if train else data[split:, :]
        self.y = target[0:split] if train else target[split:]
        self.num_samples = self.X.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def show_statistics(self):
        print(f"Number of samples: {self.num_samples}")
        print(f"X: {self.X.shape}  |  y: {self.y.shape}")

        plt.figure(figsize=(10, 4))
        # Check that the train and test targets/labels are balanced within each set
        plt.subplot(1, 2, 1)
        plt.hist(self.y, bins=10, rwidth=0.9, range=(0, 10), align='left')
        plt.xticks(range(10))
        plt.title("Histogram of the labels")
        plt.xlabel("Digits")
        plt.ylabel("Count")

        # make a TSNE plot of the data in 2 dimensions
        X_tsne = TSNE(n_components=2).fit_transform(self.X, self.y)
        plt.subplot(1, 2, 2)
        plt.scatter(X_tsne[:,0], X_tsne[:,1], c=plt.cm.tab10(self.y))
        circle_size = 2.5 if self.num_samples > 1000 else 1
        for i in range(10):
            plt.gca().add_artist(plt.Circle(np.median(X_tsne[self.y==i], axis=0), circle_size, color='white'))
            plt.gca().text(*np.median(X_tsne[self.y==i], axis=0), str(i), ha='center', va='center', color=plt.cm.tab10(i))
        plt.axis('off')
        plt.title('t-SNE of the labels')
        plt.show()

    def unnormalized_sample(self, idx):
        x = self.X[idx]*(self.std+1e-7)+self.mean if self.normalize else self.X[idx]
        return x, self.y[idx]

    def plot_grid(self, rows=5, cols=5):
        fig, axs = plt.subplots(rows, cols, figsize=(cols, rows))
        indices = np.random.choice(self.num_samples, rows*cols, replace=False)
        for i in range(rows):
            for j in range(cols):
                idx = indices[i*cols+j]
                img, lbl = self.unnormalized_sample(idx)
                # axs[i,j].imshow(img.reshape(8,8), cmap='gray')
                axs[i,j].imshow(img.reshape(8,8), cmap=plt.cm.gray_r, interpolation='nearest')
                axs[i,j].axis('off')
                axs[i,j].set_title(f"Label: {lbl}")
        plt.suptitle("Samples from the 'Digits' dataset")
        plt.tight_layout()
        plt.show()