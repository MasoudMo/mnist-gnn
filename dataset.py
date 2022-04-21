from abc import ABC
import pandas as pd
import os
from torch_geometric.data import Dataset
from torch_geometric.utils import from_networkx
import torch
import networkx as nx
import numpy as np


class MNISTPixelGraphDataset(Dataset, ABC):
    def __init__(self,
                 path_to_dataset: str,
                 train: bool = True):

        super().__init__()

        if train:
            data_df = pd.read_csv(os.path.join(path_to_dataset, 'mnist_train.csv'))
        else:
            data_df = pd.read_csv(os.path.join(path_to_dataset, 'mnist_test.csv'))

        # Extract the labels
        self.labels = torch.tensor(data_df['label'].tolist(), dtype=torch.long)

        # Find the proportion of each digit in the set
        self.class_weights = 1 / np.unique(self.labels, return_counts=True)[1]
        self.class_weights = self.class_weights[self.labels]

        # Extract the image data
        self.data = torch.tensor(data_df.loc[:, data_df.columns != 'label'].values, dtype=torch.float)

        # Create the pixel graph structure
        self.nx_graph = nx.grid_graph((28, 28))

        # Length of the dataset
        self.num_samples = self.data.shape[0]

    def get(self, idx):

        # Retrieve the sample
        sample_image = self.data[idx].view(28, 28).unsqueeze(0).unsqueeze(0)

        # Retrieve the label
        label = self.labels[idx]

        # Create the PyG data from the graph structure
        g = from_networkx(self.nx_graph)

        # Add data and label to the PyG data
        g.y = label
        g.image = sample_image
        g.nx_graph = nx.grid_graph((28, 28))

        return g

    def len(self) -> int:
        return self.num_samples