import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import os

class MyDataset(Dataset):
    def __init__(self, data, labels):
        """Reads source and target sequences from processing file ."""
        self.input_tensor = (torch.from_numpy(data)).float()

        self.label = (torch.torch.FloatTensor(labels))
        self.num_total_seqs = len(self.input_tensor)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        input_seq = self.input_tensor[index]
        input_labels = self.label[index]
        return input_seq, input_labels

    def __len__(self):
        return self.num_total_seqs


def create_dataset(data, batch_size, shuffle, drop_last):
    trainX, validX, testX, trainY, validY, testY = data
    train_dl = DataLoader(MyDataset(trainX, trainY), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    valid_dl = DataLoader(MyDataset(validX, validY), batch_size=10, shuffle=False, drop_last=False)
    test_dl = DataLoader(MyDataset(testX, testY), batch_size=10, shuffle=False, drop_last=False)
    return train_dl, valid_dl, test_dl


def create_dataset_full(data, batch_size=10, shuffle=True, drop_last=True):
    trainX, testX, trainY, testY = data
    print('training data size: ', trainX.shape, 'test data size: ', testX.shape)
    train_dl = DataLoader(MyDataset(trainX, trainY), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    test_dl = DataLoader(MyDataset(testX, testY), batch_size=10, shuffle=False, drop_last=False)
    return train_dl, test_dl

class Load_Dataset(Dataset):
    def __init__(self, dataset, dataset_configs):
        super().__init__()
        self.num_channels = dataset_configs.input_channels
        
        # Load samples
        x_data = dataset["samples"]

        # Load labels
        y_data = dataset.get("labels")
        if y_data is not None and isinstance(y_data, np.ndarray):
            y_data = torch.from_numpy(y_data)
        
        # Convert to torch tensor
        if isinstance(x_data, np.ndarray):
            x_data = torch.from_numpy(x_data)
        
        # Check samples dimensions.
        # The dimension of the data is expected to be (N, C, L)
        # where N is the #samples, C: #channels, and L is the sequence length
        if len(x_data.shape) == 2:
            x_data = x_data.unsqueeze(1)
        elif len(x_data.shape) == 3 and x_data.shape[1] != self.num_channels:
            x_data = x_data.transpose(1, 2)
        
        
        if dataset_configs.permute:
            x_data = x_data.transpose(1, 2)
        
        self.x_data = x_data.float()
        self.y_data = y_data.float() if y_data is not None else None
        self.len = x_data.shape[0]
         

    def __getitem__(self, index):
        x = self.x_data[index]
        y = self.y_data[index] if self.y_data is not None else None
        return x, y

    def __len__(self):
        return self.len


def data_generator(data_path, domain_id, dataset_configs, hparams, dtype):
    # loading dataset file from path
    dataset_file = torch.load(os.path.join(data_path, f"{dtype}_{domain_id}.pt"))
    print(dataset_file['samples'].shape, f"{dtype}_{domain_id}.pt")
    # Loading datasets
    dataset = Load_Dataset(dataset_file, dataset_configs)
    if dtype == "test":  # you don't need to shuffle or drop last batch while testing
        shuffle  = False
        drop_last = False
    else:
        shuffle = dataset_configs.shuffle
        drop_last = dataset_configs.drop_last

    # Dataloaders
    data_loader = torch.utils.data.DataLoader(dataset=dataset, 
                                              batch_size=hparams["batch_size"],
                                              shuffle=shuffle, 
                                              drop_last=drop_last, 
                                              num_workers=0)

    return data_loader
