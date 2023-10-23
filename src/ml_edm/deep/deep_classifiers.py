import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm
from sklearn.model_selection import train_test_split

from modules import *

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False
        self.min_validation_loss = np.inf

    def __call__(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
 
class DeepChronologicalClassifier:

    def __init__(self,
                 num_epochs=30,
                 batch_size=8,
                 optim_params={
                     "lr": 1e-4, 
                     "weight_decay": 1e-5
                    },
                 early_stopping=False,
                 patience=5,
                 tol=1e-4,
                 device='cpu',
                 seed=42,
                 verbose=True,
                 models_input_lengths=None
    ):
        
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.optim_params = optim_params
        self.early_stopping = early_stopping
        self.patience = patience
        self.tol = tol
        self.device = device
        self.seed = seed
        self.verbose = verbose

        self.models_input_lengths = models_input_lengths
    
    def _truncate_series(self, X, y):

        truncation_points = np.clip(
            np.random.geometric(0.1, len(X)), 1, self.max_length
        )
        """
        truncation_points = np.clip(
            np.random.randint(1, 25, len(X)), 1, self.max_length
        )
        """
        truncation_points = np.sort(truncation_points)[::-1]

        partial_series = [
            serie[:truncation_points[j]] for j, serie in enumerate(X)
        ]

        bucket_batch_sampler = BucketBatchSampler(self.batch_size, partial_series, None)
        bucket_dataset = BucketDataset(partial_series, y)

        data_iter = DataLoader(
            bucket_dataset,
            batch_sampler=bucket_batch_sampler, 
            shuffle=False, 
            num_workers=0, 
            drop_last=False
        )

        return data_iter
    
    def _eval_model(self, X, y, criterion):
        
        val_losses = []

        self.model.eval()
        val_dataloader = self._truncate_series(X, y)
        for x_batch, y_batch in val_dataloader:
            
            probas = self.model(x_batch.to(self.device))
            loss = criterion(probas, y_batch)
            val_losses.append(loss.item())

        return np.mean(val_losses)
    
    def fit(self, X, y):

        if self.seed is not None:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        if len(X.shape) == 2:
            n_sample, self.max_length = X.shape
            self.n_dim = 1
            X = np.expand_dims(X, axis=-1)
        elif len(X.shape) == 3:
            n_sample, self.max_length, self.n_dim = X.shape
        
        if self.models_input_lengths is None:
            self.models_input_lengths = np.arange(1, self.max_length+1, 1)

        X, y = (torch.FloatTensor(X), torch.LongTensor(y))
        n_classes = len(np.unique(y))

        if self.early_stopping:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=self.seed
            )
            early_stopper = EarlyStopping(self.patience, self.tol)
        else:
            X_train, y_train = (X, y)

        self.model = LSTM_FCN(self.n_dim, self.max_length, n_classes)
        self.model.to(self.device)
        optimizer = opt.Adam(self.model.parameters(), **self.optim_params)
        criterion = nn.CrossEntropyLoss()

        self.train_losses, self.val_losses = [], []
        pbar = tqdm(range(self.num_epochs)) if self.verbose \
            else range(self.num_epochs)
        
        for epoch in pbar:
            pbar.set_description("Epoch number %s" % str(epoch+1)) if self.verbose else -1
            train_dataloader = self._truncate_series(X_train, y_train)
            batch_losses = []

            for x_batch, y_batch in train_dataloader:
                self.model.train()
                optimizer.zero_grad()
                probas = self.model(x_batch.to(self.device))
                loss = criterion(probas, y_batch.to(self.device))
                loss.backward()
                optimizer.step()

                batch_losses.append(loss.item())
            self.train_losses.append(np.mean(batch_losses))

            if self.early_stopping:
                with torch.no_grad():
                    val_loss = self._eval_model(X_val, y_val, criterion)
                self.val_losses.append(val_loss.item())

                early_stopper(val_loss.item())
                if early_stopper.early_stop:
                    print("Early stop at epoch : %s" % str(epoch+1))
                    break

        return self
    
    def predict(self, X):
        return self.predict_proba(X).argmax(-1)

    def predict_proba(self, X, past_probas=False):
        
        length = X.shape[1]

        try:
            X = torch.FloatTensor(np.expand_dims(X, axis=-1))
        except ValueError:
            #if inhomogenous shape, i.e. different series lengths
            X = [torch.FloatTensor(x)[:, None] for x in X]

        predictions = []
        self.model.eval()

        with torch.no_grad():
            if past_probas:
                for ts in X:
                    if length != 1:
                        predictions.append(
                            [self.model(ts[None, :j]).detach().numpy().squeeze()
                            for j in self.models_input_lengths if j <= len(ts)]
                        )
                    else:
                        predictions.append(self.model(ts[None, :1]).detach().numpy().squeeze())
            else:
                predictions = self.model(X).detach().numpy()

        return predictions


from torch.utils.data import Sampler, Dataset
from collections import OrderedDict

class BucketDataset(Dataset):

    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        if self.targets is None:
            return self.inputs[index]
        else:
            return self.inputs[index], self.targets[index]

class BucketBatchSampler(Sampler):

    # want inputs to be an array
    def __init__(self, batch_size, inputs, targets):
        # Throw an error if the number of inputs and targets don't match
        if targets is not None:
            if len(inputs) != len(targets):
                raise Exception("[BucketBatchSampler] inputs and targets have different sizes")
        # Remember batch size
        self.batch_size = batch_size
        # For each data item (it's index), keep track of combination of input and target lengths
        self.ind_n_len = []
        if targets is None:
            for i in range(0, len(inputs)):
                self.ind_n_len.append((i, (inputs[i].shape[0], 1)))
        else:
            for i in range(0, len(inputs)):
                self.ind_n_len.append((i, (inputs[i].shape[0], targets[i].shape[0])))

        self.batch_list = self._generate_batch_map()
        self.num_batches = len(self.batch_list)


    def _generate_batch_map(self):
        # shuffle all of the indices first so they are put into buckets differently
        random.shuffle(self.ind_n_len)
        # Organize lengths, e.g., batch_map[(5,8)] = [30, 124, 203, ...] <= indices of sequences of length 10
        batch_map = OrderedDict()
        for idx, length in self.ind_n_len:
            if length not in batch_map:
                batch_map[length] = [idx]
            else:
                batch_map[length].append(idx)
        # Use batch_map to split indices into batches of equal size
        # e.g., for batch_size=3, batch_list = [[23,45,47], [49,50,62], [63,65,66], ...]
        #batch_map = dict(sorted(batch_map.items(), reverse=True))
        batch_list = []
        for length, indices in batch_map.items():
            for group in [indices[i:(i + self.batch_size)] for i in range(0, len(indices), self.batch_size)]:
                batch_list.append(group)
        return batch_list

    def batch_count(self):
        return self.num_batches

    def __len__(self):
        return len(self.ind_n_len)

    def __iter__(self):
        self.batch_list = self._generate_batch_map() # <-- Could be a waste of performance
        # shuffle all the batches so they aren't ordered by bucket size
        random.shuffle(self.batch_list)
        for i in self.batch_list:
            yield i
    
#m = LSTM_FCN(input_dim=1, seq_length=24, output_dim=2, hidden_dim=64)
#clf = DeepChronologicalClassifier()

#x = torch.randn(((16,24,1)))
#y = np.random.random_integers(0, 1, 16)

#clf.fit(x, y)

#y = m(x)