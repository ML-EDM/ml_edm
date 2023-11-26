import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm
from sklearn.model_selection import train_test_split

from deep.deep_models import *
from deep.modules import *

import sys
sys.path.append("..")

from utils import *
from trigger_models import TriggerModel

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
 
class DeepChronologicalClassifier(TriggerModel):

    def __init__(self,
                 model,
                 num_epochs=100,
                 batch_size=8,
                 optim_params={
                     "lr": 1e-3,
                     "weight_decay": 5e-4
                    },
                 early_stopping=True,
                 patience=20,
                 tol=1e-4,
                 device='cpu',
                 seed=42,
                 verbose=True,
                 models_input_lengths=None
    ):
        super().__init__()
        self.model = model

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
    
    def _eval_model(self, X, y):
        
        val_losses = []

        self.model.eval()
        if self.model.embed_trigger_model:
            val_dataset = TensorDataset(X, y)
            val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, drop_last=False)
        else:
            val_dataloader = self._truncate_series(X, y)
        for x_batch, y_batch in val_dataloader:
            
            try: # if deep model embed trigger model
                probas, stopping = self.model(x_batch.to(self.device))
                loss = self.model.compute_loss(probas, stopping, y_batch.to(self.device))
            except ValueError: 
                probas = self.model(x_batch.to(self.device))
                loss = self.model.compute_loss(probas, y_batch.to(self.device))

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
            n_sample, self.n_dim, self.max_length = X.shape
        
        if self.models_input_lengths is None:
            self.models_input_lengths = np.arange(1, self.max_length+1, 1)

        X, y = (torch.FloatTensor(X), torch.LongTensor(y))
        self.n_classes = len(np.unique(y))

        if self.early_stopping:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=self.seed
            )
            early_stopper = EarlyStopping(self.patience, self.tol)
        else:
            X_train, y_train = (X, y)

        #self.model = LSTM_FCN(self.n_dim, self.max_length, n_classes)
        self.model.to(self.device)
        optimizer = opt.Adam(self.model.parameters(), **self.optim_params)

        self.train_losses, self.val_losses = [], []
        pbar = tqdm(range(self.num_epochs)) if self.verbose \
            else range(self.num_epochs)
        
        for epoch in pbar:
            pbar.set_description("Epoch number %s" % str(epoch+1)) if self.verbose else -1
            if self.model.embed_trigger_model:
                train_dataset = TensorDataset(X_train, y_train)
                train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, drop_last=False)
            else:
                train_dataloader = self._truncate_series(X_train, y_train)
            batch_losses = []

            for x_batch, y_batch in train_dataloader:
                self.model.train()
                optimizer.zero_grad()

                if self.model.embed_trigger_model: # if deep model embed trigger model
                    probas, stopping = self.model(x_batch.to(self.device))
                    loss = self.model.compute_loss(probas, stopping, y_batch.to(self.device))
                else: 
                    probas = self.model(x_batch.to(self.device))
                    loss = self.model.compute_loss(probas, y_batch.to(self.device))

                loss.backward()
                optimizer.step()

                batch_losses.append(loss.item())
            self.train_losses.append(np.mean(batch_losses))

            if self.early_stopping:
                with torch.no_grad():
                    val_loss = self._eval_model(X_val, y_val)
                self.val_losses.append(val_loss.item())

                early_stopper(val_loss.item())
                if early_stopper.early_stop:
                    print("Early stop at epoch : %s" % str(epoch+1))
                    break

        return self
    
    def predict(self, X):
        if self.model.embed_trigger_model:
            grouped_X = {}
            for serie in X:
                length = len(serie)
                if length in grouped_X.keys():
                    grouped_X[length].append(serie)
                else:
                    grouped_X[length] = [serie]
            
            predictions, t_triggers = [], []
            self.model.eval()
            with torch.no_grad():
                for length, series in grouped_X.items():
                    series = torch.FloatTensor(series).reshape((len(series), length, -1))
                    preds, t = self.model(series, predict=True)
                    predictions.append(preds.detach().numpy())
                    t_triggers.extend(t.detach().tolist())

            return predictions, None, t_triggers
        else:
            return self.predict_proba(X).argmax(-1)

    def predict_proba(self, X):

        # Validate X format with varying series lengths
        #X, _ = check_X_y(X, None, equal_length=False)

        # Group X by batch of same length
        grouped_X = {}
        for serie in X:
            length = len(serie)
            if length in grouped_X.keys():
                grouped_X[length].append(serie)
            else:
                grouped_X[length] = [serie]

        predictions, triggers = [], []
        self.model.eval()
        with torch.no_grad():
            for length, series in grouped_X.items():
                series = torch.FloatTensor(series).reshape((len(series), length, -1))
                if self.model.embed_trigger_model:
                    preds, trigg = self.model(series)
                    triggers.extend(trigg.detach().tolist())
                else:
                    preds = self.model(series)
                    
                predictions.append(preds.detach().numpy())

        return np.vstack(predictions), np.array(triggers).squeeze()

    def predict_past_proba(self, X):

        # Validate X format with varying series lengths
        #X, _ = check_X_y(X, None, equal_length=False)

        # Group X by batch of same length
        grouped_X = {}
        for serie in X:
            length = len(serie)
            if length in grouped_X.keys():
                grouped_X[length].append(serie)
            else:
                grouped_X[length] = [serie]
        
        predictions, triggers = [], []
        for length, series in grouped_X.items():
            series = torch.FloatTensor(series).reshape((len(series), length, -1))

            all_probas = [self.model(series[:,:j,:]).detach().numpy().squeeze() 
                          for j in range(1, length+1)]
            all_probas = np.stack(all_probas).reshape((-1, length, self.n_classes))
            predictions.extend(all_probas)

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

""" 
backbone = LSTM(input_dim=1, hidden_dim=64, return_all_states=True)
clf_head = ClassificationHead(hidden_dim=64, n_classes=2)
model = ELECTS(1, backbone, clf_head, alpha=0.5, epsilon=10)

clf = DeepChronologicalClassifier(model)

x = torch.randn(((16,24,1)))
y = np.random.random_integers(0, 1, 16)

x = np.array(x)
clf.fit(x, y)
clf.predict([x[0,:,:], x[1,:10,:], x[-1,:10,:], x[2,:1,:]])

y = model(x)
"""