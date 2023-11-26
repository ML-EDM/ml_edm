import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod
class Deep_Classifiers(ABC):

    def __init__(self):
        self.embed_trigger_model = False

    @abstractmethod
    def compute_loss(self, probas, labels):
        """
        Compute specific model loss, 
        could be changing if model embed 
        trigger or not
        """


class ClassificationModel(nn.Module, Deep_Classifiers):

    def __init__(self, backbone, classif_head):
        super(ClassificationModel, self).__init__()
        self.backbone = backbone
        self.clf_head = classif_head
    
    def forward(self, x):
        x = self.backbone(x)
        return self.clf_head(x)

    def compute_loss(self, probas, labels):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(probas, labels)
        return loss 

class ELECTS(nn.Module, Deep_Classifiers):

    def __init__(self, input_dim, backbone, classif_head, alpha, epsilon):
        super(ELECTS, self).__init__()
        self.epsilon = epsilon
        self.alpha = alpha

        self.input_dim = input_dim
        self.backbone = backbone
        self.clf_head = classif_head
        self.input_norm = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.backbone.input_dim)
        )
        self.decision_head = nn.Sequential(
            nn.Linear(self.clf_head.hidden_dim, 1, bias=True),
            nn.Sigmoid()
        )
        # specific init to predict late at first epoch ???
        torch.nn.init.normal_(self.decision_head[0].bias, mean=-3, std=1e-1)

        self.embed_trigger_model = True
    
    def _sample_stop_decision(self, probas_stopping):
        dist = torch.stack([1-probas_stopping, probas_stopping], dim=1)
        return torch.distributions.Categorical(dist).sample().bool()
    
    def forward(self, x, predict=False):

        x = self.input_norm(x)
        outputs = self.backbone(x)

        probas_classes = self.clf_head(outputs)
        probas_stopping = self.decision_head(outputs).reshape((len(x), -1))

        batch_size, sequence_length = probas_stopping.shape

        stops = []
        if predict: # if prediction mode, sample trigger time from probas 
            for t in range(sequence_length):
                if t < sequence_length - 1:
                    trigger = self._sample_stop_decision(probas_stopping[:,t])
                    stops.append(trigger)
                else:
                    last_stop = torch.ones((batch_size,)).bool()
                    stops.append(last_stop)

            stopped = torch.stack(stops, dim=1).bool()
            first_stops = (stopped.cumsum(dim=1) == 1) & stopped

            tau_star = first_stops.long().argmax(dim=1)
            pred_star = torch.masked_select(probas_classes.argmax(dim=-1), first_stops)

            return pred_star, tau_star

        return probas_classes, probas_stopping
    
    def compute_loss(self, probas_classes, probas_stopping, labels):

        n_classes = probas_classes.shape[-1]
        batch_size, sequence_length = probas_stopping.shape

        # repeat labels as many times as timestamps for loss sum
        labels = torch.repeat_interleave(labels, sequence_length).view(batch_size, -1)

        prod = torch.cat(
            (torch.ones((batch_size, 1)), (1 - probas_stopping[:, 1:]).cumprod(dim=1)), dim=-1
        )
        pts = torch.cat(
            (prod[:, :-1] * probas_stopping[:, 1:], prod[:, -1:]), dim=-1
        ) + self.epsilon / sequence_length

        #pts = self.calculate_probability_making_decision(probas_stopping)

        criterion = nn.CrossEntropyLoss(reduction='none')
        #criterion = nn.NLLLoss(reduction='none')

        c_losses = pts * criterion(probas_classes.view(-1, n_classes), labels.view(-1)).view((batch_size, -1))
        #c_losses = pts * criterion(torch.log(probas_classes).view(-1, n_classes), labels.view(-1)).view((batch_size, -1))

        c_loss = c_losses.sum(dim=1).mean()

        t = torch.ones(batch_size, sequence_length) * torch.arange(sequence_length)
        labels_one_hot = torch.eye(n_classes)[labels]
        proba_correct = torch.masked_select(probas_classes, labels_one_hot.bool())
        e_losses = pts * proba_correct.view(batch_size, -1) * (1 - t / sequence_length)
        e_loss = e_losses.sum(dim=1).mean()

        loss = self.alpha * c_loss + (1 - self.alpha) * e_loss

        return loss

    def calculate_probability_making_decision(self, deltas):
        """
        Equation 3: probability of making a decision

        :param deltas: probability of stopping at each time t
        :return: comulative probability of having stopped
        """
        batchsize, sequencelength = deltas.shape

        pts = list()

        initial_budget = torch.ones(batchsize, device=deltas.device)

        budget = [initial_budget]
        for t in range(1, sequencelength):
            pt = deltas[:, t] * budget[-1]
            budget.append(budget[-1] - pt)
            pts.append(pt)

        # last time
        pt = budget[-1]
        pts.append(pt)

        return torch.stack(pts, dim=-1)
    