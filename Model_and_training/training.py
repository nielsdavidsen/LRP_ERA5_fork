
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torcheval.metrics.functional import binary_f1_score
import torch.distributed as dist


import os
import preprocessing as prep
import numpy as np
import models as mod

def device_train_test_split(X, y, device, test_size=0.2, random_state=42):
    train_X, test_X, train_y, test_y = train_test_split(X,y, test_size=0.2, random_state=42)
    # send it to the GPU
    train_X = torch.tensor(train_X, dtype=torch.float32).to(device)
    test_X = torch.tensor(test_X, dtype=torch.float32).to(device)
    train_y = torch.tensor(train_y, dtype=torch.float32).to(device)
    test_y = torch.tensor(test_y, dtype=torch.float32).to(device)
    return train_X, test_X, train_y, test_y

def FFNN_optuna_objective(trail):

    n_layers = trail.suggest_int("n_layers", 3, 7)
    layers = []
    for i in range(2):
        layers.append(trail.suggest_categorical("n_units_l{}".format(i), [1024,2048]))

    for i in range(2,n_layers):
        layers.append(trail.suggest_categorical("n_units_l{}".format(i), [64,128,256,512]))

    # add another parameter to the dictionary

    trail.set_user_attr("layers", layers)
    model = mod.FFNN(train_X.shape[1],[*layers], activation=nn.ELU()).to(device)

    # Generate the optimizers.
    optimizer_name = "Adam"
    # make the learning rate adaptive and a hyperparameter
    lr = trail.suggest_float("lr", 1e-5, 1e-3)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    gamma = trail.suggest_float("gamma", 0.9, 0.99)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    CEloss = nn.CrossEntropyLoss()  ## this loss object must be used the loop. Directly using nn.CrossEntropyLoss() gives error


    epochs = trail.suggest_int("epochs", 300,600)
    min_loss = 1e9
    since_best = 0
        

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_X)
        loss = CEloss(output.T[0], train_y)
        loss.backward()
        model.eval()
        loss_test = CEloss(model(test_X).T[0], test_y)
        if loss_test < min_loss:
            min_loss = loss_test
            since_best = 0
        else:
            since_best += 1
        if since_best > 50:
            print(f"Early stopping at epoch {epoch}, with loss {loss_test.item()}")
            break
        
        optimizer.step()


    model.eval()
    output = model(test_X)
    loss = CEloss(output.T[0], test_y)
    return loss.item()





def ddp_setup(rank,world_size):
    # taken from https://github.com/pytorch

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group(backend='gloo', rank=rank, world_size=world_size)



def output_tests(y_val, to_be_tested, loss_for_test, epoch):
    """
    Function to print tests that are done on the model

    Parameters
    ----------
    y_val: torch.Tensor
        The actual values
    to_be_tested: torch.Tensor
        The predicted values
    loss_for_test: torch.Tensor
        The loss for the epoch to be tested
    epoch: int
        The epoch number

    Returns
    -------
    None

    """
    from sklearn.metrics import roc_curve, auc, recall_score, precision_score
    from torcheval.metrics.functional import binary_f1_score
    from numpy import round as np_round
    from postprocessing import heidke_skill_score

    # F1 Needs its inputs to be torch.tensor
    F1 = np_round(binary_f1_score(to_be_tested, y_val).item(), 5)

    
    y_val_cpu = y_val.cpu().numpy()
    to_be_tested = to_be_tested.cpu().detach().numpy()
    fpr, tpr, _ = roc_curve(y_val_cpu, to_be_tested)
    roc_auc = np_round(auc(fpr, tpr), 5)
    precision = np_round(precision_score(y_val_cpu, to_be_tested > 0.5, zero_division = 0), 5)
    recall = np_round(recall_score(y_val_cpu, to_be_tested > 0.5),5)
    heidke = np_round(heidke_skill_score(to_be_tested, y_val_cpu),5)
    print(f"Epoch {epoch} Loss: {loss_for_test.item()} F1: {F1} Heidke: {heidke} ROC AUC: {roc_auc} Precision: {precision} Recall: {recall}")



def prepare_dataloader(dataset, batch_size: int, **kwargs):
    # taken from https://github.com/pytorch

    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset, 
            num_replicas=kwargs['world_size'],
            rank=kwargs['rank'],
            shuffle=True,drop_last=False),
        num_workers=num_workers,
    )

class Trainer:
    
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        lr_scheduler: torch.optim.lr_scheduler,
        early_stop: int,
        pos_weight: torch.Tensor,
        model_save_path: str,
        F_loss: torch.nn.Module,
        Flatten_Features: bool,
        world_size: int = 1,
        clip = True
    ) -> None:
        self.gpu_id = gpu_id
        self.F_loss = F_loss
        self.pos_weight = pos_weight.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.save_every = save_every
        self.model = model.to(gpu_id)
        self.model = DDP(model, device_ids=[gpu_id], output_device=gpu_id,
                        find_unused_parameters=True)    
        self.best_loss = 1e9
        self.since_best = 0
        self.early_stop_flag = False
        self.model_save_path = model_save_path
        self.early_stop = early_stop
        self.val_data = val_data
        self.Flatten_Features = Flatten_Features
        self.best_epoch = 0
        self.world_size = world_size
        self.clip = clip


        
    def cleanup_gpu(self):  
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model.module(source).squeeze()
        loss = self.F_loss(output, targets)
        loss.backward()
        if self.clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()


    def _run_epoch(self, epoch):
        
        self.train_data.sampler.set_epoch(epoch)
        first_print = 0

        for source, targets in self.train_data:
            if self.Flatten_Features:
                source = source.reshape(-1,np.prod(source.shape[1:]))
            if first_print == 0:
                first_print += 1
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)
        
    def _run_validation(self, epoch):
        with torch.no_grad():
            X_batch_val, y_batch_val = next(iter(self.val_data))
            if self.Flatten_Features:
                X_batch_val = X_batch_val.reshape(-1,np.prod(X_batch_val.shape[1:]))
            
            X_batch_val, y_batch_val = X_batch_val.to(self.gpu_id), y_batch_val.to(self.gpu_id)
            output_for_test = self.model.module(X_batch_val).squeeze()
            loss_for_test = self.F_loss(output_for_test, y_batch_val)
            self.lr_scheduler.step(loss_for_test.item())

            # Synchronize loss across processes
            loss_tensor = torch.tensor([loss_for_test.item()], device=self.gpu_id)
            torch.distributed.all_reduce(loss_tensor)
            loss_tensor.div_(self.world_size)

            if self.gpu_id == 0: 
                print('Epoch: ', epoch, 'Validation loss: ', loss_tensor.item(), 'LR: ', self.lr_scheduler.get_last_lr())
        return loss_tensor

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = self.model_save_path
        torch.save(ckp, 'checkpoint'+'.pt')
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")


    def train(self, max_epochs: int):
        # print the layers
        since_best = 0
        best_loss = 1e9
        for epoch in range(max_epochs):
            self.model.train()
            self._run_epoch(epoch)
            self.model.eval()
            loss_tensor = self._run_validation(epoch)
            self.cleanup_gpu()
            if loss_tensor.item() < best_loss:
                best_loss = loss_tensor.item()
                since_best = 0
            else:
                since_best += 1
                if since_best > self.early_stop:
                    print(f"Early stopping at epoch {epoch}, with loss {loss_tensor.item()}")
                    break


        return self.best_loss
