import sys 

import torch
import torch.optim as optim
import torch.nn as nn
from torchinfo import summary
from torcheval.metrics.functional import binary_f1_score
import numpy as np

home_dir = '../../../../'
sys.path.append(home_dir)

import preprocessing as prep
import postprocessing as post
import training as train
import models as mod
from sklearn.metrics import roc_curve, auc, recall_score precision_score
import pickle

networktype = 'FFNN'
datatype = 'CMIP6'

args = prep.setup_argperse(datatype, networktype).parse_args()

experiment_name = args.experiment_name
precision = torch.float

og_shape = torch.tensor([2, 56,141])
down_sample_rate = 0
split = 0.2
num_workers_dataloader = 1

epochs = 500
min_loss = 1e9
since_best = 0
early_stop = 50
save_every = 10
number_of_ensembles = 19
predict_region = (slice(30,69.5),slice(-64.5,35),) # in degrees



data_load_path, model_save_path, _ = prep.setup_paths(
    experiment_name, datatype, networktype, home_dir=home_dir)
# is there GPU available?
device = torch.device('cuda:'+str(args.gpu_idx) if torch.cuda.is_available() else 'cpu') 
print('Using device:', device)
print('Model :', torch.cuda.get_device_name(args.gpu_idx))

print(args.crop)

crop = None
flatten = True
if args.crop:
    crop = (slice(0,None),slice(0,90),)
    flatten = False

X_train, y_train, X_val, y_val, pos_weight, idx_train, idx_val = prep.format_CMIP_X_y(
    path=data_load_path,
    og_shape=og_shape,
    down_sample_rate=down_sample_rate,
    dtype=precision,
    split=split,
    only_over=False,
    flatten=flatten,
    verbose=True,
    vars = args.vars.split(','),
    X_file = args.X_file_postfix,
    y_file = args.y_file,
    months= np.array(args.months.split(',')).astype(int),
    regresion = True,
    return_idx = True,
    time_shift = args.time_shift,
    crop=crop
    )

X_test, y_test, test_idx, pos_weight = prep.format_CMIP_X_y(
    path=data_load_path,
    og_shape=og_shape,
    down_sample_rate=down_sample_rate,
    dtype=precision,
    split=split,
    only_over=False,
    flatten=flatten,
    verbose=True,
    vars = args.vars.split(','),
    X_file=args.X_file_postfix,
    y_file=args.y_file,
    months= np.array(args.months.split(',')).astype(int),
    regresion = True,
    return_idx=True,
    test = True,
    time_shift = args.time_shift,
    crop=crop
    )

if args.crop:
    og_shape = torch.tensor([2, 56, 90])


## combine X and y
X = torch.cat([X_train, X_val, X_test], dim=0).to(device)
y = torch.cat([y_train, y_val, y_test], dim=0).to(device)
total_index_for_save = torch.cat([torch.tensor(idx_train), torch.tensor(idx_val), torch.tensor(test_idx)], dim=0)


n_layers = 12
layers = [4096, 2048, 2048, 2048, 2048, 1024, 1024, 1024, 1024, 1024, 1024, 256]


activation = nn.ELU()
batch_norm = True # Sould definitely be true all cases without it cannot get below 0.3 MSEloss
dropout = 0.10345778206213857
lr = 0.0008956904283765739
if args.time_shift:
    #lr = 0.0001
    #activation = nn.ReLU()
    #layers = [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 512, 512, 512, 256]
    print('Time shift is on')
F_loss = nn.MSELoss()

epochs = 1000
early_stop = 100
save_every = 20
folds = 10



X = X.reshape(X.shape[0], -1)
print('X shape:', X.shape)

X = X.to(device)
y = y.to(device)
indexes = np.array_split(total_index_for_save, folds)



# Create a dictionary mapping index to value before splitting
index_to_value = {idx.item(): val.item() for idx, val in zip(total_index_for_save, y)}

for i in range(folds):

    model = mod.FFNN(torch.prod(og_shape),[*layers], activation=activation, last_activation=None, last_number_of_nodes=1, dropout=dropout, batch_norm=batch_norm)


    sample_input = torch.randn(27540, torch.prod(og_shape), device='cpu', dtype=precision)

    
    summary(model, input_data= sample_input, device='cpu', )


    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9688437450753952, 0.9339053096761587))
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.8297176731533378,      
    patience=14,
    threshold=0.001882996040376476,
    min_lr=1e-8,       # Add a minimum LR to prevent the LR from becoming too small
    threshold_mode ='rel',
    verbose=True
    )

    min_loss = 1e9
    since_best = 0



    model.init_weights(pure_zero=False) # avoid weight leakage
    model.to(device)

    all_folds = np.arange(folds)
    current_test_indexes = indexes[i]
    X_test = X[current_test_indexes]
    y_test = y[current_test_indexes]
    X_val = X[indexes[(i+1)%folds]]
    y_val = y[indexes[(i+1)%folds]]

    train_folds = np.delete(all_folds, [i, (i+1)%folds])
    X_train = X[torch.cat([indexes[j] for j in train_folds], dim=0)]
    y_train = y[torch.cat([indexes[j] for j in train_folds], dim=0)]
    
    best_model = model.state_dict()
    for epoch in range(epochs):

        optimizer.zero_grad(set_to_none=True)
        
        model.train()
        # check if there is any overlap
        output = model(X_train).flatten()
        loss = F_loss(output, y_train)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        current_lr = optimizer.param_groups[0]['lr']
        model.eval()

        to_be_tested = model(X_val).flatten()
        loss_for_test = F_loss(to_be_tested, y_val)
        lr_scheduler.step(loss_for_test.item())
        if loss_for_test < min_loss:
            min_loss = loss_for_test
            since_best = 0
            best_model = model.state_dict()
        else:
            since_best += 1
        if since_best > early_stop:
            print('early stopping at epoch:', epoch)
            break


        if epoch % 10 == 0:
            print(f'Epoch: {epoch}, Loss: {loss_for_test.item()}')
            print(f"Current learning rate: {current_lr}")

    model.load_state_dict(best_model)
    model.eval()
    # Run the test fold and save the output
    output = model(X_test).flatten()
    print(output.shape, y_test.shape)
    loss = F_loss(output, y_test)
    print(f'Loss on test fold: {loss.item()}')
    out_test = output.cpu().detach().numpy()


    # Create triples of (index, true value, prediction)
    test_triples = [(idx.item(), y_val, pred) for idx, y_val, pred in zip(current_test_indexes, y_test, out_test)]
    # Sort them by index
    test_triples.sort(key=lambda x: x[0])

    fold_data_save = []
    for idx_val, true_val, pred_val in test_triples:
        fold_data_save.append((idx_val, true_val.item(), pred_val))

    with open(model_save_path + 'output_fold_' + str(i) + '.pkl', 'wb') as f:
        pickle.dump(fold_data_save, f)

    
    with open(model_save_path + 'output_fold_' + str(i) + '.pkl', 'wb') as f:
        pickle.dump((fold_data_save), f)
    # Save the model
    torch.save(best_model, model_save_path + 'model_fold_' + str(i))
