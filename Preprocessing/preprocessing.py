import xarray as xr
import dask
# dask.config.set(scheduler='synchronous') # should fix issues with num_workers > 0
import numpy as np
from torch import tensor, float32
import torch
from torch.utils.data.distributed import DistributedSampler
import sys

### If changes are made to the names in this class will effect all
### datasets saved to dict


class era5_data:


    def __init__(self,path = None, load_data_file = False):
        if load_data_file:
            self.xr_data = xr.open_dataset(path)
        self.rescale_dict = {}
        self.target = None
        self.prediction = None
        self.pred_time_period = None
        self.pred_time_period_index = None
        self.target_region = None
        self.target_window = None
        self.prediction_region = None
        self.field_shape_predict = None
        self.field_shape_target = None



    def rescale_long(self, name_longitude = 'longitude'):
        self.xr_data[name_longitude] = self.xr_data[name_longitude].where(
                        self.xr_data[name_longitude] < 180, self.xr_data[name_longitude] - 360)

        self.xr_data = self.xr_data.sortby(self.xr_data[name_longitude])

    def reformat_time(self, name_time = 'date', time_format = "%Y%m%d", time_period = slice('1940-01-01', '2023-12-01')):
        from datetime import datetime
        self.xr_data['date'] = [datetime.strptime(str(date_), "%Y%m%d") for date_ in self.xr_data['date'].values]
        self.xr_data = self.xr_data.sel(date = time_period)
    

    #TODO: update this to alow for the percintile flags
    def make_rolling_mean_target(self, var, region, window_size, ensemble = False, test_months = None):
        self.target_region = region
        self.target_window = window_size
        target = self.xr_data[var].sel(longitude=region[0] , latitude=region[1])\
                            .mean(dim=['latitude', 'longitude'])

        # rolling mean over time_period
        months = []
        if test_months is None:
            test_months = range(1,13)
        for i in test_months:
            rolling_mean = target.sel(date= self.xr_data['date'].dt.month == i)\
                                    .rolling(date=window_size, center = True).mean()
            
            # if ensemble is True, we need to keep the first dimension
            if ensemble:
                months.append((target.sel(date= self.xr_data['date'].dt.month == i)>\
                                    rolling_mean).values[:,(window_size//2):-(window_size//2-1)])
            else:
                months.append((target.sel(date= self.xr_data['date'].dt.month == i)>\
                                    rolling_mean).values[(window_size//2):-(window_size//2-1)])


        self.field_shape_predict = (self.xr_data[var].sel(latitude=region[1]).latitude.shape[0],
                        self.xr_data[var].sel(longitude=region[0]).longitude.shape[0])
        
        
        if ensemble:
            out = np.array(months).transpose(1, 2, 0).flatten()
        
        # NOTE: This is probably wrong as it needed updated for the ensemble case
        else:
            out = np.array(months).flatten('F')
        
        self.target = out
        return out, (rolling_mean.isel(date = 15).date.values, rolling_mean.isel(date = -15).date.values)

    def rescale_prediction(self, var, region,time_period_index, time_period = None):
        self.prediction_region = region
        if time_period is not None:
            mean_var = self.xr_data[var].sel(longitude=region[0] , latitude=region[1]).isel(date = time_period_index).sel(date=time_period).mean()
            std_var  = self.xr_data[var].sel(longitude=region[0] , latitude=region[1]).isel(date = time_period_index).sel(date=time_period).mean()
            self.pred_time_period = time_period
        else:
            mean_var = self.xr_data[var].sel(longitude=region[0] , latitude=region[1]).isel(date = time_period_index).mean()
            std_var  = self.xr_data[var].sel(longitude=region[0] , latitude=region[1]).isel(date = time_period_index).std()
            self.pred_time_period_index = time_period_index
        self.rescale_dict[var] = (mean_var.values, std_var.values)

        # NOTE: Test this line next time you can
        self.field_shape_predict = (self.xr_data[var].sel(latitude=region[1]).latitude.shape[0],
                                self.xr_data[var].sel(longitude=region[0]).longitude.shape[0])
        print(mean_var.values.shape, std_var.values.shape)
        print(mean_var, std_var)
        if time_period is not None:
            return (self.xr_data[var].sel(longitude=region[0] , latitude=region[1]).isel(date = time_period_index).sel(date=time_period) - mean_var) / std_var
        else:
            return (self.xr_data[var].sel(longitude=region[0] , latitude=region[1]).isel(date = time_period_index) - mean_var) / std_var

    def flatten_predict(self, inputs):
        flat = []
        
        for var in inputs:
            flat.append(var.values.reshape(self.target.shape[0], -1))
        self.prediction = np.concatenate(flat, axis = 1)

    def load_from_dict(self, path, load_data_file = False):
        import pickle

        with open(path, 'rb') as file:
            dict_ = pickle.load(file)
        
        if load_data_file:
            self.xr_data = dict_['xr_data']
        self.rescale_dict = dict_['rescale_dict']
        self.target = tensor(dict_['target'],dtype=float32)
        self.prediction = tensor(dict_['prediction'],dtype=float32)
        self.pred_time_period = dict_['pred_time_period']
        self.pred_time_period_index = dict_['pred_time_period_index']
        self.target_region = dict_['target_region']
        self.target_window = dict_['target_window']
        self.prediction_region = dict_['prediction_region']
        self.field_shape_predict = dict_['field_shape_predict']
        self.field_shape_target = dict_['field_shape_target']
        del dict_

    def __getitem__(self, key, in_memory = True):
        if in_memory:
            return self.prediction[key], self.target[key]
    
    def __len__(self):
        return len(self.prediction)




class cmip6_data:


    def __init__(self,path1 = None, path2 = None, path3 = None, load_data_file = False, chunk_size = 1000, zg_height = 50000):
        if load_data_file:
            
            self.xr_data_pr  = xr.open_dataset(path1 + 'pr/pr'  + path2, chunks = {'time': chunk_size}).sel(time = slice('1851-01-01', None))
            self.xr_data_psl = xr.open_dataset(path1 + 'psl/psl' + path2, chunks = {'time': chunk_size}).sel(time = slice('1851-01-01', None))
            self.xr_data_ts  = xr.open_dataset(path1 + 'ts/ts'  + path2, chunks = {'time': chunk_size}).sel(time = slice('1851-01-01', None))
            self.xr_data_zg = xr.open_dataset(path3 + 'zg' + path2, chunks = {'time': chunk_size}).sel(plev = zg_height, time = slice('1851-01-01', None))
            self.xr_data = xr.merge([self.xr_data_pr, self.xr_data_psl, self.xr_data_ts, self.xr_data_zg])

        self.rescale_dict = {}
        self.target = None
        self.prediction = None
        self.pred_time_period = None
        self.pred_time_period_index = None
        self.target_region = None
        self.target_window = None
        self.prediction_region = None
        self.field_shape_predict = None
        self.field_shape_target = None
        self.length = 0



    def rescale_long(self, name_longitude = 'longitude'):
        self.xr_data[name_longitude] = self.xr_data[name_longitude].where(
                        self.xr_data[name_longitude] < 180, self.xr_data[name_longitude] - 360)

        self.xr_data = self.xr_data.sortby(self.xr_data[name_longitude])

    def detrend(self, var, region):
        X = self.xr_data.time.values.astype('datetime64[M]').astype(int)
        X += np.abs(X.min())
        A = np.vstack([X, np.ones(len(X))]).T

        Y = self.xr_data[var].sel(lat = region[0] , lon = region[1]).stack(combined_lon_lat = ['lat','lon'])
        print('shape in detrend')
        print(A.shape, Y.shape)
        a_and_b = np.linalg.lstsq(A,Y.values)[0]

        result = (a_and_b[0][np.newaxis, :] * X[:, np.newaxis] + a_and_b[1][np.newaxis, :])
        temp = self.xr_data[var].copy()
        temp.loc[dict(lat = region[0], lon =region[1])] = (Y - result).unstack(dim = 'combined_lon_lat').transpose('time','lat','lon')
        self.xr_data[var] = temp
        maps_shape = self.xr_data[var].sel(lat = region[0] , lon = region[1]).shape


    def make_rolling_mean_target(self, var, region, window_size, ensemble=False, test_months=None, chunk_size_local = 136*3, detrend = False):

        self.target_region = region
        self.target_window = window_size
        if detrend:
            self.detrend(var, [region[1],region[0]])
            print('Detrended', var, self.xr_data[var].sel(lat= region[0] , lon = region[1]).isel(lon = -1, lat = -1).values)

        target = self.xr_data[var].sel(lat=region[0], lon=region[1]).mean(dim=['lat', 'lon'])

        target = target.chunk({'time': chunk_size_local})

        def rolling_stats(x):

            rolling_mean = x.rolling(time=window_size, center=True).mean()

            rolling_std = x.rolling(time=window_size, center=True).std()

            # Compute condition
            condition = ((x > (rolling_mean + rolling_std)).astype(int) +
                        (x < (rolling_mean - rolling_std)).astype(int) * 2)
            return condition

        # Apply the rolling function to each month group
        condition = target.groupby('time.month').apply(rolling_stats)

        condition = condition.chunk({'time': -1})

        self.field_shape_predict = (
            self.xr_data[var].sel(lat=region[0]).lat.size,
            self.xr_data[var].sel(lon=region[1]).lon.size
        )


        # Select only the months of interest
        if test_months is not None:
            result = condition.where(condition['time.month'].isin(test_months), drop=True)

            self.target = result
            return result
        
        else:
            # reindex to original time
            condition = condition.isel(time = slice(window_size//2*12, -(window_size//2-1)*12))
            self.target = condition
            return condition



    def rescale_prediction(self, var, region,time_period_index, time_period = None, chunk_size_local = 135*3, detrend = False):
        self.prediction_region = region

        if detrend:
            self.detrend(var, region)
            print('Detrended', var, self.xr_data[var].sel(lat = region[0] , lon = region[1]).isel(lon = -1, lat = -1).values)
            
        if time_period is not None:
            mean_var = self.xr_data[var].sel(lat = region[0] , lon = region[1]).isel(time = time_period_index).sel(time = time_period).groupby('time.month').mean('time')
            std_var  = self.xr_data[var].sel(lat = region[0] , lon = region[1]).isel(time = time_period_index).sel(time = time_period).groupby('time.month').std('time')
            self.pred_time_period = time_period
        else:
            mean_var = self.xr_data[var].sel(lat = region[0] , lon = region[1]).isel(time = time_period_index).groupby('time.month').mean('time')
            std_var  = self.xr_data[var].sel(lat = region[0] , lon = region[1]).isel(time = time_period_index).groupby('time.month').std('time')
            self.pred_time_period_index = time_period_index
        self.rescale_dict[var] = (mean_var.values, std_var.values)

        # NOTE: Test this line next time you can
        self.field_shape_predict = (self.xr_data[var].sel(lat = region[1]).lat.shape[0],
                                self.xr_data[var].sel(lon = region[0]).lon.shape[0])
        # rechunk the data
        self.xr_data[var] = self.xr_data[var].chunk({'time': chunk_size_local})
        if time_period is not None:
            return (self.xr_data[var].sel(lat = region[0] , lon = region[1]).isel(time = time_period_index).sel(time = time_period).groupby('time.month') - mean_var).groupby('time.month') / std_var
        else:
            return (self.xr_data[var].sel(lat = region[0] , lon = region[1]).isel(time = time_period_index).groupby('time.month') - mean_var).groupby('time.month') / std_var

    def __getitem__(self, key, in_memory = True):
        return self.prediction[key], self.target[key]
        

    
    def __len__(self):
        return len(self.prediction['time'])







def setup_argperse(datatype, networktype):
    import argparse 
    parser = argparse.ArgumentParser(description='Train a' + networktype + ' model on ' + datatype + ' data')
    parser.add_argument('--Train', type=bool, default=True, help='Train the model')
    parser.add_argument('--Load_complete', type=bool, default=False, help='Load the complete data')
    parser.add_argument('--Test_data', type=bool, default=False, help='Test the data')
    parser.add_argument('--gpu_idx', type=int, default=0, help='Index of the GPU to use')
    parser.add_argument('--add_sigmoid', type=bool, default=False, help='Add sigmoid to the output')
    parser.add_argument('--gpu_idxes', type=str, default='1,2,3,4')
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--X_file_postfix', type=str, default='')
    parser.add_argument('--vars', type=str, default='ts,psl', help='The variables to use in the model will be split by ,')
    parser.add_argument('--y_file', type=str, default='CMIP6_target_combined.nc')
    parser.add_argument('--months', type=str, default='0,1,2,3,4,5,6,7,8,9,10,11')
    parser.add_argument('--turn_of_TP_mask', type=bool, default=False, help='If true uses all the observations')
    parser.add_argument('--sample_lrp_plots', type=bool, default=False, help='If true make 3 sample lrp plots of just one month')
    parser.add_argument('--time_shift', type=bool, default=False, help='If true shifts the data one month')
    parser.add_argument('--crop', type=bool, default=False, help='If true crops the data')
    return parser



def setup_paths(experiment_name, datatype, networktype, home_dir = '../../../../'):
    import os
    import sys
    # remove all '/' from datatype, networktype and exp_name 
    datatype = datatype.replace('/','')
    networktype = networktype.replace('/','')
    experiment_name = experiment_name.replace('/','')

    print('Using a network of type:', networktype, 'on data of type:', datatype,
    'the experiment being run is:', experiment_name)

    data_load_path = home_dir + 'Data/'+datatype+'/'
    model_save_path = home_dir + 'experiments/'+datatype+'/'+networktype+'/'+experiment_name+'/Models/'
    fig_save_path = home_dir + 'experiments/'+datatype+'/'+networktype+'/'+experiment_name+'/Figures/'

    # check if the directory exists
    if not os.path.exists(model_save_path):
        sys.exit('The model save path does not exist, '+model_save_path)

    if not os.path.exists(data_load_path):
        sys.exit('The data load path does not exist, '+data_load_path)
    
    if not os.path.exists(fig_save_path):
        sys.exit('The figure save path does not exist, '+fig_save_path)
    
    return data_load_path, model_save_path, fig_save_path


from torch.utils.data import DataLoader, TensorDataset, Dataset
class CustomImageDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def netcdf_to_X_and_y(data_load_path, data_save_path, load_file_name, region, number_of_ensembles,
    test_plot = False, save_file_name = None):
    import xarray as xr
    import pickle
    import matplotlib.pyplot as plt
    from pandas import DataFrame, Series    
    combined_data = xr.open_dataset(data_load_path + load_file_name)

    psl_values = combined_data.psl.sel(lat = region[0], lon = region[1]).values
    ts_values = combined_data.ts.sel(lat = region[0], lon = region[1]).values
    total_data_points = len(combined_data.time)*number_of_ensembles
    psl_values = psl_values.reshape(total_data_points,-1)
    ts_values = ts_values.reshape(total_data_points,-1)

    X = np.array(list(zip(psl_values, ts_values)))
    y = combined_data.target.values.flatten().reshape(-1,1)
    X = X.reshape(y.shape[0],-1)
    y[y>1] = 0
    X = DataFrame(X)
    y = Series(y.flatten())

    if test_plot:
        plt.imshow(X[0,:56*141].reshape(56,141), origin='lower')
        plt.savefig('test.png')

        plt.imshow(X[0,56*141:].reshape(56,141), origin='lower')
        plt.savefig('test2.png')

    if save_file_name is not None:
        with open(data_save_path + 'X' + save_file_name, 'wb') as f:
            pickle.dump(X, f)

        with open(data_save_path + 'y' + save_file_name, 'wb') as f:
            pickle.dump(y, f)

    return X, y


def down_sample(X, y, down_sample_rate, og_shape):
    from pandas import DataFrame, Series
    from numpy import concatenate

    down_sampled_X = []
    down_sampled_y = []
    for i in range(down_sample_rate):
        down_sampled_X.append(X.iloc[:,i::down_sample_rate].values)
        down_sampled_y.append(y)

    X = DataFrame(concatenate(down_sampled_X, axis=0))
    y = Series(concatenate(down_sampled_y, axis=0))
    og_shape[2] = og_shape[2]//down_sample_rate
    return X, y, og_shape


def transform_data(X, y, X_set,y_set, transformer, DDP = False, verbose = False, kwarg_Dataloader = None):
    from torchvision import transforms

    X_pix_min = torch.amin(X, dim = (0,1))
    X_pix_max = torch.amax(X, dim = (0,1))
    print(X.shape, X_pix_min.shape, X_pix_max.shape)

    X = (X - X_pix_min) / (X_pix_max - X_pix_min)
    X_mean = torch.mean(X, dim = (0,2,3))
    X_std = torch.std(X, dim = (0,2,3))
    
    transformer.append(transforms.Normalize(X_mean, X_std))
    transformer = transforms.Compose(transformer)

    temp_dataset = CustomImageDataset(
        X_set,
        y_set,
        transform=transformer
    )

    if DDP:
        return temp_dataset 
        
    return DataLoader(temp_dataset, **kwarg_Dataloader)

def chosse_months(data, months, og_shape = None, flatten = True):
    if og_shape is not None:
        data = data.reshape(-1, 12, torch.prod(og_shape))

        if flatten:
            data = data[:,months,:].reshape(-1, torch.prod(og_shape))
            
        else:
            data = data[:,months,:].reshape(-1, *og_shape)
        
    else:
        data = data.reshape(-1, 12)
        data = data[:,months].flatten()
    return data

def format_CMIP_X_y(path, og_shape, X_file, vars,y_file, down_sample_rate = 0, dtype = torch.float, split = 0.2,
                    only_over = False, pos_weight = True, transformer = None, DDP = False,
                    flatten = False, kwarg_Dataloader = None, verbose = False,
                    test = False, crop = None, 
                    netcdf_file = None, predict_region = (slice(30,69.5),slice(-64.5,35),), number_of_ensembles = 19, save_file_name = ' ', seed = 27051797,
                    regresion = False, return_idx = False, months = None, time_shift = False):

    from torchvision import transforms
    from torch.utils.data import DataLoader, TensorDataset, Dataset
    import pandas as pd
    import torch
    import pickle
    from sklearn.model_selection import train_test_split

    if netcdf_file is not None:
        X,y = netcdf_to_X_and_y(path, path, netcdf_file, predict_region, number_of_ensembles, save_file_name = save_file_name)

    
    else:
        X = []
        for var in vars:
            with open(path + 'X_' + var + '_' + X_file + '.pkl', 'rb') as f:
                X.append(pickle.load(f))
        X = pd.concat(X, axis = 1)
        if verbose:
            print('loaded X file with ', vars, ' as the variables, and shape', X.shape)

        if regresion:
            y = xr.open_dataset(path + y_file).pr.values.flatten()
            if verbose:
                print('using mean precipitation as target')
                print('With ', y.shape, ' values')
                print('First value', y[0])
                print('Mean and std of y', y.mean(), y.std())


        else:
            with open(path + y_file, 'rb') as f:
                y = pickle.load(f).to_numpy().flatten()

    if only_over and not regresion:
        y[y>1] = 0

    np.random.seed(seed)
    y_shape_for_month_cheak = y.shape[0]
    
    if down_sample_rate != 0:
        X, y, og_shape = down_sample(X, y, down_sample_rate, og_shape)

    X = torch.tensor(X.values.reshape(-1, og_shape[0], og_shape[1], og_shape[2]),dtype=dtype)
    y = torch.tensor(y,dtype=dtype)


    if months is not None:
        y = chosse_months(y, months, None, flatten)
        print('Shape of X before month picking', X.shape)
        X = chosse_months(X, months, og_shape, flatten)

    if time_shift:
        if flatten:
            if 0 in months:
                print('Time shift only works for winter data')
                X = X.reshape(-1, len(months), torch.prod(og_shape))
                X = X[:,[0,1,3,4],:].reshape(-1, torch.prod(og_shape))[:-1]
                y = y.reshape(-1, len(months))[:,[0,1,2,4]].flatten()[1:]
                print('Shape of X after time shift', X.shape)
                print('Shape of y after time shift', y.shape)
            else:
                sys.exit('Time shift not implemented for non winter data')
        else:
            sys.exit('Time shift not implemented for non-flatten data')
    if crop is not None:
        if verbose:
            print('Cropping the data', crop)
        X = X[:,:,crop[0].start:crop[0].stop,crop[1].start:crop[1].stop]
    
    if verbose:
        print('lengths of y', len(y))
        if months is not None:
            print('Check of months picking ', 12*len(y)/len(months), '=', y_shape_for_month_cheak)
    pos_weight = len(y) / torch.sum(y)
    
    total_indices = np.arange(len(y))
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(X, y, total_indices, test_size = split, random_state = seed)

    X_test, X_val, y_test, y_val, idx_test, idx_val = train_test_split(X_val, y_val, idx_val, test_size = 0.5, random_state = seed)
    print('Idx for test', idx_test)
    np.save('idx_test.npy', idx_test)
    if test:
        

        if transformer is not None:

            test_loader = transform_data(X,y, X_test, y_test, transformer, DDP = DDP, verbose = verbose, kwarg_Dataloader = kwarg_Dataloader)
            return test_loader, pos_weight


        if flatten:
            X_test = X_test.reshape(-1, og_shape[0]*og_shape[1]*og_shape[2])
            
        return X_test, y_test, idx_test, pos_weight

    if transformer is not None:        
        if verbose:
            print('Shape of X_train', X_train.shape)
            print('Shape of X_val', X_val.shape)
            print('Shape of y_train', y_train.shape)
            print('Shape of y_val', y_val.shape)
        train_dataloader = transform_data(X, y, X_train, y_train, transformer, DDP = DDP, verbose = verbose, kwarg_Dataloader = kwarg_Dataloader)
        val_dataloader = transform_data(X, y, X_val, y_val, transformer, DDP = DDP, verbose = verbose, kwarg_Dataloader = kwarg_Dataloader)
        return train_dataloader, val_dataloader, pos_weight

    if flatten:
        X_train = X_train.reshape(-1, og_shape[0]*og_shape[1]*og_shape[2])
        X_val = X_val.reshape(-1, og_shape[0]*og_shape[1]*og_shape[2])

    if return_idx:
        return X_train, y_train, X_val, y_val, pos_weight, idx_train, idx_val
    return (X_train, y_train), (X_val, y_val), pos_weight





if __name__ == '__main__':
    print('Runing the preproccsing for finding regions')

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help = 'path to the data file to be loaded', required= True)
    parser.add_argument('-c','--data_class', help = 'what class that should be used', required= True)
    parser.add_argument('-v','--verbosity', help = 'verbosity level, with 0 only xarray output, 1 gives plots', default= 0 )

    args = parser.parse_args()

    data_class = globals()[args.data_class](args.path)

    import matplotlib.pyplot as plt
    target_region = (slice(-4.5,15),slice(62.5,53))
    predict_region = (slice(-64.5,35),slice(69.5,30))
    
    data_class.rescale_long()
    data_class.reformat_time()
    print('='*20,'target_region','='*20)
    print(data_class.xr_data['t2m'].sel(lon = target_region[0] , lat = target_region[1]).isel(time = 0 ))
    if int(args.verbosity) > 0:
        data_class.xr_data['t2m'].sel(lon = target_region[0] , lat = target_region[1]).isel(time = 0 )[0].plot()
        plt.show()
    
    print('='*20,'predict_region','='*20)
    print(data_class.xr_data['t2m'].sel(lon = predict_region[0] , lat = predict_region[1]).isel(time = 0 ))
    if int(args.verbosity) > 0:
        data_class.xr_data['t2m'].sel(lon = predict_region[0] , lat = predict_region[1]).isel(time = 0 )[0].plot()
        plt.show()
    
