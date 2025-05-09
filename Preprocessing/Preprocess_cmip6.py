import sys 
home_dir = '../../../'
sys.path.append(home_dir)
import preprocessing as prep
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


test_plots = True

path1a = '/data/projects/nckf/cmip6/historical/EC-Earth3/r'
path1b = 'i1p1f1/Amon/'
path2 = '_Amon_EC-Earth3_historical_r'

path3 = 'i1p1f1_gr_185001-201412.nc'
path4 = 'i1p1f1_gr_184912-201412.nc'
path5 = home_dir + 'Data/CMIP6/model/EC-earth/Merged/'

data_save_path = home_dir+'Data/CMIP6/'
#check if the paths
os.makedirs(data_save_path, exist_ok=True)

# 1 wrong order of data
# 25 wrong years
# 5,8,20 not in the dmi database
# problems with downloading 3 in correct format for zg from the servers
# ^ has double points at each lat
# 17 not correct with fitting a straight line
naughty_list = [1,3,5,8,17,20,22,25]

predict_region = (slice(30,69.5),slice(-64.5,35))

window_size = 30

target_region = (slice(50,54),slice(-62,-58)) # east_canada
target_region = (slice(58,62),slice(3,7)) # Bergen  
target_region = (slice(53,58),slice(7,15)) # Denmark

detrend = True


i = 2
print(path1a + str(i) + path1b+path2+str(i)+path3)
temp_data = prep.cmip6_data(path1a + str(i) + path1b, path2+str(i)+path3, path5, True)
temp_data.rescale_long(name_longitude = 'lon')
print(temp_data.xr_data.time)
if test_plots:
    temp_data.xr_data.ts.sel( lat=target_region[0],lon=target_region[1]).isel(time = 0).plot()
    plt.savefig('test_target_region.png')
    plt.close('all')
    temp_data.xr_data.ts.sel( lat=predict_region[0],lon=predict_region[1]).isel(time = 0).plot()
    plt.savefig('test_predict_region.png')
    plt.close('all')

print('done with test plots')

def rolling_mean(x):

    rolling_mean = x.rolling(time=window_size, center=True,min_periods = 1).mean()

    return rolling_mean

def rolling_std(x):

    rolling_std = x.rolling(time=window_size, center=True,min_periods = 1).std()

    return rolling_std


final_data = []
problem_list = [11,13,15]

R = 6371e3


for i in range(1,25):
    if i in naughty_list:
        continue
    print(path1a + str(i) + path1b + path2+str(i)+path3)
    post_path = path2+str(i)+path3
    time_slice = slice(window_size//2*12,-(window_size//2-1)*12)
    post_path = path2+str(i)+path3

    if i in problem_list:
        post_path = path2+str(i)+path4


    temp_data = prep.cmip6_data(path1a + str(i) + path1b, post_path, path5, True)
    temp_data.rescale_long(name_longitude = 'lon')
    
    if detrend:
        temp_data.detrend('pr', target_region)

    mean_pr = temp_data.xr_data.pr.sel(lat = target_region[0], lon = target_region[1]).mean(dim = ('lat','lon'))
    mean_pr_cut = mean_pr.isel(time = time_slice)
    month_mean_pr = mean_pr.groupby('time.month').apply(rolling_mean).isel(time = time_slice)
    month_mean_pr.chunk({'time':2})
    month_std_pr = mean_pr.groupby('time.month').apply(rolling_std).isel(time = time_slice)
    month_std_pr.chunk({'time':2})

    pair2 = (mean_pr_cut - month_mean_pr)/month_std_pr

    zg_500 = temp_data.xr_data['zg']
    lats = temp_data.xr_data.lat
    lons = temp_data.xr_data.lon
    cos_lat = np.cos(np.radians(lats))
    d_lat = lats[1]-lats[0]
    d_lon = lons[1]-lons[0]
    print('-- Testing gradient --')
    print('lons: ', lons)
    print('lats: ', lats)
    print('cos_lat: ', cos_lat)
    print('d_lat: ', d_lat)
    print('d_lon: ', d_lon)
    grad = 1/R * np.sqrt((zg_500.differentiate('lon')/(d_lon*cos_lat))**2 + (zg_500.differentiate('lat')/d_lat)**2)
    print(grad.values)
    print('-- Done testing gradient --')

    #add the gradient to the data
    temp_data.xr_data['zg_grad'] = grad


    pair0 = temp_data.rescale_prediction('ts', predict_region, time_slice, detrend = detrend)
    pair1 = temp_data.rescale_prediction('psl', predict_region, time_slice, detrend = detrend)
    pair3 = temp_data.rescale_prediction('zg', predict_region, time_slice, detrend = detrend)
    pair4 = temp_data.rescale_prediction('zg_grad', predict_region, time_slice, detrend = detrend)


    target = temp_data.make_rolling_mean_target('pr', region = target_region, window_size = window_size, chunk_size_local=136, detrend = detrend)
    combined = xr.combine_by_coords([pair0, pair1, pair2, pair3, pair4])
    
    combined['target'] = ('time', target.values)

    final_data.append(combined)

    print('done with ' + str(i))
    print(target)
    print(combined)
    print('-'*50)
    print('\n')



common_lat = final_data[0]['lat']

for i,ds in enumerate(final_data):
    final_data[i] = ds.assign_coords(lat=common_lat)


test = xr.concat(xr.align(*final_data, join='exact'), dim='ensemble')
del final_data



if detrend:
    processing_name = 'detrended_precep_and_normed'

else:
    processing_name = 'raw'

processing_name += '_'+'Bergen'

test.to_netcdf(data_save_path+'CMIP6_target_combined_' + processing_name + '.nc')

def save_var_to_disk(var, name, processing_name):
    X_temp = pd.DataFrame(var.values.reshape(-1,var.shape[-2]*var.shape[-1]))
    with open(data_save_path+'X_'+name+'_'+processing_name+'.pkl', 'wb') as f:
        pickle.dump(X_temp, f)
    del X_temp

save_var_to_disk(test.ts, 'ts', processing_name)
save_var_to_disk(test.psl, 'psl', processing_name)
save_var_to_disk(test.zg, 'zg', processing_name)
save_var_to_disk(test.zg_grad, 'zg_grad', processing_name)

y = pd.DataFrame(test.target.values)
with open(data_save_path+'y_'+processing_name+'.pkl', 'wb') as f:
    pickle.dump(y, f)
