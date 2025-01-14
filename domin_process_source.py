import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import PIL.Image as Image
import torchvision.transforms as transforms
import xarray as xr
import cv2
import datetime
from dateutil import relativedelta
import re

# 设置参数
size_w=256;size_h=256;step=256
path_all_save = 'data_source/train_data'
scale_factor = 0.1

def get_norm_month(file_name):
    pattern = re.compile(r'\d{8}T\d{6}')
    # Search for the first match in the string
    match = re.search(pattern, file_name)
    first_date = match.group(0)
    # parse the date string into a datetime object
    date = datetime.datetime.strptime(first_date, "%Y%m%dT%H%M%S")
    # calculate the number of days between January 1st and the given date
    delta = relativedelta.relativedelta(date, datetime.datetime(date.year, 1, 1))
    # delta = (date - datetime.datetime(date.year, 1, 1)).days
    months = delta.months
    norm_months = 2*months/11-1
    return norm_months

def norm(x):
    return ((x-x.min())/(x.max()-x.min()))

def pad_image(image, target_size):
    channels, height, width = image.shape
    pad_height = target_size[1] - height
    pad_width = target_size[2] - width
    top_pad = pad_height // 2
    bottom_pad = pad_height - top_pad
    left_pad = pad_width // 2
    right_pad = pad_width - left_pad
    padded_image = np.pad(image, ((0, 0), (top_pad, bottom_pad), (left_pad, right_pad)), mode='constant')
    return padded_image

def pad_border(image, pad_size):
    pad_height = pad_size[0]
    pad_width = pad_size[1]
    padded_image = np.pad(image, ((0, 0), (pad_height, pad_height), (pad_width, pad_width)), mode='constant')
    return padded_image


def image_crop(img):
    # 进行阈值处理，将黑色部分设为255，其他部分设为0
    img = (img*255).astype(np.uint8)
    _, threshold = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    # 找到黑色部分的轮廓
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 确保至少找到一个轮廓
    if contours:
        # 初始化裁剪区域的参数
        x_min = img.shape[1]
        y_min = img.shape[0]
        x_max = 0
        y_max = 0
        # 遍历轮廓，找到黑色区域的边界
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)
        # 裁剪图像，只保留中间的有效遥感图像
        # cropped_image = image[y_min:y_max, x_min:x_max]
    else:
        print("No contours found.")
    return y_min,y_max,x_min,x_max

source = os.listdir('data_source/adaptdomin/train_source')
target = os.listdir('data_source/adaptdomin/train_target')

for data_name in source:
    scenename = data_name[:-9]+'_prep.nc'
    scene = xr.open_dataset('data_source/AI4Arctic Sea Ice Challenge Dataset/'+scenename)
    mm_SIC = scene['SIC'].values

    image = scene['nersc_sar_primary'].values
    original_height, original_width = image.shape[:2]
    sic_variables = ['SIC', 'SOD', 'FLOE']

    sar_variables=['nersc_sar_primary', 'nersc_sar_secondary']
    geo_varibles = ['sar_grid2d_latitude',
                'sar_grid2d_longitude']
    amsr_variables = [
            'btemp_18_7h', 'btemp_18_7v',
            'btemp_89_0h', 'btemp_89_0v'
            ]
    env_variables = ['u10m_rotated', 'v10m_rotated',
        't2m', 'tcwv', 'tclw']
    sic_values = scene[sic_variables].to_array().values
    sar_values = scene[sar_variables].to_array().values
    amsr_values = scene[amsr_variables].to_array().values
    env_values = scene[env_variables].to_array().values
    geo_values = scene[geo_varibles].to_array().values

    lat_array = scene['sar_grid2d_latitude'].values
    lat_array = (lat_array - lat_array.mean())/lat_array.std()
    inter_lat_array = torch.nn.functional.interpolate(input=torch.from_numpy(lat_array).view((1, 1, lat_array.shape[0], lat_array.shape[1])),
                                                        size=(int(original_height*scale_factor), int(original_width*scale_factor)),
                                                        mode='nearest')
    long_array = scene['sar_grid2d_longitude'].values
    lon_array = (long_array - long_array.mean())/long_array.std()
    inter_long_array = torch.nn.functional.interpolate(input=torch.from_numpy(lon_array).view((1, 1, lat_array.shape[0], lon_array.shape[1])),
                                                        size=(int(original_height*scale_factor), int(original_width*scale_factor)),
                                                        mode='nearest')
    
    scene_id = scene.attrs['scene_id']
    norm_time = get_norm_month(scene_id)
    time_array = torch.from_numpy(
        np.full((int(original_height*scale_factor), int(original_width*scale_factor)), norm_time)).unsqueeze(0).unsqueeze(0)
    
    resize_size = [int(scene['nersc_sar_primary'].values.shape[0]*scale_factor),int(scene['nersc_sar_primary'].values.shape[1]*scale_factor)]
    x = torch.cat((torch.nn.functional.interpolate(
                    input=torch.from_numpy(sar_values).unsqueeze(0),
                    size=resize_size, 
                    mode='bilinear',align_corners=True),

                    torch.nn.functional.interpolate(
                    input=torch.from_numpy(amsr_values).unsqueeze(0),
                    size=resize_size, 
                    mode='bilinear',align_corners=True),

                    torch.nn.functional.interpolate(
                    input=torch.from_numpy(env_values).unsqueeze(0),
                    size=resize_size, 
                    mode='bilinear',align_corners=True),

                    inter_lat_array,
                    inter_long_array,
                    time_array,
                    torch.nn.functional.interpolate(
                    input=torch.from_numpy(sic_values).unsqueeze(0),
                    size=resize_size, 
                    mode='nearest'),
                    ),axis=1)
    
    img = x[0]
    img = img.numpy()
    if img.shape[-2]<256:
        img = pad_image(img,(img.shape[0],256,img.shape[-1]))
    if img.shape[-1]<256: 
        img = pad_image(img,(img.shape[0],img.shape[-2],256))
    img = torch.from_numpy(img)
    for i in range(0,14):
        mask_sar = img[i]==0
        img[i,:,:] = norm(img[i,:,:])
        if i<2:
            img[i][mask_sar] = 0
        img[i,:,:]=(img[i,:,:]*8000).to(torch.int16)/8000.0

    img = img.numpy()
    size = img.shape
    i=0
    for h in range(0,size[1],step):
        star_h = h 
        for w in range(0,size[2],step):
            star_w = w 
            end_h = star_h + size_h 
            
            if end_h > size[1]:
                star_h = size[1] - size_h
                end_h = star_h + size_h
                i=i-1                   
            end_w = star_w + size_w 
            if end_w > size[2]:
                star_w = size[2] - size_w
                end_w = star_w + size_w
                i=i-1
            cropped = img[:,star_h:end_h, star_w:end_w]          
            i=i+1
            name_img = str(data_name[:-12]) + '_'+ str(star_h) +'_' + str(star_w)
            mask = cropped[-3]==255          
            try:
                if cropped[-3][~mask].max()<1e-5:
                    continue   
            except:
                continue     
            np.save('{}/{}.npy'.format(path_all_save,name_img),cropped)
            