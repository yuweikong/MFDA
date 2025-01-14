from wsgiref.simple_server import demo_app
import segmentation_models_pytorch as smp
from statistics import mode
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np
import scipy.io as io

from PIL import Image
import os
from torch.utils.data import DataLoader, Dataset, dataset, random_split
import torchvision
import cv2
from tqdm import tqdm
import time
from skimage.feature import hog
from skimage import exposure
import torchvision.transforms.functional as TF


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def norm(x):
    return ((x-x.min())/(x.max()-x.min()))

datas = []
image_base = 'data_source/train_data'   # 训练集路径

def generate_ellipse_mask(image_shape, major_axis, minor_axis):
    height, width = image_shape
    y, x = np.mgrid[:height, :width]
    center_x = width // 2
    center_y = height // 2
    # Calculate the distance from the center of the ellipse
    distance = ((x - center_x) / major_axis) ** 2 + ((y - center_y) / minor_axis) ** 2
    # Generate the elliptical mask
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[distance <= 1] = 1
    return mask

def apply_lowpass_filter(image):
    # Perform Fourier transform on the image
    f = np.fft.fft2(image)
    # Shift the zero frequency component to the center of the spectrum
    f_shifted = np.fft.fftshift(f)
    # Create a mask for the lowpass filter
    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    major_axis = 64
    minor_axis = 64
    mask = generate_ellipse_mask(mask.shape,major_axis,minor_axis)
    # Apply the mask to the shifted spectrum
    f_filtered = f_shifted * mask
    # Shift the result back to the original position
    f_filtered_shifted = np.fft.ifftshift(f_filtered)
    # Perform inverse Fourier transform to obtain the filtered image
    filtered_image = np.fft.ifft2(f_filtered_shifted)
    # Convert the complex-valued image to real-valued image
    filtered_image = np.abs(filtered_image)
    return filtered_image

def apply_highpass_filter(image):
    # Perform Fourier transform on the image
    f = np.fft.fft2(image)
    # Shift the zero frequency component to the center of the spectrum
    f_shifted = np.fft.fftshift(f)
    # Create a mask for the lowpass filter
    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    major_axis = 64
    minor_axis = 64
    mask = generate_ellipse_mask(mask.shape,major_axis,minor_axis)
    mask = ~mask
    f_filtered = f_shifted * mask
    f_filtered_shifted = np.fft.ifftshift(f_filtered)
    filtered_image = np.fft.ifft2(f_filtered_shifted)
    filtered_image = np.abs(filtered_image)
    return filtered_image

# from SiMMIM
class MaskGenerator:
    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        return mask
    

train_pretrain = 'data_source/pretrain'   # 训练集路径

train_list = []
for i in os.listdir(train_pretrain):
    train_list.append(train_pretrain+'/'+i)

class SimMIMDataset(torch.utils.data.Dataset): 
    def __init__(self,datalist = None):
        super(SimMIMDataset,self).__init__()
        self.imgs = []
        self.datalist=datalist[:] 
        self.mask_gen = MaskGenerator(256,32,4,0.6)

    def __getitem__(self, index):
        mask = torch.from_numpy(self.mask_gen())
        input = np.load(self.datalist[index])
        input = torch.from_numpy(input)[0:14,:,:]
        
        x = np.load(self.datalist[index])
        x = torch.from_numpy(x)
        x = x[0:14,:,:]

        x_flip = np.zeros_like(x)
        if torch.rand(1) < 0.5:
            for i in range(len(x_flip)):
                img_single = Image.fromarray((x[i,:,:]*4000).astype(np.int16))
                x_flip[i,:,:] = np.array(TF.hflip(img_single))/4000.0
            x = x_flip[:]

        if torch.rand(1) < 0.5:
            for i in range(len(x_flip)):
                img_single = Image.fromarray((x[i,:,:]*4000).astype(np.int16))
                x_flip[i,:,:] = np.array(TF.vflip(img_single))/4000.0
            x = x_flip[:]

        low_filter_img = np.zeros_like(x[0:2,:,:])
        for i in range(2):
            image = x[i,:,:]
            low_filter_img[i,:,:] = apply_lowpass_filter(image)
        low_filter_img = torch.from_numpy(low_filter_img)
        target = x[0:2,:,:]
        

        return x.float(),low_filter_img.float(),target.float(),mask
    def __len__(self):
        return len(self.datalist)

data = SimMIMDataset(datalist=train_list)

train_dataloader = DataLoader(data, batch_size=16, shuffle=True, num_workers=0)
test_dataloader = DataLoader(data, batch_size=16, shuffle=False, num_workers=0)


from MLFMIM import MLFMIM_encoder,MLFMIM
encoder = MLFMIM_encoder()
model = MLFMIM(encoder=encoder, encoder_stride=32)


optimizer = torch.optim.Adam(model.parameters(), lr=5e-4,weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000//20, gamma=0.9, verbose=True) 



net=model.to(device)

L1_loss = nn.L1Loss()
L2_loss = nn.MSELoss()

net.train()

tic = time.time()
for epoch in range(2000):
    epoch_loss = 0
    for batch_data in train_dataloader:
        # hr_input = batch_data[0].to(device)
        # lr_input = batch_data[1].to(device) 
        input_img = batch_data[0].to(device)
        low_hr_input = batch_data[1].to(device)
        high_hr_input = batch_data[2].to(device)
        target = batch_data[3].to(device) 
        mask = batch_data[4].to(device) 

        x_2,x_3,x_4 = model(input_img,mask)
        optimizer.zero_grad()
        mask = mask.repeat_interleave(4, 1).repeat_interleave(4, 2).unsqueeze(1).contiguous()
        mask = mask.repeat(1, 2, 1, 1)

        loss_x_2 = F.l1_loss(x_2, low_hr_input,reduction='none')
        loss_2 = (loss_x_2 * mask).sum() / (mask.sum() + 1e-5) / 2 
        + \
            0.5*L2_loss(low_hr_input, x_2) + 0.5*L1_loss(low_hr_input, x_2)

        loss_x_3 = F.l1_loss(x_3, low_hr_input,reduction='none')
        loss_3 = (loss_x_3 * mask).sum() / (mask.sum() + 1e-5) / 2 
        + \
            0.5*L2_loss(target, x_3) + 0.5*L1_loss(target, x_3)
        
        loss_x_4 = F.l1_loss(x_4, low_hr_input,reduction='none')
        loss_4 = (loss_x_4 * mask).sum() / (mask.sum() + 1e-5) / 2 
        + \
            0.5*L2_loss(target, x_4) + 0.5*L1_loss(target, x_4)

        loss = loss_2 + loss_3 + loss_4

        loss.backward()
        optimizer.step()
        epoch_loss = epoch_loss + loss.item()

    print('epoch',str(epoch),'--:',epoch_loss)



def save_model(model=None,epoch=0):
    output_dir = 'D:/chenyuhan/Sea_Ice_multi_task_learning/pretrain_model_weight'
    epoch_name = str(epoch)
    checkpoint_path = output_dir + '/MFDA_pretrain'
    to_save = {
        'model': model.state_dict(),
        'epoch': epoch,
    }
    torch.save(to_save, checkpoint_path)    

save_model(net,epoch=epoch)
toc = time.time()
print("Running Time: {:.2f}".format(toc-tic))
print("**************************************************")

