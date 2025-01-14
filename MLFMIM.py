from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from MFDA import MFDA_Base,SABlock_Windows
import numpy as np
import PIL.Image as Image
# from torch.utils.data import DataLoader, Dataset
import os

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
    

class SimMIMDataset(torch.utils.data.Dataset): 
    def __init__(self,transform=None,path='data_source/train_data'):
        super(SimMIMDataset,self).__init__()
        self.imgs = []
        self.path = path
        self.transform = transform
        self.datalist=os.listdir(path) 
    def __getitem__(self, index):
        img = np.load(self.path+'/'+self.datalist[index])
        low_filter_img = np.zeros_like(img[:2,:,:])
        img = torch.from_numpy(img)
        # img=dataclip(img)
        hr_img = img[:2,:,:]
        lr_img = img[2:12,:,:]
        
        for i in range(len(low_filter_img)):
            image = hr_img[i,:,:]
            low_filter_img[:,:,i] = apply_lowpass_filter(image)
        low_filter_img = torch.from_numpy(low_filter_img)
        
        return hr_img.float(),lr_img.float(),low_filter_img.float()



class MLFMIM_encoder(MFDA_Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim=32
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 32))
        trunc_normal_(self.mask_token, mean=0., std=.02)


    def forward(self, hr_img,lr_img, mask):
        a = hr_img
        b = lr_img
        b = F.interpolate(b, size=(64,64), mode='bilinear', align_corners=True)
        
        encoder_list = []

        a = self.patch_embed1(a)
        a = self.pos_drop(a)
        B, C, H, W = a.shape
        a = a.flatten(2).transpose(1, 2)

        b = self.patch_amsr_embed1(b)
        b = self.pos_drop(b)
        assert mask is not None
        B, L, _ = a.shape

        mask_tokens = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)

        # 计算a的均值
        # print('a:',x.shape)
        # print('w:',w.shape)
        # print('mask_tokens:',mask_tokens.shape)

        # x[w.repeat(1, 1, x.shape[-1])==0] = x.mean() + x.std()#赋值
        # print('a:',x)
        # print('w:',w)
        # print('mask tokens:',mask_tokens)
            # 计算x的均值
        # a_mean = a.mean(dim=1, keepdim=True)
        # a = a * (1. - w) + mask_tokens * w * a_mean
        a = a * (1. - w) + mask_tokens * w
        a = a.transpose(1, 2).reshape(B, C, H, W)
        # a = a + b

        for blk in self.blocks1:
            a = blk(a)
        for blk in self.blocks_amsr_1:
            b = blk(b)
        a = a + b
        out_a= self.norm1(a.permute(0, 2, 3, 1))
        encoder_list.append(out_a.permute(0, 3, 1, 2))

        a = self.patch_embed2(a)
        b = self.patch_amsr_embed2(b)
        for blk in self.blocks2:
            a = blk(a)
        for blk in self.blocks_amsr_2:
            b = blk(b)
        a = a + b

        out_a= self.norm2(a.permute(0, 2, 3, 1))
        encoder_list.append(out_a.permute(0, 3, 1, 2))

        a = self.patch_embed3(a)
        b = self.patch_amsr_embed3(b)
        for blk in self.blocks3:
            a = blk(a)
        for blk in self.blocks_amsr_3:
            b = blk(b)
        a = a + b
        out_a= self.norm3(a.permute(0, 2, 3, 1))
        encoder_list.append(out_a.permute(0, 3, 1, 2))

        a = self.patch_embed4(a)
        b = self.patch_amsr_embed4(b)
        for blk in self.blocks4:
            a = blk(a)
        for blk in self.blocks_amsr_4:
            b = blk(b)
        a = a + b
        out_a= self.norm4(a.permute(0, 2, 3, 1))
        encoder_list.append(out_a.permute(0, 3, 1, 2))


        return encoder_list[-3],encoder_list[-2],encoder_list[-1]
    @torch.jit.ignore
    def no_weight_decay(self):
        return super().no_weight_decay() | {'mask_token'}

class MLFMIM(nn.Module):
    def __init__(self, encoder, encoder_stride):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride
        self.embed_dims = [32, 64, 128, 256]
        self.decoder_x_2 = nn.Sequential(
            # SABlock(self.embed_dims[1], 4, 0.1),
            # SABlock(self.embed_dims[1], 4, 0.1),
            SABlock_Windows(dim=self.embed_dims[1], num_heads=4, window_size=4, qkv_bias=True,drop_path=0.1),
            SABlock_Windows(dim=self.embed_dims[1], num_heads=4, window_size=4, qkv_bias=True,drop_path=0.1),
            nn.Conv2d(
                in_channels=self.embed_dims[1],
                out_channels=8 ** 2 * 2, kernel_size=1),
            nn.PixelShuffle(8),
        )
        self.decoder_x_3 = nn.Sequential(
            SABlock_Windows(dim=self.embed_dims[2], num_heads=4, window_size=4, qkv_bias=True,drop_path=0.1),
            SABlock_Windows(dim=self.embed_dims[2], num_heads=4, window_size=4, qkv_bias=True,drop_path=0.1),
            nn.Conv2d(
                in_channels=self.embed_dims[2],
                out_channels=16 ** 2 * 2, kernel_size=1),
            nn.PixelShuffle(16),
        )
        self.decoder_x_4 = nn.Sequential(
            SABlock_Windows(dim=self.embed_dims[3], num_heads=4, window_size=4, qkv_bias=True,drop_path=0.1),
            SABlock_Windows(dim=self.embed_dims[3], num_heads=4, window_size=4, qkv_bias=True,drop_path=0.1),
            nn.Conv2d(
                in_channels=self.embed_dims[3],
                out_channels=32 ** 2 * 2, kernel_size=1),
            nn.PixelShuffle(32),
        )
    def forward(self, hr, lr, mask):
        x_2,x_3,x_4 = self.encoder(hr,lr, mask)
        rec_x2 = self.decoder_x_2(x_2)
        rec_x3 = self.decoder_x_3(x_3)
        rec_x4 = self.decoder_x_4(x_4)
        return rec_x2,rec_x3,rec_x4
    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}    


def save_model(model=None,epoch=0):
    output_dir = 'Sea_Ice_multi_task_learning/pretrain_model_weight'
    epoch_name = str(epoch)
    checkpoint_path = output_dir + '/MLFMIM_weight'
    to_save = {
        'model': model.state_dict(),
        'epoch': epoch,
    }
    torch.save(to_save, checkpoint_path)    


if __name__ == '__main__':
    gen = MaskGenerator(256,16,4,0.6)
    mask = gen()
    Image.fromarray((mask*255).astype(np.uint8))
    hr = torch.rand(1,2,256,256)
    lr = torch.rand(1,12,256,256)
    mask = torch.from_numpy(mask)
    mask = mask.unsqueeze(0)
    
    encoder = MLFMIM_encoder()
    model = MLFMIM(encoder=encoder, encoder_stride=32)

    x1,x2,x3 = model(hr,lr,mask)

    
