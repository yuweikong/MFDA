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
import torch.optim as optim
import os
from torch.utils.data import DataLoader, Dataset, dataset, random_split
import torchvision
import cv2
from tqdm import tqdm
import torchvision.transforms.functional as TF
from discriminator import FCDiscriminator, OutspaceDiscriminator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
os.chdir('D:/chenyuhan/Sea_Ice_multi_task_learning')

def norm(x):
    return ((x-x.min())/(x.max()-x.min()))

datas = []
train_source = 'data_source/train_data'   # 训练集路径
train_target = 'data_source/target_data'
test_source = 'data_source/test_data'

train_list = []
for i in os.listdir(train_source):
    train_list.append(train_source+'/'+i)

target_list = []
for i in os.listdir(train_target):
    target_list.append(train_target+'/'+i)


class MyDataset(Dataset): 
    def __init__(self,datalist = None):
                            
        super(MyDataset,self).__init__()
        self.imgs = []
        self.datalist=datalist[:]
    def __getitem__(self, index):
        img = np.load(self.datalist[index])
        ignore = img[-3:-2,:,:]==255
        img[-3:-2,:,:][img[-3:-2,:,:]==255]=0
        img[-2:-1,:,:][img[-2:-1,:,:]==255]=0
        img[-1:,:,:][img[-1:,:,:]==255]=0
        x = img[:-3,:,:]
        y = img[-3:,:,:]

        x_flip = np.zeros_like(x)
        y_flip = np.zeros_like(y)

        if torch.rand(1) < 0.5:
            for i in range(len(x_flip)):
                # img_single = Image.fromarray((norm(x[i,:,:])*255).astype(np.int16))
                img_single = Image.fromarray((x[i,:,:]*4000).astype(np.int16))
                x_flip[i,:,:] = np.array(TF.hflip(img_single))/4000.0
            for i in range(len(y_flip)):
                y_single = Image.fromarray((y[i,:,:]).astype(np.uint8))
                y_flip[i,:,:] = np.array(TF.hflip(y_single))  
            x = x_flip[:]
            y = y_flip[:]

        if torch.rand(1) < 0.5:
            for i in range(len(x_flip)):
                # img_single = Image.fromarray((norm(x[i,:,:])*255).astype(np.int16))
                img_single = Image.fromarray((x[i,:,:]*4000).astype(np.int16))
                x_flip[i,:,:] = np.array(TF.vflip(img_single))/4000.0
            for i in range(len(y_flip)):
                y_single = Image.fromarray((y[i,:,:]).astype(np.uint8))
                y_flip[i,:,:] = np.array(TF.vflip(y_single))  
            x = x_flip[:]
            y = y_flip[:]   

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        SIC = y[0,:,:]
        SOD = y[1,:,:]
        FLOE = y[2,:,:]
        water = SIC[:]
        water[water>0]=1
        water[ignore]=255

        ignore = SIC==0
        SIC[SIC>1]=2
        SIC[SIC==1]=1
        SIC[ignore]=0

        labels = {
            'SIC':SIC.long(),
            'SOD':SOD.long(),
            'FLOE':FLOE.long(),
            'water':water.long()
        }
        return x.float(),labels
    
    def __len__(self):
        return len(self.datalist)
    

source_data=MyDataset(datalist=train_list)
source_dataloader = DataLoader(source_data, batch_size=16, shuffle=True, num_workers=0)

target_data=MyDataset(datalist=target_list)
target_dataloader = DataLoader(target_data, batch_size=16, shuffle=True, num_workers=0)

# discriminator attention module
num_class_list = [480, 11, 6, 7]
model_DA = nn.ModuleList(
    [FCDiscriminator(num_classes=num_class_list[0]).cuda(),
        OutspaceDiscriminator(num_classes=num_class_list[1]).cuda(),
        OutspaceDiscriminator(num_classes=num_class_list[2]).cuda(),
        OutspaceDiscriminator(num_classes=num_class_list[3]).cuda()
        ]
)

optimizer_D = torch.optim.Adam(model_DA.parameters(), lr=1e-4, betas=(0.9, 0.99),weight_decay=0.01)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer_D, 50, gamma=0.5, last_epoch=-1)

from MFDA import MFDA_Base
model = MFDA_Base()
############################################################################################
###############PreTrain weight
pretrained_dict = torch.load('pretrain_model_weight/'+'MFDA_pretrain_base', map_location=lambda storage,loc:storage.cpu())['model']
# model.load_state_dict(pretrained_dict, strict=False)
# # 获取模型和权重文件中的参数名称
model_state_dict = model.state_dict()
pretrained_state_dict = pretrained_dict
# 遍历模型参数,如果在预训练权重文件中存在且大小一致,则载入该权重
for name, param in model_state_dict.items():
    # name = 'encoder.'+name
    pretrain_name = 'encoder.'+name
    if pretrain_name in pretrained_state_dict and param.size() == pretrained_state_dict[pretrain_name].size():
        model_state_dict[name] = pretrained_state_dict[pretrain_name]
    else:
        print(f"Skipping parameter {name}")
# 加载修改后的权重
model.load_state_dict(model_state_dict)
############################################################################################


net=model.to(device)


class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing=0.0, num_classes=10, ignore_index=-100):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing

    def forward(self, input, target):
        target_one_hot = torch.zeros_like(input)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)

        smooth_target = target_one_hot * self.confidence + (1 - target_one_hot) * self.smoothing / (self.num_classes - 1)

        mask = (target != self.ignore_index).float()
        smooth_target = smooth_target * mask.unsqueeze(1)

        log_prob = torch.nn.functional.log_softmax(input, dim=1)
        loss = torch.sum(-smooth_target * log_prob, dim=1) * mask
        return loss.mean()

loss_func_sic = LabelSmoothingCrossEntropyLoss(ignore_index=0,smoothing=0.1,num_classes=11)
loss_func_sod = LabelSmoothingCrossEntropyLoss(ignore_index=0,smoothing=0.1,num_classes=6)
loss_func_floe = LabelSmoothingCrossEntropyLoss(ignore_index=0,smoothing=0.1,num_classes=7)
loss_func_water = LabelSmoothingCrossEntropyLoss(ignore_index=255,smoothing=0.1,num_classes=2)

criterion_mse = nn.MSELoss()

optimizer = optim.AdamW(net.parameters(), lr=1e-4,weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3000//30, gamma=0.5, last_epoch=-1)


total_loss = 0
loss_number=0
last_loss = 100

source_label = 0
target_label = 1

PATH='model_weight/'
name_weight = 'MFDA'

net.train()

for epoch in tqdm(iterable=range(3000), position=0):
    model.train()
    model_DA[0].train()
    model_DA[1].train()
    optimizer.zero_grad()
    optimizer_D.zero_grad()

    # train G
    # don't accumulate grads in D
    for param in model_DA.parameters():
        param.requires_grad = False
    # train with source
    try:
        _, traindata = next(enumerate(source_dataloader))
    except StopIteration:
        trainloader_iter = iter(source_dataloader)
        _, traindata = next(trainloader_iter)
    
    inputs,label = traindata
    inputs = inputs.cuda()

    feat_source, pred_source = model(inputs,model_DA,'source')

    loss = 2*loss_func_sic(input=pred_source['SIC'].squeeze(1), target=(label['SIC'].squeeze(1)).to(device)) + \
            2*loss_func_sod(input=pred_source['SOD'].squeeze(1), target=(label['SOD'].squeeze(1)).to(device)) + \
            loss_func_floe(input=pred_source['FLOE'].squeeze(1), target=(label['FLOE'].squeeze(1)).to(device)) + \
            loss_func_water(input=pred_source['water'].squeeze(1), target=(label['water'].squeeze(1)).to(device))

    loss.backward()    

    # train with target
    try:
        _, validdata = next(enumerate(target_dataloader))
    except StopIteration:
        targetloader_iter = iter(target_dataloader)
        _, validdata = next(targetloader_iter)

    inputs,label_val = validdata
    inputs = inputs.cuda()
    # label_val  = label_val.cuda()
    feat_target, pred_target = model(inputs, model_DA, 'target')

    loss_adv = 0
    D_out = model_DA[0](feat_target)
    loss_adv += criterion_mse(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).cuda())
    
    D_out = model_DA[1](F.softmax(pred_target['SIC'], dim=1))
    loss_adv += criterion_mse(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).cuda())
    D_out = model_DA[2](F.softmax(pred_target['SOD'], dim=1))
    loss_adv += criterion_mse(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).cuda())
    D_out = model_DA[3](F.softmax(pred_target['FLOE'], dim=1))
    loss_adv += criterion_mse(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).cuda())
    loss_adv = loss_adv * 0.01
    loss_adv.backward()
    optimizer.step()
    # train D
    # bring back requires_grad
    for param in model_DA.parameters():
        param.requires_grad = True

    # train with source
    loss_D_source = 0
    D_out_source = model_DA[0](feat_source.detach())
    loss_D_source += criterion_mse(D_out_source,
                                    torch.FloatTensor(D_out_source.data.size()).fill_(source_label).cuda())
    D_out_source = model_DA[1](F.softmax(pred_target['SIC'].detach(), dim=1))
    loss_D_source += criterion_mse(D_out_source,
                                    torch.FloatTensor(D_out_source.data.size()).fill_(source_label).cuda())
    D_out_source = model_DA[2](F.softmax(pred_target['SOD'].detach(), dim=1))
    loss_D_source += criterion_mse(D_out_source,
                                    torch.FloatTensor(D_out_source.data.size()).fill_(source_label).cuda())
    D_out_source = model_DA[3](F.softmax(pred_target['FLOE'].detach(), dim=1))
    loss_D_source += criterion_mse(D_out_source,
                                    torch.FloatTensor(D_out_source.data.size()).fill_(source_label).cuda())
    loss_D_source.backward()

    # train with target
    loss_D_target = 0
    D_out_target = model_DA[0](feat_target.detach())
    loss_D_target += criterion_mse(D_out_target,
                                    torch.FloatTensor(D_out_target.data.size()).fill_(target_label).cuda())
    D_out_target = model_DA[1](F.softmax(pred_target['SIC'].detach(), dim=1))
    loss_D_target += criterion_mse(D_out_target,
                                    torch.FloatTensor(D_out_target.data.size()).fill_(target_label).cuda())
    D_out_target = model_DA[2](F.softmax(pred_target['SOD'].detach(), dim=1))
    loss_D_target += criterion_mse(D_out_target,
                                    torch.FloatTensor(D_out_target.data.size()).fill_(target_label).cuda())
    D_out_target = model_DA[3](F.softmax(pred_target['FLOE'].detach(), dim=1))
    loss_D_target += criterion_mse(D_out_target,
                                    torch.FloatTensor(D_out_target.data.size()).fill_(target_label).cuda())
    loss_D_target.backward()
    optimizer_D.step()
    total_loss += loss

    print("Epoch: {:03d}, loss_seg: {:.4f}, loss_adv: {:.4f}, loss_D_s: {:.4f}, loss_D_t: {:.4f}"
            .format(epoch + 1, loss, loss_adv, loss_D_source, loss_D_target))
    torch.save(model.state_dict(), PATH+name_weight+'_Check')
    torch.save(model_DA.state_dict(), PATH+name_weight+'DA_Check')

print('Finished Training')

