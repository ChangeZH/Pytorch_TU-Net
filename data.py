import os
import torch
import argparse
import numpy as np
import torchvision
from loss import *
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms

loader = transforms.Compose([transforms.ToTensor()])  
unloader = transforms.ToPILImage()

def image_save(tensor,name):
	image=tensor.cpu().clone()
	image=image.squeeze(0)
	image=unloader(image)
	image.save(name)

def image_loader(image_name,shape=572):
	img = Image.open(image_name)#.convert('RGB')
	image = img.resize((shape,shape))
	image = loader(image).unsqueeze(0)
	image=image.to('cuda', torch.float32)
	if image.shape[1]>1:
		image=image[:,0,:,:]
		image=torch.reshape(image,(1,1,shape,shape))
	return image

def load_data(path,batch,batch_size,if_train):
	dirname=os.listdir(path)
	imgname=[]
	for i in dirname:
		img=os.listdir(path+i)
		if if_train:
			img=[path+i+'/'+img[0],path+i+'/'+img[1],path+i+'/'+img[2]]
		else:
			img=[path+i+'/'+img[0],path+i+'/'+img[1]]
		imgname.append(img)
	for i in range(batch*batch_size,min(len(imgname),(batch+1)*batch_size)):
		if i==batch*batch_size:
			train_data1=image_loader(imgname[i][0],572)
			train_data2=image_loader(imgname[i][1],572)
			if if_train:
				GT=image_loader(imgname[i][2],388)
		else:
			data1=image_loader(imgname[i][0],572)
			data2=image_loader(imgname[i][1],572)
			if if_train:
				gt=image_loader(imgname[i][2],388)
			train_data1=torch.cat((train_data1,data1),0)
			train_data2=torch.cat((train_data2,data2),0)
			if if_train:
				GT=torch.cat((GT,gt),0)
	if if_train:
		return train_data1.cuda(),train_data2.cuda(),GT.cuda(),len(imgname)
	else:
		return train_data1.cuda(),train_data2.cuda(),len(imgname)

if __name__=='__main__':
	img1,img2,GT,img=load_data('./data/',0,1,True)
	print(img1.shape,img2.shape,GT.shape)