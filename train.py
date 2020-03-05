import math
import torch
import argparse
from loss import *
from data import *
from model import *
from tqdm import tqdm

parser = argparse.ArgumentParser(description = 'train')

parser.add_argument('--epoch', type = int, default = 2500, help = 'config file')
parser.add_argument('--bs', type = int, default = 2, help = 'config file')
parser.add_argument('--classes', type = int, default = 2, help = 'config file')
parser.add_argument('--lr', type = float, default = 0.1, help = 'config file')
parser.add_argument('--test', type = bool, default = True, help = 'config file')

args = parser.parse_args()

def train():
	epoch=args.epoch
	batch_size=args.bs
	lr=args.lr
	classes=args.classes
	criterion=TL_Loss()
	model=Siamese_TU_Net_TL(classes).cuda()
	_,_,_,traindata=load_data('./data/TNO/',0,1,True)
	opt=torch.optim.Adam(model.parameters(),lr)
	for i in range(1,epoch+1):
		allloss=0
		with tqdm(total=math.ceil(traindata/batch_size)) as train_bar:
			for j in range(0,math.ceil(traindata/batch_size)):
				train_bar.update(1)
				input1,input2,GT,_=load_data('./data/',j,batch_size,True)
				f11,f12,t1,f21,f22,t2=model(input1,input2,input1,input2)
				loss=criterion(GT,t1,GT,t2,f11,f12,f21,f22)
				opt.zero_grad()
				loss.backward()
				opt.step()
				allloss=allloss+loss
				if i//600-i/600==0:
					lr=lr/5
					opt=torch.optim.Adam(model.parameters(),lr)
				train_bar.set_description('epoch:%s loss:%.5f'%(i,allloss))
		torch.save(model,'./model/model'+str(i)+'.pth')

if __name__=='__main__':
	train()