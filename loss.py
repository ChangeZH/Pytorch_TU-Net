import torch
import torch.nn as nn
from torch.nn import functional as F

class CE_Loss(nn.Module):
	def __init__(self):
		super(CE_Loss,self).__init__()

	def forward(self,y,x):
		S=x.shape[2]*x.shape[3]
		x=torch.log(x)
		return -torch.sum(x*y)/S

class MSE_Loss(nn.Module):
	def __init__(self):
		super(MSE_Loss,self).__init__()

	def forward(self,x1,x2):
		return torch.pow((x1-x2),2).mean()

class BL_Loss(nn.Module):
	def __init__(self,lam=200):
		super(BL_Loss, self).__init__()
		self.lam=lam
		self.CE=CE_Loss()
		self.MSE=MSE_Loss()

	def forward(self,y1,x1,y2,x2,t1,t2):
		CE1=self.CE(y1,x1)
		CE2=self.CE(y2,x2)
		MSE=self.MSE(t1,t2)
		return CE1+CE2+self.lam*MSE

class ML_Loss(nn.Module):
	def __init__(self,lam=200):
		super(ML_Loss, self).__init__()
		self.lam=lam
		self.CE=CE_Loss()
		self.MSE=MSE_Loss()

	def forward(self,y1,x1,y2,x2,t1,t2):
		CE1=self.CE(y1,x1)
		CE2=self.CE(y2,x2)
		MSE=self.MSE(t1,t2)
		return CE1+CE2+self.lam*MSE

class TL_Loss(nn.Module):
	def __init__(self,lam=200):
		super(TL_Loss, self).__init__()
		self.lam=lam
		self.CE=CE_Loss()
		self.MSE=MSE_Loss()

	def forward(self,y1,x1,y2,x2,t1_1,t1_2,t2_1,t2_2):
		CE1=self.CE(y1,x1)
		CE2=self.CE(y2,x2)
		MSE1=self.MSE(t1_1,t2_1)
		MSE2=self.MSE(t1_2,t2_2)
		return CE1+CE2+self.lam*(MSE1+MSE2)/2

if __name__=='__main__':
	criterion=BL_Loss()
	input=torch.rand((1,3,320,320))
	output=torch.rand((1,3,320,320))
	loss=criterion(output,input,output,input,output,input)
	print(loss.item())