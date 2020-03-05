import torch
import torch.nn as nn

def centercrop(tensor,size):
	[batch_size,C,W,H]=tensor.shape
	tensor=tensor[:,:,int(W/2-size/2):int(W/2+size/2),\
					int(H/2-size/2):int(H/2+size/2)]
	return tensor

class Contracting_units(nn.Module):
	def __init__(self,in_channels,out_channels):
		super(Contracting_units,self).__init__()
		self.Conv1=nn.Sequential(
			nn.Conv2d(in_channels,out_channels,
						kernel_size=3,stride=1),
			nn.ReLU())
		self.Conv2=nn.Sequential(
			nn.Conv2d(out_channels,out_channels,
						kernel_size=3,stride=1),
			nn.ReLU())
		self.Pool=nn.MaxPool2d(kernel_size=2,stride=2)

	def forward(self,x):
		x=self.Conv1(x)
		y=self.Conv2(x)
		x=self.Pool(y)
		return x,y

class Expansive_unit1(nn.Module):
	def __init__(self,in_channels,out_channels):
		super(Expansive_unit1,self).__init__()
		self.Conv1=nn.Sequential(
			nn.Conv2d(in_channels,out_channels,
						kernel_size=3,stride=1),
			nn.ReLU())
		self.Conv2=nn.Sequential(
			nn.Conv2d(out_channels,out_channels,
						kernel_size=3,stride=1),
			nn.ReLU())
		self.DC=nn.Sequential(
			nn.ConvTranspose2d(out_channels,
			out_channels,kernel_size=2,stride=2),
			nn.ReLU())

	def forward(self,x,y):
		x=self.Conv1(x)
		x=self.Conv2(x)
		x=self.DC(x)
		y=centercrop(y,x.shape[2])
		x=torch.cat((x,y),1)
		return x

class Expansive_unit2(nn.Module):
	def __init__(self,in_channels,out_channels,N):
		super(Expansive_unit2,self).__init__()
		self.Conv1=nn.Sequential(
			nn.Conv2d(in_channels,out_channels,
						kernel_size=3,stride=1),
			nn.ReLU())
		self.Conv2=nn.Sequential(
			nn.Conv2d(out_channels,out_channels,
						kernel_size=3,stride=1),
			nn.ReLU())
		self.Conv3=nn.Sequential(
			nn.Conv2d(out_channels,N,
						kernel_size=1,stride=1),
			nn.ReLU())

	def forward(self,x):
		x=self.Conv1(x)
		x=self.Conv2(x)
		x=self.Conv3(x)
		return x

class Expansive_unit3(nn.Module):
	def __init__(self,in_channels,out_channels):
		super(Expansive_unit3,self).__init__()
		self.Conv1=nn.Sequential(
			nn.Conv2d(in_channels,out_channels,
						kernel_size=3,stride=1),
			nn.ReLU())
		self.Conv2=nn.Sequential(
			nn.Conv2d(out_channels,out_channels,
						kernel_size=3,stride=1),
			nn.ReLU())

	def forward(self,x):
		x=self.Conv1(x)
		x=self.Conv2(x)
		return x

class TU_Net_BL(nn.Module):
	def __init__(self,N):
		super(TU_Net_BL,self).__init__()

		self.Unit1=Contracting_units(4,64)
		self.Unit2=Contracting_units(64,128)
		self.Unit3=Contracting_units(128,256)
		self.Unit4=Contracting_units(256,512)

		self.Unit5=Expansive_unit1(512,1024)
		self.Unit6=Expansive_unit1(1536,512)
		self.Unit7=Expansive_unit1(768,256)
		self.Unit8=Expansive_unit1(384,128)
		self.Unit9=Expansive_unit2(192,64,N)

	def forward(self,x1,x2):

		x=torch.cat((x1,x2),1)
		x,y1=self.Unit1(x)
		x,y2=self.Unit2(x)
		x,y3=self.Unit3(x)
		feature,y4=self.Unit4(x)

		x=self.Unit5(feature,y4)
		x=self.Unit6(x,y3)
		x=self.Unit7(x,y2)
		x=self.Unit8(x,y1)
		x=self.Unit9(x)

		return feature,x

class TU_Net_ML(nn.Module):
	def __init__(self,N):
		super(TU_Net_ML,self).__init__()

		self.Unit1_1=Contracting_units(3,64)
		self.Unit1_2=Contracting_units(64,128)
		self.Unit1_3=Contracting_units(128,256)
		self.Unit1_4=Contracting_units(256,512)

		self.Unit2_1=Contracting_units(1,64)
		self.Unit2_2=Contracting_units(64,128)
		self.Unit2_3=Contracting_units(128,256)
		self.Unit2_4=Contracting_units(256,512)

		self.Unit5=Expansive_unit1(1024,512)
		self.Unit6=Expansive_unit1(1536,256)
		self.Unit7=Expansive_unit1(768,128)
		self.Unit8=Expansive_unit1(384,64)
		self.Unit9=Expansive_unit2(192,64,N)

	def forward(self,x1,x2):

		x1,y1_1=self.Unit1_1(x1)
		x1,y1_2=self.Unit1_2(x1)
		x1,y1_3=self.Unit1_3(x1)
		x1,y1_4=self.Unit1_4(x1)

		x2,y2_1=self.Unit2_1(x2)
		x2,y2_2=self.Unit2_2(x2)
		x2,y2_3=self.Unit2_3(x2)
		x2,y2_4=self.Unit2_4(x2)

		feature=torch.cat((x1,x2),1)
		y1=torch.cat((y1_1,y2_1),1)
		y2=torch.cat((y1_2,y2_2),1)
		y3=torch.cat((y1_3,y2_3),1)
		y4=torch.cat((y1_4,y2_4),1)

		x=self.Unit5(feature,y4)
		x=self.Unit6(x,y3)
		x=self.Unit7(x,y2)
		x=self.Unit8(x,y1)
		x=self.Unit9(x)

		return feature,x

class TU_Net_TL(nn.Module):
	def __init__(self,N):
		super(TU_Net_TL,self).__init__()

		self.Unit1_1=Contracting_units(3,64)
		self.Unit1_2=Contracting_units(64,128)
		self.Unit1_3=Contracting_units(128,256)
		self.Unit1_4=Contracting_units(256,512)

		self.Unit1_5=Expansive_unit1(512,512)
		self.Unit1_6=Expansive_unit1(1024,256)
		self.Unit1_7=Expansive_unit1(512,128)
		self.Unit1_8=Expansive_unit1(256,64)
		self.Unit1_9=Expansive_unit3(128,64)

		self.Unit2_1=Contracting_units(1,64)
		self.Unit2_2=Contracting_units(64,128)
		self.Unit2_3=Contracting_units(128,256)
		self.Unit2_4=Contracting_units(256,512)

		self.Unit2_5=Expansive_unit1(512,512)
		self.Unit2_6=Expansive_unit1(1024,256)
		self.Unit2_7=Expansive_unit1(512,128)
		self.Unit2_8=Expansive_unit1(256,64)
		self.Unit2_9=Expansive_unit3(128,64)

		self.Conv1x1=nn.Sequential(
			nn.Conv2d(in_channels=128,out_channels=N,
						kernel_size=1,stride=1),
			nn.ReLU())

	def forward(self,x1,x2):

		x1,y1_1=self.Unit1_1(x1)
		x1,y1_2=self.Unit1_2(x1)
		x1,y1_3=self.Unit1_3(x1)
		feature1,y1_4=self.Unit1_4(x1)

		x1=self.Unit1_5(feature1,y1_4)
		x1=self.Unit1_6(x1,y1_3)
		x1=self.Unit1_7(x1,y1_2)
		x1=self.Unit1_8(x1,y1_1)
		x1=self.Unit1_9(x1)

		x2,y2_1=self.Unit2_1(x2)
		x2,y2_2=self.Unit2_2(x2)
		x2,y2_3=self.Unit2_3(x2)
		feature2,y2_4=self.Unit2_4(x2)

		x2=self.Unit2_5(feature2,y2_4)
		x2=self.Unit2_6(x2,y2_3)
		x2=self.Unit2_7(x2,y2_2)
		x2=self.Unit2_8(x2,y2_1)
		x2=self.Unit2_9(x2)

		x=torch.cat((x1,x2),1)
		x=self.Conv1x1(x)

		return feature1,feature2,x

class Siamese_TU_Net_BL(nn.Module):
	def __init__(self,N):
		super(Siamese_TU_Net_BL,self).__init__()
		self.T1=TU_Net_BL(N)
		self.T2=TU_Net_BL(N)

	def forward(self,t1_1,t1_2,t2_1,t2_2):

		feature1,t1=self.T1(t1_1,t1_2)
		feature2,t2=self.T1(t2_1,t2_2)

		return feature1,t1,feature2,t2

class Siamese_TU_Net_ML(nn.Module):
	def __init__(self,N):
		super(Siamese_TU_Net_ML,self).__init__()
		self.T1=TU_Net_ML(N)
		self.T2=TU_Net_ML(N)

	def forward(self,t1_1,t1_2,t2_1,t2_2):

		feature1,t1=self.T1(t1_1,t1_2)
		feature2,t2=self.T1(t2_1,t2_2)

		return feature1,t1,feature2,t2

class Siamese_TU_Net_TL(nn.Module):
	def __init__(self,N):
		super(Siamese_TU_Net_TL,self).__init__()
		self.T1=TU_Net_TL(N)
		self.T2=TU_Net_TL(N)

	def forward(self,t1_1,t1_2,t2_1,t2_2):

		feature1_1,feature1_2,t1=self.T1(t1_1,t1_2)
		feature2_1,feature2_2,t2=self.T1(t2_1,t2_2)

		return feature1_1,feature1_2,t1,feature2_1,feature2_2,t2

if __name__=='__main__':
	x1=torch.rand((1,3,572,572))
	x2=torch.rand((1,1,572,572))
	model=TU_Net_TL(3)
	output=model(x1,x2)
	print(output.shape)