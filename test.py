import os
from data import *
from model import *

def test(model_path,index):
	_,_,_,testdata=load_data('./data/',0,1,False)
	model=torch.load(model_path)
	model=model.cuda()
	model.eval()
	# for i in range(0,testdata):
	input1,input2,_,_=load_data('./data/',0,1,False)
	output=model(input1,input2)
	image_save(output,'./test/'+str(index)+'_'+str(0)+'.jpg')

if __name__=='__main__':
	modelname=os.listdir('./model/')
	for i in range(0,len(modelname)):
		test('./model/'+modelname[i],i)