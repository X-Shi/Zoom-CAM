import torch
import argparse
import os
import cv2
import numpy as np
from torch.autograd import Variable
from torchvision import models

def get_args():
	model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith('__') and callable(models.__dict__[name]))
	parser = argparse.ArgumentParser()
	parser.add_argument('--input', help = 'Input path')
	parser.add_argument('--output', help = 'Output folder path')
	parser.add_argument('--target_index', type=int, default = -1, help = 'Index of the target category')
	parser.add_argument('--intermediate_layers', type = str, default = '1,3,6,8,11,13,15,18,20,22,25,27,29',help = 'Intermediate layers numbers in features of the model, use comma to split')
	parser.add_argument('--arch', default = 'vgg16', choices = model_names, help = 'Model architectures:'+'|'.join(model_names)+'(default:vgg16)')
	parser.add_argument('--model_path', default = None, help = 'Path to your pytorch model')
	args = parser.parse_args()
	return args

def preprocess_image(image_root):
	img = cv2.imread(image_root, 1)
	img = np.float32(cv2.resize(img, (224,224))) / 255
	# since we use Imagenet pretrained model
	means=[0.485, 0.456, 0.406]
	stds=[0.229, 0.224, 0.225]
	preprocessed_img = img.copy()[: , :, ::-1]
	for i in range(3):
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
	preprocessed_img = \
		np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
	preprocessed_img = torch.from_numpy(preprocessed_img)
	preprocessed_img.unsqueeze_(0)
	input = Variable(preprocessed_img, requires_grad = True)
	return input

def normalize_CAM(CAM):
	zero_tensor = torch.zeros(CAM.shape)
	CAM = torch.max(CAM, zero_tensor)
	CAM_min = CAM.clone()
	CAM_max = CAM.clone()
	for dim in (2,3):
		CAM_max = torch.max(input=CAM_max, dim=dim, keepdim=True)[0]
		CAM_min = torch.min(input=CAM_min, dim=dim, keepdim=True)[0]
	CAM = CAM - CAM_min
	CAM = CAM / CAM_max
	return CAM