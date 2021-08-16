import torch
import cv2
import os
import numpy as np
import torch.nn as nn
import function
import object_class
from torchvision import models
#%%
if __name__ == 'main':
	# preparation of models
	args = function.get_args()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if args.model_path is not None:
		model = torch.load(args.model_path)
	else:
		model = models.__dict__[args.arch](pretrained = True)
	model = model.to(device)
	model.eval()

	# preprocess the image
	img = function.preprocess_image(args.input)
	size_1 = img.size(2)
	size_2 = img.size(3)
	image = cv2.imread(args.input, 1)
	image = np.float32(cv2.resize(image, (224,224))) / 255


	# get intermediate_layers
	intermediate_layers = args.intermediate_layers.split(',')
	intermediate_layers = sorted([int(i) for i in intermediate_layers])
	intermediate_layers = [str(i) for i in intermediate_layers]

	# target, if none, returns the predicted category
	if args.target_index == -1:
		target = None
	else:
		target = args.target_index
	#%%

	zoom_cam_model = object_class.ZoomCAM_gradients(model, intermediate_layers)
	activations, weights = zoom_cam_model(img, device, target)

	zoom_cams = []
	for i in range(len(intermediate_layers)):
		zoom_cams.append(activations[len(intermediate_layers)-i].cpu() * weights[i].cpu())
		zoom_cams[i] = torch.sum(zoom_cams[i], dim = (0,1), keepdim = True)

	for i in range(len(intermediate_layers)):
		zoom_cams[i] = function.normalize_CAM(zoom_cams[i])
		scale = size_1 / zoom_cams[i].size(2)
		upsample = nn.Upsample(scale_factor = scale, mode = 'bilinear')
		zoom_cam = upsample(zoom_cams[i])
		heatmap = cv2.applyColorMap(np.uint8(255*zoom_cam[0,0,:,:].detach().numpy()), cv2.COLORMAP_JET)
		heatmap = np.float32(heatmap) / 255
		cam = heatmap + np.float32(image)
		cam = cam / np.max(cam)
		cv2.imwrite(os.path.join(args.output, intermediate_layers[i]+'.png'), cam)

	for i in range(len(intermediate_layers)):
		if i == 0:
			aggregated_zoom_cam = zoom_cams[i]
		else:
			scale = zoom_cams[i].size(2) / zoom_cams[i-1].size(2)
			upsample = nn.Upsample(scale_factor = scale, mode = 'bilinear')
			aggregated_zoom_cam = torch.max(upsample(aggregated_feature_maps), zoom_cams[i])
	heatmap = cv2.applyColorMap(np.uint8(255*aggregated_zoom_cam[0,0,:,:].detach().numpy()), cv2.COLORMAP_JET)
	heatmap = np.float32(heatmap) / 255
	cam = cam / np.max(cam)
	cv2.imwrite(os.path.join(args.output, 'zoom_cam.png'), cam)