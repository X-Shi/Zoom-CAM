import numpy as np
import torch
from torch.autograd import Variable

class FeatureExtractor():
	def __init__(self, model, intermediate_layers):
		self.model = model
		self.intermediate_layers = intermediate_layers[::-1]
		self.weights = []
		self.num = len(self.intermediate_layers)
		self.activations = []
	def save_weight(self, grad):
		self.weights.append(grad)
	def __call__(self, x):
		for name, module in self.model._modules.items():
			x = module(x)
			for i in range(self.num):
				if name == self.intermediate_layers[i]:
					self.activations.append(x)
					x.register_hook(self.save_weight)
					break
		return self.activations, x

class ZoomCAM_gradients():
	def __init__(self, model, intermediate_layers):
		self.model = model
		self.intermediate_layers = intermediate_layers
		self.extractor = FeatureExtractor(self.model.features, intermediate_layers)
	def __call__(self, input, device, index):
		target_layer_activations, last_activations = self.extractor(input)
		final_output = self.model.classifier(last_activations.view(last_activations.size(0), -1))
		# final_output size: torch.Size([batch_size, num_classes])
		if index == None:
			index = np.argmax(final_output.cpu().data.numpy())
		one_hot = torch.zeros(final_output.size())
		one_hot[0,index] = 1
		one_hot = Variable(one_hot, requires_grad = True)
		one_hot = torch.sum(one_hot.to(device) * final_output)
		self.model.features.zero_grad()
		self.model.classifier.zero_grad()
		one_hot.backward(retain_graph = True)
		grads_val = self.extractor.weights
		return target_layer_activations, grads_val
