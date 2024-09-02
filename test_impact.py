from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from cifar import get_classwise_DataLoader
import _pickle as pickle

import os
import sys
import copy
import argparse

#from model_factory import *

class impactModel(nn.Module):

	def __init__(self, net, layer=0):
		super(impactModel, self).__init__()
		self.layer_index = layer
		self.classifier = net.classifier

		if net.__class__.__name__.lower().startswith('vgg'):	
			self.head = nn.Sequential(*list(net._modules['features'].children())[:self.layer_index+1])
			wt_size = list(net._modules['features'].children())[self.layer_index].out_channels
			self.wt_imp = nn.Parameter(torch.ones(wt_size), requires_grad=True)
			self.tail = nn.Sequential(*list(net._modules['features'].children())[self.layer_index+1:])

		else:
			print('model not supported')
			sys.exit(0)

	def forward(self, x):
		head_out = self.head(x)
		imp_out = torch.transpose(torch.mul(torch.transpose(head_out, 1, 3), self.wt_imp), 1, 3)#.clone()
		tail_out = self.tail(imp_out)
		tail_out = tail_out.view(tail_out.size(0), -1)
		output = self.classifier(tail_out)
		return output


class Scores():

	def __init__(self, args, net, impact_data_path):

		self.net = net
		if args.dataset.lower() == 'cifar10':
			self.num_classes = 10
		elif args.dataset.lower() == 'cifar100':
			self.num_classes = 100
		else:
			print('not implemented')
			sys.exit(0)
		self.dataset = get_classwise_DataLoader(dataset=args.dataset)
		print(self.net.features)
		if self.net.features:
			layers = list(self.net.features.children())
			self.layer_channels = np.asarray([layers[layer].out_channels 
								if isinstance(layers[layer], nn.Conv2d) 
								else 0 for layer in range(len(layers))])
		else:
			print('model not implemented')
			sys.exit(0)
		
		self.scores = {}
		self.write_flag = args.write
		self.one_hot = {}
		for i in range(self.num_classes):
			one_hot = torch.zeros(self.num_classes)
			one_hot[i] = 1
			self.one_hot[i] = torch.eq(one_hot,1)
		self.impact_data_path = impact_data_path


	def get_scores_in_layer(self, layer=0, max_samples=1000):

		impact_net = impactModel(copy.deepcopy(self.net), layer)
		out_channels = self.layer_channels[layer]
		lyr_score = torch.zeros(self.num_classes, out_channels) 

		for class_id in range(self.num_classes):
			num_samples = min(self.dataset[class_id].__len__(), max_samples)
			if num_samples < 1:
				continue
			print('Class :', class_id, 'in layer ', layer, 'is starting')

			imp_scores = torch.zeros(int(out_channels), num_samples)
			dataloader = torch.utils.data.DataLoader(self.dataset[class_id], batch_size=1, 
														shuffle=True, num_workers=4)
			
			impact_net.eval()
			impact_net.cuda()
			for batch_idx, (input, label) in enumerate(dataloader):
				if batch_idx >= max_samples - 1:
					break
				input = input.cuda()
				impact_net.zero_grad()
				out = impact_net(input)
				pred = F.softmax(out, dim=1)
				pred = pred.masked_select(self.one_hot[class_id].cuda())
				pred.backward()
				imp_scores[:, batch_idx] = impact_net.wt_imp.grad.data

			assert batch_idx, max_samples
			print('Class: ', class_id, 'in_layer ', layer, 'is done')
			self.write_scores(imp_scores.cpu(), class_label=class_id, layer=layer)
			lyr_score[class_id,:] = torch.squeeze(torch.sum(imp_scores, 1))

		self.write_scores(lyr_score.cpu(),layer=layer)
		return lyr_score

	def get_scores_all_layer(self):
		print('Scores calculation starting')
		l_ind = np.where(self.layer_channels > 0)[0]
		for i in range(l_ind.size):
			print('Layer ', l_ind[i], 'is starting')
			self.scores[l_ind[i]] = self.get_scores_in_layer(layer=l_ind[i]).cpu()
			print('layer ', l_ind[i], 'is done')

		print('All layers are done')
		self.write_scores(self.scores)
		print('Scores obtained')

	def write_scores(self, imp_scores, layer=None, class_label=None):
		if self.write_flag:
			if class_label is not None and layer is not None:
				name = self.impact_data_path + '/class' + str(class_label) + 'in' +str(layer) + '.pkl'
			
			elif layer is not None:
				name = self.impact_data_path + '/layer'+ str(layer)  + '.pkl'

			elif class_label is not None:
				name = self.impact_data_path + '/class' + str(class_label) + '.pkl'
			else:
				name = self.impact_data_path + '/impact_vgg_imgnet.pkl'
			
			with open(name, 'wb') as outfile:
				pickle.dump(imp_scores, outfile) #, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-net', type=str, required=True, help='net type')
	parser.add_argument('-dataset', type=str, default='cifar100', help='dataset')
	parser.add_argument('-write', type=bool, default=True, help='write impact scores')
	parser.add_argument('-ckpt', type=str, required=True, help='baseline ckpt path')
	parser.add_argument('-beta', type=int)
	args = parser.parse_args()

	net = get_network(args)
	print(net.__class__.__name__)
	#ckpt_path = '../decompose_master/tucker/checkpoint/'+args.dataset+'/Undecomposed'+args.net+'.pth'
	#ckpt_path = 'checkpoint/dump/interpretable/cifar100/vgg16/vgg16-1305-5.pth' 
	ckpt_path = 'checkpoint/dump/interpretable/'+args.dataset+'/'+args.net+'/'+args.net+'-'+args.ckpt+'.pth'
	net.load_state_dict(torch.load(ckpt_path))
	
	impact_data_path = './impact_scores/semantic/' + args.dataset + '/' + args.net + '/' + str(args.beta)
	if not os.path.exists(impact_data_path):
		os.makedirs(impact_data_path)
	scores = Scores(args, net, impact_data_path)
	scores.get_scores_all_layer()
	scores_samples = scores.scores
			
	







