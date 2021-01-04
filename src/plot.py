import matplotlib.pyplot as plt

def disp_2D(tensor):
    plt.imshow(tensor[0])
    print(tensor[2], tensor[1])


import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import numpy as np
import copy
from mnist_loader import search_train_data, search_retrain_data, search_validation_data, train_data, test_data, batch_size
from utils_model import test_accuracy
from utils_misc import get_sparsity

import seaborn as sns
import matplotlib.pyplot as plt

def show_weights(model):
	"""
	shows histograms of 3 weight layers in a model - built for LeNet 300-100
	"""
	weight_list = [x for x in model.state_dict().keys() if 'weight' in x]
	plt.clf()
	plt.figure(figsize=(18, 3))
	for i,weight in enumerate(weight_list):
		plt.subplot(131 + i)
		fc_w = model.state_dict()[weight]
		sns.distplot(fc_w.view(-1).cpu().numpy())
		plt.title('Layer: {}'.format(weight))
	plt.show()
	
def print_dims(model):
	"""
	print dimensions of a model
	"""
	for i,params in enumerate(model.parameters()):
		param_list = []
		for pdim in params.size():
			param_list.append(str(pdim))
		if i%2==0:
			dim_str = "x".join(param_list)
		else:
			print (dim_str + " + " + "x".join(param_list))
			
def prune_plot(temp, dev_res, perc_res, test_acc_o, train_acc_o, weight_penalty_o, test_acc_kd, train_acc_kd, weight_penalty_kd):
	"""
	KD pruning plots for standard deviation and percentile based weight pruning
	-- deprecated after SWS pruning has been implemented
	"""
	c1 = '#2ca02c'
	c2 = '#1f77b4'
	c3 = '#ff7f0e'
	c4 = '#d62728'
	plt.clf()
	ncols = 5
	nrows = 1

	plt.figure(figsize=(25,4))
	plt.subplot(nrows, ncols, 1)
	plt.plot(perc_res['pruned'], perc_res['train ce'], color = c1, label = "0-Mean Pruning")
	plt.plot(dev_res['pruned'], perc_res['train ce'], color = c2, label = "Mean-Deviation Pruning")
	plt.axhline(y=train_acc_o[1], label="Original", color = c3, linestyle='--')
	plt.axhline(y=train_acc_kd[1], label="Distilled", color = c4, linestyle='--')
	plt.xlim([0, 100])
	plt.ylabel("Cross Entropy Loss")
	plt.xlabel("Parameters Pruned(%)")
	plt.legend(loc=2)
	plt.title("Train CE Loss")

	plt.subplot(nrows, ncols, 2)
	plt.plot(perc_res['pruned'], perc_res['test ce'], color = c1, label = "0-Mean Pruning")
	plt.plot(dev_res['pruned'], perc_res['test ce'], color = c2, label = "Mean-Deviation Pruning")
	plt.axhline(y=test_acc_o[1], label="Original", color = c3, linestyle='--')
	plt.axhline(y=test_acc_kd[1], label="Distilled", color = c4, linestyle='--')
	plt.xlim([0, 100])
	plt.ylabel("Cross Entropy Loss")
	plt.xlabel("Parameters Pruned(%)")
	plt.legend(loc=2)
	plt.title("Test CE Loss")

	plt.subplot(nrows, ncols, 3)
	plt.plot(perc_res['pruned'], perc_res['train acc'], color = c1, label = "0-Mean Pruning")
	plt.plot(dev_res['pruned'], perc_res['train acc'], color = c2, label = "Mean-Deviation Pruning")
	plt.axhline(y=train_acc_o[0], label="Original", color = c3, linestyle='--')
	plt.axhline(y=train_acc_kd[0], label="Distilled", color = c4, linestyle='--')
	plt.xlim([0, 100])
	plt.ylabel("Accuracy(%)")
	plt.xlabel("Parameters Pruned(%)")
	plt.legend(loc=6)
	plt.title("Train Accuracy")

	plt.subplot(nrows, ncols, 4)
	plt.plot(perc_res['pruned'], perc_res['test acc'], color = c1, label = "0-Mean Pruning")
	plt.plot(dev_res['pruned'], perc_res['test acc'], color = c2, label = "Mean-Deviation Pruning")
	plt.axhline(y=test_acc_o[0], label="Original", color = c3, linestyle='--')
	plt.axhline(y=test_acc_kd[0], label="Distilled", color = c4, linestyle='--')
	plt.xlim([0, 100])
	plt.ylabel("Accuracy(%)")
	plt.xlabel("Parameters Pruned(%)")
	plt.legend(loc=6)
	plt.title("Test Accuracy")

	plt.subplot(nrows, ncols, 5)
	plt.plot(perc_res['pruned'], perc_res['L2'], color = c1, label = "0-Mean Pruning")
	plt.plot(dev_res['pruned'], perc_res['L2'], color = c2, label = "Mean-Deviation Pruning")
	plt.axhline(y=weight_penalty_o, label="Original", color = c3, linestyle='--')
	plt.axhline(y=weight_penalty_kd, label="Distilled", color = c4, linestyle='--')
	plt.xlim([0, 100])
	plt.ylabel("L2")
	plt.xlabel("Parameters Pruned(%)")
	plt.legend(loc=6)
	plt.title("Model L2")
	plt.show()
	
	
###
def show_sws_weights(model, means=0, precisions=0, epoch=-1, accuracy=-1, savefile = ""):
	"""
	show model weight histogram with mean and precisions
	"""
	weights = np.array([], dtype=np.float32)
	for layer in model.state_dict():
		weights = np.hstack( (weights, model.state_dict()[layer].view(-1).cpu().numpy()) )
		
	plt.clf()
	plt.figure(figsize=(20, 6))
	
	#1 - Non-log plot
	plt.subplot(2,1,1)
	
	#Title
	if (epoch !=-1 and accuracy == -1):
		plt.title("Epoch: {:0=3d}".format(epoch+1))
	if (accuracy != -1 and epoch == -1):
		plt.title("Accuracy: {:.2f}".format(accuracy))
	if (accuracy != -1 and epoch != -1):
		plt.title("Epoch: {:0=3d} - Accuracy: {:.2f}".format(epoch+1, accuracy))
	
	sns.distplot(weights, kde=False, color="g",bins=200,norm_hist=True, hist_kws={'log':False})
	
	#plot mean and precision
	if not (means==0 or precisions==0):
		plt.axvline(0, linewidth = 1)
		std_dev0 = np.sqrt(1/np.exp(precisions[0]))
		plt.axvspan(xmin=-std_dev0, xmax=std_dev0, alpha=0.3)

		for mean, precision in zip(means, precisions[1:]):
			plt.axvline(mean, linewidth = 1)
			std_dev = np.sqrt(1/np.exp(precision))
			plt.axvspan(xmin=mean - std_dev, xmax=mean + std_dev, alpha=0.1)
	
	#plt.xticks([])
	#plt.xlabel("Weight Value")
	plt.ylabel("Density")
	
	plt.xlim([-1, 1])
	plt.ylim([0, 60])
	
	#2-Logplot
	plt.subplot(2,1,2)
	sns.distplot(weights, kde=False, color="g",bins=200,norm_hist=True, hist_kws={'log':True})
	#plot mean and precision
	if not (means==0 or precisions==0):
		plt.axvline(0, linewidth = 1)
		std_dev0 = np.sqrt(1/np.exp(precisions[0]))
		plt.axvspan(xmin=-std_dev0, xmax=std_dev0, alpha=0.3)

		for mean, precision in zip(means, precisions[1:]):
			plt.axvline(mean, linewidth = 1)
			std_dev = np.sqrt(1/np.exp(precision))
			plt.axvspan(xmin=mean - std_dev, xmax=mean + std_dev, alpha=0.1)
	plt.xlabel("Weight Value")
	plt.ylabel("Density")
	plt.xlim([-1, 1])
	plt.ylim([1e-3, 1e2])
	
	if savefile!="":
		plt.savefig("./figs/{}_{}.png".format(savefile, epoch+1), bbox_inches='tight')
		plt.close()
	else:
		plt.show()

def show_sws_weights_log(model, means=0, precisions=0, epoch=-1, accuracy=-1, savefile = ""):
	"""
	show model weight histogram with mean and precisions
	"""
	weights = np.array([], dtype=np.float32)
	for layer in model.state_dict():
		weights = np.hstack( (weights, model.state_dict()[layer].view(-1).cpu().numpy()) )
		
	plt.clf()
	plt.figure(figsize=(20, 3))

	#2-Logplot
	sns.distplot(weights, kde=False, color="g",bins=200,norm_hist=True, hist_kws={'log':True})
	#plot mean and precision
	if not (means==0 or precisions==0):
		plt.axvline(0, linewidth = 1)
		std_dev0 = np.sqrt(1/np.exp(precisions[0]))
		plt.axvspan(xmin=-std_dev0, xmax=std_dev0, alpha=0.3)

		for mean, precision in zip(means, precisions[1:]):
			plt.axvline(mean, linewidth = 1)
			std_dev = np.sqrt(1/np.exp(precision))
			plt.axvspan(xmin=mean - std_dev, xmax=mean + std_dev, alpha=0.1)
	plt.xlabel("Weight Value")
	plt.ylabel("Density")
	plt.xlim([-1.5, 1.5])
	plt.ylim([1e-3, 1e2])
	
	if savefile!="":
		plt.savefig("./figs/{}_{}.png".format(savefile, epoch+1), bbox_inches='tight')
		plt.close()
	else:
		plt.show()
		
		
###
def draw_sws_graphs(means = -1, stddev = -1, mixprop = -1, acc = -1, savefile=""):
	"""
	plot showing evolution of sws retraining
	"""
	plt.clf()
	plt.figure(figsize=(20, 10))
	plt.subplot(2,2,1)
	plt.plot(means)
	plt.title("Mean")
	plt.xlim([0, means.shape[0]-1])
	plt.xlabel("Epoch")

	plt.subplot(2,2,2)
	plt.plot(mixprop[:,1:])
	plt.yscale("log")
	plt.title("Mixing Proportions")
	plt.xlim([0, mixprop.shape[0]-1])
	plt.xlabel("Epoch")

	plt.subplot(2,2,3)
	plt.plot(stddev[:,1:])
	plt.yscale("log")
	plt.title("Standard Deviations")
	plt.xlim([0, stddev.shape[0]-1])
	plt.xlabel("Epoch")

	plt.subplot(2,2,4)
	plt.plot(acc)
	plt.title("Accuracy")
	plt.xlim([0, acc.shape[0]-1])
	plt.xlabel("Epoch")
	plt.show()
	
	if savefile!="":
		plt.savefig("./exp/{}.png".format(savefile), bbox_inches='tight')
		plt.close()
	else:
		plt.show()
		
		
def joint_plot(model, model_orig, gmp, epoch, retraining_epochs, acc, savefile = ""):
	"""
	joint distribution plot weights before and after sws retraining
	"""
	weights_T = np.array([], dtype=np.float32)
	for layer in model.state_dict():
		weights_T = np.hstack( (weights_T, model.state_dict()[layer].view(-1).cpu().numpy()) )

	weights_0 = np.array([], dtype=np.float32)
	for layer in model_orig.state_dict():
		weights_0 = np.hstack( (weights_0, model_orig.state_dict()[layer].view(-1).cpu().numpy()) )

	#get mean, stddev
	mu_T = np.concatenate([np.zeros(1), gmp.means.clone().data.cpu().numpy()])
	std_T = np.sqrt(1/np.exp(gmp.gammas.clone().data.cpu().numpy()))

	x0 = -1.2
	x1 = 1.2
	I = np.random.permutation(len(weights_0))
	f = sns.jointplot(weights_0[I], weights_T[I], size=8, kind="scatter", color="b", stat_func=None, edgecolor='w',
					  marker='o', joint_kws={"s": 8}, marginal_kws=dict(bins=1000), ratio=4)
	f.ax_joint.hlines(mu_T, x0, x1, lw=0.5)

	for k in range(len(mu_T)):
		if k == 0:
			f.ax_joint.fill_between(np.linspace(x0, x1, 10), mu_T[k] - 2 * std_T[k], mu_T[k] + 2 * std_T[k],
									color='g', alpha=0.1)
		else:
			f.ax_joint.fill_between(np.linspace(x0, x1, 10), mu_T[k] - 2 * std_T[k], mu_T[k] + 2 * std_T[k],
									color='b', alpha=0.1)
	
	plt.title("Epoch: %d /%d\nTest accuracy: %.4f " % (epoch+1, retraining_epochs, acc))
	f.ax_marg_y.set_xscale("log")
	f.set_axis_labels("Pretrained", "Retrained")
	f.ax_marg_x.set_xlim(-1, 1)
	f.ax_marg_y.set_ylim(-1, 1)
	if savefile!="":
		plt.savefig("./figs/jp_{}_{}.png".format(savefile, epoch+1), bbox_inches='tight')
		plt.close()
	else:
		plt.show()


class plot_data():
	def __init__(self, init_model, gmp="", mode="retrain", full_model = "", data_size = 'search', loss_type='CE', mv = (0,0), zmv = (0,0), tau = 1, temp = 0, mixtures = 1, dset="mnist"):
		self.layers =  [x.replace(".weight", "") for x in init_model.state_dict().keys() if "weight" in x]
		self.layer_init_weights = {}
		for l in self.layers:
			self.layer_init_weights[l] = np.concatenate([ init_model.state_dict()[l + ".weight"].clone().view(-1).cpu().numpy() , init_model.state_dict()[l + ".bias"].clone().view(-1).cpu().numpy() ])
		self.layer_weights = {}
			
		self.mode = mode
		self.loss_type = loss_type
			
		#accuracy and lost tracking flags
		self.data_size = data_size
		
		#accuracy and loss history
		self.epochs = []
		self.train_accuracy = []
		self.test_accuracy = []
		self.val_accuracy = []
		self.train_loss = []
		self.test_loss = []
		self.val_loss = []
		self.complexity_loss = []
		
		if (mode == 'layer_retrain'):
			self.full_model = full_model
			
		self.prune_layer_weight = {}
		self.prune_acc = {}
		self.sparsity=0
		
		self.mean = mv[0]
		self.var = mv[1]
		self.zmean = zmv[0]
		self.zvar = zmv[1]
		self.tau = tau
		self.temp = temp
		self.mixtures = mixtures
		self.use_prune = False
		
		#gmp tracking
		self.use_gmp =  ((mode == 'retrain' or mode == 'layer_retrain' ) and gmp != "")
		if (self.use_gmp):
			self.gmp_stddev = np.sqrt(1. / gmp.gammas.exp().data.clone().cpu().numpy())
			self.gmp_means = gmp.means.data.clone().cpu().numpy()
			self.gmp_mixprop = gmp.rhos.exp().data.clone().cpu().numpy()
			self.gmp_scale = gmp.scale.exp().data.clone().cpu().numpy()
			
		self.test_data_full = Variable(test_data(fetch='data', dset=dset)).cuda()
		self.test_labels_full = Variable(test_data(fetch='labels', dset=dset)).cuda()
			
		if (data_size =='search'):
			self.val_data_full = Variable(train_data(fetch='data', dset=dset)[50000:60000]).cuda()
			self.val_labels_full = Variable(train_data(fetch='labels', dset=dset)[50000:60000]).cuda()
			self.train_data_full = Variable(train_data(fetch='data', dset=dset)[40000:50000]).cuda()
			self.train_labels_full = Variable(train_data(fetch='labels', dset=dset)[40000:50000]).cuda()
			
		else:
			self.train_data_full = Variable(train_data(fetch='data', dset=dset)).cuda()
			self.train_labels_full = Variable(train_data(fetch='labels', dset=dset)).cuda()
		
	def data_epoch(self, epoch, model_in, gmp=""):
		self.epochs.append(epoch)
		
		#Updated Model Weights
		for l in self.layers:
			self.layer_weights[l] = np.concatenate([ model_in.state_dict()[l + ".weight"].clone().view(-1).cpu().numpy() , model_in.state_dict()[l + ".bias"].clone().view(-1).cpu().numpy() ])
		
		if (self.mode == "layer_retrain"):
			model = copy.deepcopy(self.full_model)
			for layer in model_in.state_dict():
				model.state_dict()[layer] = model_in.state_dict()[layer]
		else:
			model = model_in
		

		test_acc = test_accuracy(self.test_data_full, self.test_labels_full, model, loss_type = self.loss_type)
		self.test_accuracy.append(test_acc[0])
		self.test_loss.append(test_acc[1])
		train_acc = test_accuracy(self.train_data_full, self.train_labels_full, model, loss_type = self.loss_type)
		self.train_accuracy.append(train_acc[0])
		self.train_loss.append(train_acc[1])
		if (self.data_size == 'search'):
			val_acc = test_accuracy(self.val_data_full, self.val_labels_full, model, loss_type = self.loss_type)
			self.val_accuracy.append(val_acc[0])
			self.val_loss.append(val_acc[1])
			
		if (self.use_gmp):
			self.complexity_loss.append(float(gmp.call()[0]))
			self.gmp_stddev = np.vstack((self.gmp_stddev,  np.sqrt(1. / gmp.gammas.exp().data.clone().cpu().numpy()) ))
			self.gmp_means = np.vstack((self.gmp_means, gmp.means.data.clone().cpu().numpy() ))
			self.gmp_mixprop = np.vstack((self.gmp_mixprop, gmp.rhos.exp().data.clone().cpu().numpy() ))
			self.gmp_scale = np.vstack((self.gmp_scale, gmp.scale.exp().data.clone().cpu().numpy() ))
			
	def get_weights(self, source='in'):
		if 'in':
			return np.concatenate([self.layer_weights[x] for x in self.layer_weights])
		else:
			np.concatenate([self.layer_weights[x] for x in self.layer_weights])
			
	def data_prune(self, model_in):
		if (self.mode == "layer_retrain"):
			model = copy.deepcopy(self.full_model)
			for layer in model_in.state_dict():
				model.state_dict()[layer] = model_in.state_dict()[layer]
		else:
			model = model_in
			
		for l in self.layers:
			self.prune_layer_weight[l] = np.concatenate([ model.state_dict()[l + ".weight"].clone().view(-1).cpu().numpy() , model.state_dict()[l + ".bias"].clone().view(-1).cpu().numpy() ])
			
		test_acc = test_accuracy(self.test_data_full, self.test_labels_full, model, loss_type = self.loss_type)
		self.prune_acc['test'] = test_acc[0]
		train_acc = test_accuracy(self.train_data_full, self.train_labels_full, model, loss_type = self.loss_type)
		self.prune_acc['train'] = train_acc[0]
		self.train_loss.append(train_acc[1])
		if (self.data_size == 'search'):
			val_acc = test_accuracy(self.val_data_full, self.val_labels_full, model, loss_type = self.loss_type)
			self.prune_acc['val'] = val_acc[0]

		self.sparsity = get_sparsity(model_in)
		self.use_prune = True
		
	def gen_dict(self):
		res = {}
		res['init_weights'] = self.layer_init_weights
		res['final_weights'] = self.layer_weights
		res['data_size'] = self.data_size
		res['epochs'] = self.epochs
		res['train_acc'] = self.train_accuracy
		res['test_acc'] = self.test_accuracy
		res['val_acc'] = self.val_accuracy
		res['train_loss'] = self.train_loss
		res['test_loss'] = self.test_loss
		res['val_loss'] = self.val_loss
		res['complexity_loss'] = self.complexity_loss

		
		#gmp tracking
		if (self.use_gmp):
			res['gmp_stddev'] = self.gmp_stddev
			res['gmp_means'] = self.gmp_means
			res['gmp_mixprop'] = self.gmp_mixprop
			res['scale'] = self.gmp_scale
			
			res['mean'] = self.mean
			res['var'] = self.var
			res['zmean'] = self.zmean
			res['zvar'] = self.zvar
			res['tau'] = self.tau
			res['temp'] = self.temp
			res['mixtures'] = self.mixtures
			
		if(self.use_prune):
			res['prune_acc'] = self.prune_acc
			res['prune_weights'] = self.prune_layer_weight
			res['sparsity'] = self.sparsity
		return res