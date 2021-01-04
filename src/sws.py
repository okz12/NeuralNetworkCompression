import torch
from torch.autograd import Variable
import numpy as np
import copy
from torch.nn.modules import Module

def logsumexp(t, w=1, axis=1):
	#print (t.shape)
	t_max, _ = t.max(dim=1)
	if (axis==1):
		t = t-t_max.repeat(t.size(1), 1).t()
	else:
		t = t-t_max.repeat(1, t.size(0)).t()
	t = w * t.exp()
	t = t.sum(dim=axis)
	t.log_()
	return t + t_max

def nplogsumexp(ns):
	max_val = np.max(ns)
	ds = ns - max_val
	sumOfExp = np.exp(ds).sum()
	return max_val + np.log(sumOfExp)


class GaussianMixturePrior(Module):
	def __init__(self, nb_components, network_weights, pi_zero, init_var=0.25, zero_ab=(5e3, 2), ab=(2.5e4, 1),
				 means=[], scaling=False):
		super(GaussianMixturePrior, self).__init__()

		self.nb_components = nb_components
		self.network_weights = [p.view(-1) for p in network_weights]
		self.pi_zero = pi_zero

		self.scaling = scaling

		# Build
		J = self.nb_components
		pi_zero = self.pi_zero

		#	... means
		if means == []:
			init_means = torch.linspace(-1., 1., J - 1)
		else:
			init_means = torch.Tensor(means)
		self.means = Variable(init_means.cuda(), requires_grad=True)

		# precision
		init_stds = torch.FloatTensor(np.tile(init_var, J))
		self.gammas = Variable((- torch.log(torch.pow(init_stds, 2))).cuda(), requires_grad=True)

		# mixing proportions
		init_mixing_proportions = torch.ones((J - 1))
		init_mixing_proportions *= (1. - pi_zero) / (J - 1)
		self.rhos = Variable((init_mixing_proportions).cuda(), requires_grad=True)
		self.print_batch = True

		self.zero_ab = zero_ab
		self.ab = ab
		print("0-component Mean: {} Variance: {}".format(zero_ab[0] / zero_ab[1], zero_ab[0] / (zero_ab[1] ** 2)))
		print("Non-zero component Mean: {} Variance: {}".format(ab[0] / ab[1], ab[0] / (ab[1] ** 2)))
		# self.loss = Variable(torch.cuda.FloatTensor([0.]), requires_grad=True)

		# scaling
		scale = torch.ones(int(len(network_weights) / 2))

		for i, x in enumerate(network_weights):
			if (len(x.size())) != 1:
				weight = x.clone().data
			else:
				bias = x.clone().data
				w_std = torch.cat((weight.view(-1), bias)).std()
				scale[int(i / 2)] = w_std

		if (scaling):
			self.scale = Variable((scale[1:] / scale[0]).log().cuda(), requires_grad=True)
		else:
			self.scale = Variable(torch.Tensor([0]).cuda())

	def call(self, mask=None):
		J = self.nb_components
		loss = Variable(torch.cuda.FloatTensor([0.]), requires_grad=True)
		means = torch.cat((Variable(torch.cuda.FloatTensor([0.]), requires_grad=True), self.means), 0)
		scale = torch.cat((Variable(torch.cuda.FloatTensor([0.]), requires_grad=True), self.scale), 0)
		# mean=self.means
		precision = self.gammas.exp()

		min_rho = self.rhos.min().repeat(self.rhos.size())
		mixing_proportions = (self.rhos - min_rho).exp()
		mixing_proportions = (1 - self.pi_zero) * mixing_proportions / mixing_proportions.sum().repeat(
			mixing_proportions.size())
		mixing_proportions = torch.pow(mixing_proportions, 2)
		mixing_proportions = torch.cat((Variable(torch.cuda.FloatTensor([self.pi_zero])), mixing_proportions), 0)

		for i, weights in enumerate(self.network_weights):
			if (self.scaling):
				weight_loss = self.compute_loss(weights / scale[int(i / 2)].exp(), mixing_proportions, means, precision)
			else:
				weight_loss = self.compute_loss(weights, mixing_proportions, means, precision)
			if (self.print_batch):
				print("Layer Loss: {:.3f}".format(float(weight_loss.data)))
			loss = loss + weight_loss

		# GAMMA PRIOR ON PRECISION
		# ... for the zero component
		# Replacing gather with indexing -- same calculation?
		(alpha, beta) = self.zero_ab
		# print (torch.gather(self.gammas, 0, Variable(torch.cuda.LongTensor([0,1,2]))))
		neglogprop = (1 - alpha) * self.gammas[0] + beta * precision[0]
		if (self.print_batch):
			print("0-neglogprop Loss: {:.3f}".format(float(neglogprop.data)))
		loss = loss + neglogprop.sum()
		# ... and all other component
		alpha, beta = self.ab
		neglogprop = (1 - alpha) * self.gammas[1:J] + beta * precision[1:J]
		if (self.print_batch):
			print("Remaining-neglogprop Loss: {:.3f}".format(float(neglogprop.sum().data)))
		loss = loss + neglogprop.sum()
		self.print_batch = False
		return loss

	def compute_loss(self, weights, mixing_proportions, means, precision):
		diff = weights.expand(means.size(0), -1) - means.expand(weights.size(0), -1).t()
		unnormalized_log_likelihood = (-(diff ** 2) / 2).t() * precision
		# unnormalized_log_likelihood = (-1/2) * precision.matmul((diff ** 2))
		Z = precision.sqrt() / (2 * np.pi)
		# global myt
		# myt=unnormalized_log_likelihood
		log_likelihood = logsumexp(unnormalized_log_likelihood, w=(mixing_proportions * Z), axis=1)
		return -log_likelihood.sum()

def special_flatten(state_dict):
	return torch.cat (([state_dict[x].view(-1) for x in state_dict]), 0)

def KL(means, logprecisions):
	"""Compute the KL-divergence between 2 Gaussian Components."""
	precisions = np.exp(logprecisions)
	return 0.5 * (logprecisions[0] - logprecisions[1]) + precisions[1] / 2. * (
	1. / precisions[0] + (means[0] - means[1]) ** 2) - 0.5


def compute_responsibilies(xs, mus, logprecisions, pis):
	"Computing the unnormalized responsibilities."
	xs = xs.flatten()
	K = len(pis)
	W = len(xs)
	responsibilies = np.zeros((K, len(xs)))
	for k in range(K):
		# Not normalized!!!
		responsibilies[k] = pis[k] * np.exp(0.5 * logprecisions[k]) * np.exp(
			- np.exp(logprecisions[k]) / 2 * (xs - mus[k]) ** 2)
	return np.argmax(responsibilies, axis=0)

def merger(inputs):
	"""Comparing and merging components."""
	for _ in range(3):
		lists = []
		for inpud in inputs:
			for i in inpud:
				tmp = 1
				for l in lists:
					if i in l:
						for j in inpud:
							l.append(j)
						tmp = 0
				if tmp is 1:
					lists.append(list(inpud))
		lists = [np.unique(l) for l in lists]
		inputs = lists
	return lists

def sws_prune_l2(model_dict, gmp):
	if (gmp.scaling):
		scale = [1] + list(gmp.scale.exp().data.clone().cpu().numpy())
		layer_mult = [float(scale[int(i/2)]) for i in range(len(model_dict))]
		#layer_mult = [1, 1, 1.1566674, 1.1566674,1.4920919, 1.4920919]
		weights = np.concatenate([model_dict[array].clone().cpu().numpy().flatten() / layer_mult[i] for i, array in enumerate(model_dict)])
		mult_list = layer_mult
	else:
		weights = np.concatenate([model_dict[array].clone().cpu().numpy().flatten() for i, array in enumerate(model_dict)])
	weights = weights.reshape((len(weights), 1))
	means = np.concatenate([np.zeros(1), gmp.means.clone().data.cpu().numpy()])
	
	sorted_means = np.sort(means)
	bins = (sorted_means[1:] + sorted_means[:-1])/2
	
	for i, b in enumerate(bins):
		if (i==0):
			weights[np.where(weights < b)] = sorted_means[i]
		elif (i== len(bins)-1):
			weights[np.where(weights > b)] = sorted_means[i]
			weights[np.where(np.logical_and(weights < b, weights > prev_b))] = sorted_means[i]
		else:
			weights[np.where(np.logical_and(weights < b, weights > prev_b))] = sorted_means[i]
		prev_b = b
# 	weights[np.where(np.abs(weights) < 0.1)] = 0
	out = weights
	pruned_state_dict = copy.deepcopy(model_dict)
	dim_start = 0

	
	for i, layer in enumerate(model_dict):
		layer_mult = 1
		if (gmp.scaling):
			layer_mult = mult_list[int(i)]
		elems = model_dict[layer].numel()
		pruned_state_dict[layer] = torch.from_numpy(np.array(out[dim_start:dim_start + elems]).reshape(model_dict[layer].shape)) * layer_mult
		dim_start += elems
	return pruned_state_dict

def sws_prune_0(model, gmp):
	if (gmp.scaling):
		layer_mult = [float(gmp.scale[int(i/2)].exp()) for i in range(len(model.state_dict()))]
		weights = np.concatenate([model.state_dict()[array].clone().cpu().numpy().flatten() / layer_mult[i] for i, array in enumerate(model.state_dict())])
	else:
		weights = np.concatenate([model.state_dict()[array].clone().cpu().numpy().flatten() for i, array in enumerate(model.state_dict())])
	weights = weights.reshape((len(weights), 1))
	means = np.concatenate([np.zeros(1), gmp.means.clone().data.cpu().numpy()])
	
	sorted_means = np.sort(means)
	bins = (sorted_means[1:] + sorted_means[:-1])/2
	for i, b in enumerate(bins):
		'''
		if (i==0):
			weights[np.where(weights < b)] = sorted_means[i]
		elif (i== len(bins)-1):
			weights[np.where(weights > b)] = sorted_means[i]
			weights[np.where(np.logical_and(weights < b, weights > prev_b))] = sorted_means[i]
		else:
		'''
		if (sorted_means[i] == 0):
			weights[np.where(np.logical_and(weights < b, weights > prev_b))] = sorted_means[i]
		prev_b = b
	#weights[np.where(np.abs(weights) < 0.025)] = 0
	out = weights
	pruned_state_dict = copy.deepcopy(model.state_dict())
	dim_start = 0
	for i, layer in enumerate(model.state_dict()):
		layer_mult = 1
		if (gmp.scaling):
			layer_mult = float(gmp.scale[int(i/2)].exp())
		elems = model.state_dict()[layer].numel()
		pruned_state_dict[layer] = torch.from_numpy(np.array(out[dim_start:dim_start + elems]).reshape(model.state_dict()[layer].shape)) * layer_mult
		dim_start += elems
	return pruned_state_dict


def sws_prune(model_dict, gmp):
	"""
	model: Model retrained with Gaussian Mixture Prior
	gmp: Gaussian mixture prior object
	returns: Pruned model state_dict
	"""
	if (gmp.scaling):
		layer_mult = [float(gmp.scale[int(i/2)].exp()) for i in range(len(model_dict))]
		weights = np.concatenate([model_dict[array].clone().cpu().numpy().flatten() * layer_mult[i] for i, array in enumerate(model_dict)])
	else:
		weights = np.concatenate([model_dict[array].clone().cpu().numpy().flatten() for i, array in enumerate(model_dict)])
	weights = weights.reshape((len(weights), 1))


	pi_zero = gmp.pi_zero
	#weights = special_flatten(model.state_dict()).clone().cpu().numpy()
	means = np.concatenate([np.zeros(1), gmp.means.clone().data.cpu().numpy()])
	logprecisions = gmp.gammas.clone().data.cpu().numpy()
	logpis = np.concatenate([np.log(pi_zero) * np.ones(1), gmp.rhos.clone().data.cpu().numpy()])

	'''
	# classes K
	J = len(logprecisions)
	# compute KL-divergence
	K = np.zeros((J, J))
	L = np.zeros((J, J))

	for i, (m1, pr1, pi1) in enumerate(zip(means, logprecisions, logpis)):
		for j, (m2, pr2, pi2) in enumerate(zip(means, logprecisions, logpis)):
			K[i, j] = KL([m1, m2], [pr1, pr2])
			L[i, j] = np.exp(pi1) * (pi1 - pi2 + K[i, j])

	# merge -- KL divergence not low enough
	
	idx, idy = np.where(K <1e-10)
	lists = merger(zip(idx, idy))

	# compute merged components
	# print lists
	new_means, new_logprecisions, new_logpis = [], [], []

	for l in lists:
		new_logpis.append(nplogsumexp(logpis[l]))
		new_means.append(
			np.sum(means[l] * np.exp(logpis[l] - np.min(logpis[l]))) / np.sum(np.exp(logpis[l] - np.min(logpis[l]))))
		new_logprecisions.append(np.log(
			np.sum(np.exp(logprecisions[l]) * np.exp(logpis[l] - np.min(logpis[l]))) / np.sum(
				np.exp(logpis[l] - np.min(logpis[l])))))

	new_means[np.argmin(np.abs(new_means))] = 0.0
	'''
	# compute responsibilities
	#argmax_responsibilities = compute_responsibilies(weights, new_means, new_logprecisions, np.exp(new_logpis))
	argmax_responsibilities = compute_responsibilies(weights, means, logprecisions, np.exp(logpis))
	out1 = [means[i] for i in argmax_responsibilities]
# 	print(means)
# 	print (argmax_responsibilities)
	out = [out1[i] if out1[i] == 0 else float(weights[i]) for i in range(len(out1))]
	#print (out)
	
	pruned_state_dict = copy.deepcopy(model_dict)
	dim_start = 0
	for i, layer in enumerate(model_dict):
		layer_mult = 1
		if (gmp.scaling):
			layer_mult = float(gmp.scale[int(i/2)].exp())
		elems = model_dict[layer].numel()
		pruned_state_dict[layer] = torch.from_numpy(np.array(out[dim_start:dim_start + elems]).reshape(model_dict[layer].shape)) / layer_mult
		dim_start += elems
	return pruned_state_dict

def sws_prune_copy(model, gmp, mode='l2'):
	new_model = copy.deepcopy(model)
	if(mode==''):
		state_dict = sws_prune(model.state_dict(), gmp)
		state_dict = sws_prune_l2(state_dict, gmp)
	if(mode=='l2'):
		state_dict = sws_prune_l2(model.state_dict(), gmp)
	if(mode=='p'):
		state_dict = sws_prune(model.state_dict(), gmp)
	new_model.load_state_dict(state_dict)
	return new_model

class compressed_model():
	def __init__(self, state_dict, gmp_list):
		gmp_means = []
		self.scale_size = 0
		for g in gmp_list:
			gmp_means.append(list(g.means.clone().data.cpu().numpy()))
			
		if (g.scaling):#if scaling , only one gmp should be present
			mult_scale = [1] + list(g.scale.data.exp().clone().cpu().numpy())
			weights = torch.cat([state_dict[layer].view(-1) * float(mult_scale[int(i/2)]) for i,layer in enumerate(state_dict)]).cpu().numpy()
			self.scale_size = float(g.scale.size()[0])
		else:
			weights = torch.cat([state_dict[layer].view(-1) for layer in state_dict]).cpu().numpy()
			
		means = np.sort( np.append( np.array(gmp_means), np.zeros(1)) )
		bins = means + 0.001 * abs(means)
		binned_weights = np.digitize(weights, bins, right=True)

		unique, counts = np.unique(binned_weights, return_counts=True)
		zero_idx = np.argmax(counts)
		binned_weights[ binned_weights==unique[zero_idx] ] = 0
		new_means = []
		new_means.append(0.0)

		means = np.append(means, means[-1])
		set_idx = 1
		for idx in unique:
			if idx != unique[zero_idx]:
				new_means.append(means[idx])
				binned_weights[ binned_weights==idx] = set_idx
				set_idx += 1
				
		self.binned_weights = binned_weights
		self.means = means

	def encode(self, index_bits):
		index_spacing = index_bits**2
		index_list = []
		weight_list = []
		index_counter=0
		for pos, weight in enumerate(list(self.binned_weights)):
			index_counter+=1
			if (weight==0):
				if index_counter==index_spacing:
					index_list.append(index_spacing)
					weight_list.append(0)
					index_counter=0
			else:
				weight_list.append(weight)
				index_list.append(index_counter)
				index_counter=0
		if(index_counter>0):
			index_list.append(index_counter)
			weight_list.append(0)
		return index_list, weight_list
	
	def decode(self, index_list, weight_list):
		R_recov = []
		for index, weight in zip(index_list, weight_list):
			for z in range(int(index-1)):
				R_recov.append(0)
			R_recov.append(weight)
		return R_recov
	
	def recover(self):
		unique, counts = np.unique(self.binned_weights, return_counts=True)
		pd_recov = np.zeros(self.binned_weights.size)

		for u, nm in zip(unique, self.means):
			pd_recov[ binned_weights==u ] = nm
# 			print(u, nm)
# 			print(pd_recov)
		return pd_recov
	
	def get_cr(self, index_bits=0):
		full_size = self.binned_weights.size * 32
		codebook_size = 32 * (self.means.size + self.scale_size)
		min_idx_bit = 0
		min_idx_size = -1
		if (index_bits !=0):
			index_list, weight_list = self.encode(index_bits)
			index_size = len(index_list) * index_bits
			weight_size = len(weight_list) * np.ceil(np.log2(self.means.size))
			return ((full_size / (codebook_size + weight_size + index_size)), min_idx_bit)
		
		for index_bits in range (6,11):
			index_list, weight_list = self.encode(index_bits)
			index_size = len(index_list) * index_bits
			weight_size = len(weight_list) * np.ceil(np.log2(self.means.size))
			if (index_size + weight_size < min_idx_size or min_idx_size ==-1):
				min_idx_size = index_size + weight_size
				min_idx_bit = index_bits
		return ((full_size / (codebook_size + min_idx_size)), min_idx_bit)

	def get_cr_list(self):
		cm_size = {}
		for index_bits in range(6,11):
			cm_size[index_bits] = self.get_cr(index_bits)[0]
		return cm_size

def clamp_weights(model, means):
	weights = np.concatenate([model.state_dict()[array].clone().cpu().numpy().flatten() for i, array in enumerate(model.state_dict())])
	sorted_means = np.sort(means)
	bins = (sorted_means[1:] + sorted_means[:-1])/2
	new_means = []
	for i, b in enumerate(bins):
		if (i==0):
			if (np.where(weights < b)[0].size > 0):
				new_mean = (weights[np.where(weights < b)].sum()) / (np.where(weights < b)[0].size)
				weights[np.where(weights < b)] = new_mean
				new_means.append(new_mean)
				
			else:
				new_means.append(sorted_means[i])
		elif (i== len(bins)-1):
			if (np.where(weights > b)[0].size > 0):
				new_mean = (weights[np.where(weights > b)].sum()) / (np.where(weights > b)[0].size)
				weights[np.where(weights > b)] = new_mean
				new_means.append(new_mean)
			else:
				new_means.append(sorted_means[i])
			
			if (np.where(np.logical_and(weights < b, weights > prev_b))[0].size > 0):
				new_mean = (weights[np.where(np.logical_and(weights < b, weights > prev_b))].sum()) / (np.where(np.logical_and(weights < b, weights > prev_b))[0].size)
				new_means.append(new_mean)
				weights[np.where(np.logical_and(weights < b, weights > prev_b))] = new_mean
				#weights[np.where(weights > b)] = new_mean
				#weights[np.where(np.logical_and(weights < b, weights > prev_b))] = sorted_means[i]
			else:
				new_means.append(sorted_means[i])
			
		else:
			if (np.where(np.logical_and(weights < b, weights > prev_b))[0].size > 0):
				if (0 < b and 0 > prev_b):
					new_mean = 0
				else:
					new_mean = weights[np.where(np.logical_and(weights < b, weights > prev_b))].sum() / np.where(np.logical_and(weights < b, weights > prev_b))[0].size
				weights[np.where(np.logical_and(weights < b, weights > prev_b))] = new_mean
				new_means.append(new_mean)
			else:
				new_means.append(sorted_means[i])
		#print (new_means)
		prev_b = b
	out = weights
	pruned_state_dict = copy.deepcopy(model.state_dict())
	dim_start = 0
	for i, layer in enumerate(model.state_dict()):
		elems = model.state_dict()[layer].numel()
		pruned_state_dict[layer] = torch.from_numpy(np.array(out[dim_start:dim_start + elems]).reshape(model.state_dict()[layer].shape))
		dim_start += elems

	model.load_state_dict(pruned_state_dict)
	return (model, np.array(new_means))

def sws_replace(model_orig, layer_models):
	new_model = copy.deepcopy(model_orig)
	new_dict = new_model.state_dict()
	for model_dict in layer_models:
		for layer in model_dict:
			new_dict[layer] = model_dict[layer]
	new_model.load_state_dict(new_dict)
	return new_model