#!/usr/bin/env python

import scipy.io as spio
import scipy.sparse as sp_sparse
import numpy as np 
from time import time
import itertools 
from collections import OrderedDict
import pickle 
import os

try:
	from pycuda import driver, compiler, gpuarray, tools
	import pycuda.autoinit 
except:
	print('Warning : pycuda not found')


class GraphMat(object):
	def __init__(self,G,name='mol'):
		
		#self.nodes_pssm = { name : pssm for name,pssm in zip( G['Nodes'][()]['Name'][()],G['Nodes'][()]['pssm'][()])}
		#self.edges = G['Edges'][()]['EndNodes'][()]
		self.name = name
		self.nodes_pssm_data = np.array([p.tolist() for p in G['Nodes'][()]['pssm'][()]])
		self.nodes_info_data = G['Nodes'][()]['info'][()]
		self.num_nodes = np.int32(len(self.nodes_info_data))

		self.edges_index = np.array(G['Edges'][()]['idx'][()])-1
		self.num_edges = np.int32(len(self.edges_index))
		
		self.edges_pssm = []
		for ind in self.edges_index:
			self.edges_pssm.append( self.nodes_pssm_data[ind[0]].tolist() + self.nodes_pssm_data[ind[1]].tolist()  )
		self.edges_pssm = np.array(self.edges_pssm)

class graphRank(object):

	def __init__(self,testIDs='testID.lst',trainIDs='trainID.lst',graph_path='graphMAT/',gpu_block=(8,8,1)):

		# all the options
		self.trainIDs = trainIDs
		self.testIDs = testIDs
		self.graph_path = graph_path		
		self.gpu_block = gpu_block

		# the cuda kernel
		self.kernel = os.path.dirname(os.path.abspath(__file__)) + '/cuda_kernel.c'
		assert os.path.isfile(self.kernel)

	##############################################################
	#
	# Import the data from the precomputed Matlab file
	#
	##############################################################

	def import_from_mat(self):

		self.max_num_edges_train = 0
		self.max_num_edges_test = 0

		if not os.path.isfile(self.trainIDs):
			raise FileNotFoundError('file %s not found' %(self.trainIDs))

		if not os.path.isfile(self.testIDs):
			raise FileNotFoundError('file %s not found' %(self.testIDs))

		if not os.pathisdir(self.graph_path):
			raise NotADirectoryError('Directory %s not found' %(self.graph_path))

		self.train_graphs = {}
		train_names = self._get_file_names(self.trainIDs)

		self.test_graphs = {}
		test_names  = self._get_file_names(self.testIDs)
		
		for name in train_names:
			self.train_graphs[name] = self._import_single_graph_from_mat(self.graph_path + '/' + name)
			self.max_num_edges_train = np.max([self.max_num_edges_train,self.train_graphs[name].num_edges])
		
		for name in test_names:
			self.test_graphs[name] = self._import_single_graph_from_mat(self.graph_path + '/' + name)
			self.max_num_edges_test = np.max([self.max_num_edges_test,self.test_graphs[name].num_edges])

	@staticmethod
	def _import_single_graph_from_mat(fname):
		data = spio.loadmat(fname,squeeze_me=True)['G']
		return GraphMat(data,fname.split('.')[0])

	@staticmethod
	def _get_file_names(filename):
		with open(filename) as f:
			names = [name.split()[0] for name in f.readlines() if name.split()]
		return names

	# try to find the check file. 
	# return the filename is it's not None
	# return ./kernelMAT/K_<test_name>.mat if this file exists
	# return None otherwise
	def get_check_file(self,fname):
		if fname is not None:
			return fname
		else:
			test_name = os.path.splitext(self.testIDs)[0]
			kname = 'kernelMAT/K_' + test_name + '.mat'
			if os.path.isfile(kname):
				return kname 
			else:
				return None


	##############################################################
	#
	# Main function to compute all the data
	#
	##############################################################
	
	def run(self,lamb,walk,outfile='kernel.pkl',cuda=False,gpu_block=(8,8,1),check=None):

		# do all the single-time cuda operations
		if args.cuda:

			# compile the kernel
			GR.compile_kernel()

			# prebook the weight and index matrix
			# required for the calculation of self.Wx
			t0 = time()
			n1 = self.max_num_edges_test
			n2 = self.max_num_edges_train
			n_edges_prod = 2*n1*n2
			self.weight_product = gpuarray.zeros(n_edges_prod, np.float32)
			self.index_product = gpuarray.zeros((n_edges_prod,2), np.int32)
			print('GPU - Mem  : %f' %(time()-t0))

		# store the parametres
		K = {}
		K['lambda'] = lamb
		K['walk'] = walk
		K['cuda'] = cuda
		K['gpu_block'] = gpu_block

		# check file if it exists
		Kcheck = spio.loadmat(check)['K']

		# go through all the data
		for i1,(name1,G1) in enumerate(self.test_graphs.items()):
			for i2,(name2,G2) in enumerate(self.train_graphs.items()):

				print('')
				print(name1,name2)
				print('-'*20)
				# compute the matrices
				if args.cuda:
					self.compute_kron_mat_cuda(G1,G2)
					self.compute_px_cuda(G1,G2)
					self.compute_W0_cuda(G1,G2)
			
				else:
					self.compute_kron_mat(G1,G2)
					self.compute_px(G1,G2)
					self.compute_W0(G1,G2)		

				# compute/print the kernel
				K[(name1,name2)] = self.compute_K(lamb=lamb,walk=walk)
				print('-'*20)
				print('K      :  ' + '  '.join(list(map(lambda x: '{:1.3}'.format(x),K[(name1,name2)]))))
				
				# print the check if present
				if check is not None:
					print('Kcheck :  ' + '  '.join(list(map(lambda x: '{:1.3}'.format(x),Kcheck[i1][i2]))))

		# save the data
		pickle.dump(K,open(outfile,'wb'))


	########################################################
	#
	#	Test the routine on a single pair
	#
	########################################################
	def test(self,lamb,walk,cuda=False,gpu_block=(8,8,1), check=None,itest=None,itrain=None):

		if check is None:
			print('Warning: the check file containing the precomputed kernel was not found')

		print('')
		print('-'*20)
		print('- timing')
		print('-'*20)
		print('')

		# get a random index of the test/train
		if itest is None:
			itest = np.random.randint(len(self.test_graphs))
		if itrain is None:
			itrain = np.random.randint(len(self.train_graphs))

		# get the names
		test_name = list(self.test_graphs.keys())[itest]
		train_name = list(self.train_graphs.keys())[itrain]

		if cuda:
			self.compile_kernel()
			self.compute_kron_mat_cuda(self.test_graphs[test_name],self.train_graphs[train_name])
			self.compute_px_cuda(self.test_graphs[test_name],self.train_graphs[train_name])
			self.compute_W0_cuda(self.test_graphs[test_name],self.train_graphs[train_name])
	
		else:
			self.compute_kron_mat(self.test_graphs[test_name],self.train_graphs[train_name])
			self.compute_px(self.test_graphs[test_name],self.train_graphs[train_name])
			self.compute_W0(self.test_graphs[test_name],self.train_graphs[train_name])		

		K = GR.compute_K(lamb=lamb,walk=walk)
		
		print('')
		print('-'*20)
		print('- Accuracy')
		print('-'*20)
		print('')
		print('K      :  ' + '  '.join(list(map(lambda x: '{:1.3}'.format(x),K))))
		if check is not None:
			Kcheck = spio.loadmat(check)['K'][itest][itrain]
			print('Kcheck :  ' + '  '.join(list(map(lambda x: '{:1.3}'.format(x),Kcheck))))

	##############################################################
	#
	# Calculation of the K using CPU
	#
	##############################################################
	
	def compute_K(self,lamb=1,walk=4):

		K = []
		t0 = time()
		self.px /= np.sum(self.px)
		K.append(np.dot(self.px*self.W0,self.px))
		pW = self.Wx.transpose().dot(self.px)
		
		for i in range(1,walk+1):
			K.append(   K[i-1] + lamb**i * np.dot(pW,self.px) )
			pW = self.Wx.transpose().dot(pW)
		print('CPU - K    : %f' %(time()-t0))
		return K

	##############################################################
	#
	# Properties calculation CPU
	#
	##############################################################

	# Kroenecker matrix calculation edges pssm similarity 
	def compute_kron_mat(self,g1,g2):
		t0 = time()
		row,col,weight = [],[],[]
		n1,n2 = g1.num_edges,g2.num_edges
		N = n1*n2

		# double the edges index for g1
		index1 = np.vstack((g1.edges_index,np.flip(g1.edges_index,axis=1)))
		index2 = g2.edges_index

		# double the pssm edges for g1
		pssm1 = np.vstack((g1.edges_pssm,np.hstack((g1.edges_pssm[:,20:],g1.edges_pssm[:,:20]))))
		pssm2 = g2.edges_pssm

		# compute the weight 
		weight  = np.array([ self._rbf_kernel(p[0],p[1]) for p in itertools.product(*[pssm1,pssm2]) ])
		ind     = np.array([ self._get_index(k[0],k[1],g2.num_nodes)  for k in itertools.product(*[index1,index2])])
		index = ( ind[:,0].tolist(),ind[:,1].tolist() )
				
		# final size	
		n_nodes_prod = g1.num_nodes*g2.num_nodes

		# create the Wx matrix
		self.Wx = sp_sparse.coo_matrix( (weight,index),shape=( n_nodes_prod,n_nodes_prod ) )
		self.Wx += self.Wx.transpose()
		print('CPU - Kron : %f' %(time()-t0))

	# px vector calculation with nodes info
	def compute_px(self,g1,g2,cutoff=0.5):

		t0 = time()
		n1,n2 = g1.num_nodes,g2.num_nodes
		self.px = [t[0]*t[1] if (float(t[0])>cutoff or float(t[1])>cutoff) else 0 for t in itertools.product(*[g1.nodes_info_data,g2.nodes_info_data])]
		print('CPU - Px   : %f' %(time()-t0))

	# W0 alculation : nodes pssm similarity
	def compute_W0(self,g1,g2):
		t0 = time()
		self.W0  = np.array([ self._rbf_kernel(p[0],p[1]) for p in itertools.product(*[g1.nodes_pssm_data,g2.nodes_pssm_data]) ])
		print('CPU - W0   : %f' %(time()-t0))
	
	def _get_data(self,g1,g2,k):

		index1 = k[0]
		index2 = k[1]

		pssm1 = g1.nodes_pssm_data[index1[0]] + g1.nodes_pssm_data[index1[1]]
		pssm2 = g2.nodes_pssm_data[index2[0]] + g2.nodes_pssm_data[index2[1]]

		w   = self._rbf_kernel(pssm1,pssm2)
		ind = self._get_index(index1,index2,g2.num_nodes)

		return [w,ind[0],ind[1]]

	# Kernel for the edges pssm similarity calculation
	@staticmethod
	def _rbf_kernel(data1,data2,sigma=10):
		delta = np.sum((data1-data2)**2)
		beta = 2*sigma**2
		return np.exp(-delta/beta)

	@staticmethod
	def _get_index(index1,index2,size2):
		index = np.array(index1.tolist()) * size2 + np.array(index2.tolist())
		return index.tolist()


	##############################################################
	#
	# Properties calculation GPU
	#
	##############################################################

	# compile the cuda kernel
	def compile_kernel(self):
		t0 = time()
		kernel_code = open(self.kernel, 'r').read()
		self.mod = compiler.SourceModule(kernel_code)
		print('GPU - Kern : %f' %(time()-t0))

	# kronecker matrix with the edges pssm
	def compute_kron_mat_cuda(self,g1,g2,gpu_block=None):

		t0 = time()
		driver.Context.synchronize()
		create_kron_mat_gpu = self.mod.get_function('create_kron_mat')

		n1 = g1.num_edges
		n2 = g2.num_edges
		n_edges_prod = 2*n1*n2

		pssm1 = gpuarray.to_gpu(np.array(g1.edges_pssm).astype(np.float32))
		pssm2 = gpuarray.to_gpu(np.array(g2.edges_pssm).astype(np.float32))

		ind1 = gpuarray.to_gpu(np.array(g1.edges_index).astype(np.int32))
		ind2 = gpuarray.to_gpu(np.array(g2.edges_index).astype(np.int32))

		# create the gpu arrays only if we have to
		# i.e. in case we run the calculation once (test or tune)
		# in other cases the weigh and index are booked in self.run()
		if not hasattr(self,'weight_product'):
			self.weight_product = gpuarray.zeros(n_edges_prod, np.float32)
			self.index_product = gpuarray.zeros((n_edges_prod,2), np.int32)

		# get the gpu block size if specified
		if gpu_block is not None:
			block = gpu_block
		else:
			block = self.gpu_block
		dim = (n1,n2,1)
		grid = tuple([int(np.ceil(n/t)) for n,t in zip(dim,block)])

		driver.Context.synchronize()
		print('GPU - Mem  : %f \t (block size:%dx%d)' %(time()-t0,block[0],block[1]))

		# call the CUDA kernel
		t0 =time()
		create_kron_mat_gpu (ind1,ind2,
							 pssm1,pssm2,
			                 self.index_product,self.weight_product,
			                 n1,n2,g2.num_nodes,
							 block=block,grid=grid)

		# extract the data
		# restrict to the ones calculated here
		ind = self.index_product.get()
		ind = (ind[:n_edges_prod,0],ind[:n_edges_prod,1])
		w = self.weight_product.get()[:n_edges_prod]

		# final size
		n_nodes_prod = g1.num_nodes*g2.num_nodes

		# create the matrix
		self.Wx = sp_sparse.coo_matrix( (w,ind),shape=( n_nodes_prod,n_nodes_prod ) )
		self.Wx += self.Wx.transpose()

		driver.Context.synchronize()
		print('GPU - Kron : %f \t (block size:%dx%d)' %(time()-t0,block[0],block[1]))


	# Px vector with the node info
	def compute_px_cuda(self,g1,g2,gpu_block=None):

		t0 = time()
		driver.Context.synchronize()

		create_p_vect = self.mod.get_function('create_p_vect')
		info1 = gpuarray.to_gpu(np.array(g1.nodes_info_data).astype(np.float32))
		info2 = gpuarray.to_gpu(np.array(g2.nodes_info_data).astype(np.float32))

		n_nodes_prod = g1.num_nodes*g2.num_nodes
		pvect = gpuarray.zeros(n_nodes_prod,np.float32)

		if gpu_block is not None:
			block = gpu_block
		else:
			block = self.gpu_block
		dim = (g1.num_nodes,g2.num_nodes,1)
		grid = tuple([int(np.ceil(n/t)) for n,t in zip(dim,block)])

		create_p_vect(info1,info2,pvect,g1.num_nodes,g2.num_nodes,block=block,grid=grid)
		self.px = pvect.get()
		driver.Context.synchronize()
		print('GPU - Px   : %f \t (block size:%dx%d)' %(time()-t0,block[0],block[1]))

	# W0 matrix with the nodes pssm
	def compute_W0_cuda(self,g1,g2,gpu_block=None):

		t0 = time()
		driver.Context.synchronize()

		compute = self.mod.get_function('create_nodesim_mat')
		pssm1 = gpuarray.to_gpu(np.array(g1.nodes_pssm_data).astype(np.float32))
		pssm2 = gpuarray.to_gpu(np.array(g2.nodes_pssm_data).astype(np.float32))
		n_nodes_prod = g1.num_nodes*g2.num_nodes
		w0 = gpuarray.zeros(n_nodes_prod,np.float32)

		if gpu_block is not None:
			block = gpu_block
		else:
			block = self.gpu_block
		dim = (g1.num_nodes,g2.num_nodes,1)
		grid = tuple([int(np.ceil(n/t)) for n,t in zip(dim,block)])

		compute(pssm1,pssm2,w0,g1.num_nodes,g2.num_nodes,block=block,grid=grid)
		self.W0 = w0.get()
		driver.Context.synchronize()
		print('GPU - W0   : %f \t (block size:%dx%d)' %(time()-t0,block[0],block[1]))


	##############################################################
	#
	# kernel tuner
	#
	##############################################################

	def tune_kernel(self,func='create_kron_mat',test_all_func=False):

		try:
			from kernel_tuner import tune_kernel
		except:
			print('Install the Kernel Tuner : \n \t\t pip install kernel_tuner')
			print('http://benvanwerkhoven.github.io/kernel_tuner/')		


		test_name = list(self.test_graphs.keys())[0]
		train_name = list(self.train_graphs.keys())[0]

		g1 = self.test_graphs[test_name],
		g2 = self.train_graphs[train_name]

		tune_params = OrderedDict()
		tune_params['block_size_x'] = [2,4,8,16,32,64,128]
		tune_params['block_size_y'] = [2,4,8,16,32,64,128]

		kernel_code = open(self.kernel, 'r').read()
		tunable_kernel = self._tunable_kernel(kernel_code)

		try:

			if func == 'create_kron_mat' or test_all_func:

				func = 'create_kron_mat'
				print('\n')
				print('Tuning function %s from %s' %(func,self.kernel))
				print('-'*40)

				n1 = g1.num_edges
				n2 = g2.num_edges
				n_edges_prod = 2*n1*n2

				pssm1 = np.array(g1.edges_pssm).astype(np.float32)
				pssm2 = np.array(g2.edges_pssm).astype(np.float32)

				ind1 = np.array(g1.edges_index).astype(np.int32)
				ind2 = np.array(g2.edges_index).astype(np.int32)

				weight_product = np.zeros(n_edges_prod, np.float32)
				index_product = np.zeros((n_edges_prod,2), np.int32)

				dim = (n1,n2,1)
				args = [ind1,ind2,pssm1,pssm2,index_product,weight_product,n1,n2,g2.num_nodes]

				result = tune_kernel(func,tunable_kernel,dim,args,tune_params)

			if func == 'create_nodesim_mat' or test_all_func:

				func = 'create_nodesim_mat'
				print('\n')
				print('Tuning function %s from %s' %(func,self.kernel))
				print('-'*40)

				pssm1 = np.array(g1.nodes_pssm_data).astype(np.float32)
				pssm2 = np.array(g2.nodes_pssm_data).astype(np.float32)
				n_nodes_prod = g1.num_nodes*g2.num_nodes
				w0 = np.zeros(n_nodes_prod,np.float32)

				dim = (g1.num_nodes,g2.num_nodes,1)
				args = [pssm1,pssm2,w0,g1.num_nodes,g2.num_nodes]

				result = tune_kernel(func,tunable_kernel,dim,args,tune_params)

			if func == 'create_p_vect' or test_all_func:

				func = 'create_p_vect'
				print('\n')
				print('Tuning function %s from %s' %(func,self.kernel))
				print('-'*40)

				info1 = np.array(g1.nodes_info_data).astype(np.float32)
				info2 = np.array(g2.nodes_info_data).astype(np.float32)
				n_nodes_prod = g1.num_nodes*g2.num_nodes
				pvect = np.zeros(n_nodes_prod,np.float32)

				dim = (g1.num_nodes,g2.num_nodes,1)
				args = [info1,info2,pvect,g1.num_nodes,g2.num_nodes]

				result = tune_kernel(func,tunable_kernel,dim,args,tune_params)

		except:

			print('Function %s not found in %s' %(func,self.kernel))

	# tranform the kernel to a tunable one
	@staticmethod
	def _tunable_kernel(kernel):
		switch_name = { 'blockDim.x' : 'block_size_x', 'blockDim.y' : 'block_size_y' }
		for old,new in switch_name.items():
			kernel = kernel.replace(old,new)
		return kernel



if __name__ == "__main__":

	import argparse 
	parser = argparse.ArgumentParser(description=' test graphRank')

	# test and train IDS
	parser.add_argument('--testID', type=str, default='testID.lst',help='list of ID for testing. Default: testID.lst')
	parser.add_argument('--trainID', type=str, default='trainID.lst',help='list of ID for training. Default: trainID.lst')

	# graphs of the individual complex
	parser.add_argument('--graph',type=str,default='graphMAT',help='folder containing the graph of each complex. Default: graphMAT')

	# file containing the kernel for checking
	parser.add_argument('--check',type=str,default=None,help='file containing the kernel. Default: kernelMAT/<testID_name>.mat')

	# where to write the output file
	parser.add_argument('--outfile',type=str,default='kernel.pkl',help='Output file containing the calculated Kernel values. Default: kernel.pkl')

	# what to do:  tune the kernel, test the calculation, run the entire calculations
	parser.add_argument('--tune_kernel',action='store_true',help='Only tune the CUDA kernel')
	parser.add_argument('--test',action='store_true',help='Only test the functions on a single pair pair of graph ')

	# parameter of the calculations
	parser.add_argument('--lamb',type=float,default=1,help='Lambda parameter in the Kernel calculations. Default: 1')
	parser.add_argument('--walk',type=int,default=4,help='Max walk length in the Kernel calculations. Default: 4')
	

	# cuda parameters
	parser.add_argument('--func',type=str,default='all',help='functions to tune in the kernel. Defaut: all functions')
	parser.add_argument('--cuda',action='store_true', help='Use CUDA kernel')
	parser.add_argument('--gpu_block',nargs='+',default=[8,8,1],type=int,help='number of gpu block to use. Default: 8 8 1')

	args = parser.parse_args()


	# init and load the data
	GR = graphRank(testIDs=args.testID,trainIDs=args.trainID,graph_path=args.graph,gpu_block=tuple(args.gpu_block))
	GR.import_from_mat()

	# get the path of the check file
	checkfile = GR.get_check_file(args.check)

	# only tune the kernel
	if args.tune_kernel:
		GR.tune_kernel(func=args.func,test_all_func=args.func=='all')

	# only run a pair of graph with or w/o CUDA
	elif args.test:
		GR.test(lamb=args.lamb,walk=args.walk,cuda=args.cuda,gpu_block=args.gpu_block, check=checkfile)
	
	# run the entire calculation
	else :
		GR.run(lamb=args.lamb,walk=args.walk,outfile=args.outfile,cuda=args.cuda,gpu_block=tuple(args.gpu_block),check=checkfile)



