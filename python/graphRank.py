import scipy.io as spio
import scipy.sparse as sp_sparse
import numpy as np 
import timeit 
from time import time
import itertools 

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

	def __init__(self,testIDs='testID.lst',trainIDs='trainID.lst',graph_path='../graphMAT/'):

		self.trainIDs = trainIDs
		self.testIDs = testIDs
		self.graph_path = graph_path

	##############################################################
	#
	# Import the data from the precomputed Matlab file
	#
	##############################################################

	def import_from_mat(self):

		self.train_graphs = {}
		train_names = self._get_file_names(self.trainIDs)
		for name in train_names:
			self.train_graphs[name] = self._import_single_graph_from_mat(self.graph_path + '/' + name)

		self.test_graphs = {}
		test_names  = self._get_file_names(self.testIDs)
		for name in test_names:
			self.test_graphs[name] = self._import_single_graph_from_mat(self.graph_path + '/' + name)

	@staticmethod
	def _import_single_graph_from_mat(fname):
		data = spio.loadmat(fname,squeeze_me=True)['G']
		return GraphMat(data,fname.split('.')[0])

	@staticmethod
	def _get_file_names(filename):
		with open(filename) as f:
			names = [name.split()[0] for name in f.readlines() if name.split()]
		return names



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
		print('K          : %f' %(time()-t0))
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
		print(np.sum(self.Wx))
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
	def compile_kernel(self,fname):
		t0 = time()
		kernel_code = open(fname, 'r').read()
		self.mod = compiler.SourceModule(kernel_code)
		print('GPU - Kern : %f' %(time()-t0))

	# kronecker matrix with the edges pssm
	def compute_kron_mat_cuda(self,g1,g2):

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

		weight_product = gpuarray.zeros(n_edges_prod, np.float32)
		index_product = gpuarray.zeros((n_edges_prod,2), np.int32)

		block = (5,5,1)
		dim = (n1,n2,1)
		grid = tuple([int(np.ceil(n/t)) for n,t in zip(dim,block)])

		create_kron_mat_gpu (ind1,ind2,
							 pssm1,pssm2,
			                 index_product,weight_product,
			                 n1,n2,g2.num_nodes,
							 block=block,grid=grid)

		ind = index_product.get()
		ind = (ind[:,0],ind[:,1])
		n_nodes_prod = g1.num_nodes*g2.num_nodes

		self.Wx = sp_sparse.coo_matrix( (weight_product.get(),ind),shape=( n_nodes_prod,n_nodes_prod ) )
		self.Wx += self.Wx.transpose()
		print(self.Wx)
		print(np.sum(self.Wx))
		driver.Context.synchronize()
		print('GPU - Kron : %f' %(time()-t0))


	# Px vector with the node info
	def compute_px_cuda(self,g1,g2):

		t0 = time()
		driver.Context.synchronize()

		create_p_vect = self.mod.get_function('create_p_vect')
		info1 = gpuarray.to_gpu(np.array(g1.nodes_info_data).astype(np.float32))
		info2 = gpuarray.to_gpu(np.array(g2.nodes_info_data).astype(np.float32))

		n_nodes_prod = g1.num_nodes*g2.num_nodes
		pvect = gpuarray.zeros(n_nodes_prod,np.float32)

		block = (5,5,1)
		dim = (g1.num_nodes,g2.num_nodes,1)
		grid = tuple([int(np.ceil(n/t)) for n,t in zip(dim,block)])

		create_p_vect(info1,info2,pvect,g1.num_nodes,g2.num_nodes,block=block,grid=grid)
		self.px = pvect.get()
		driver.Context.synchronize()
		print('GPU - Px   : %f' %(time()-t0))

	# W0 matrix with the nodes pssm
	def compute_W0_cuda(self,g1,g2):

		t0 = time()
		driver.Context.synchronize()

		compute = self.mod.get_function('create_nodesim_mat')
		pssm1 = gpuarray.to_gpu(np.array(g1.nodes_pssm_data).astype(np.float32))
		pssm2 = gpuarray.to_gpu(np.array(g2.nodes_pssm_data).astype(np.float32))
		n_nodes_prod = g1.num_nodes*g2.num_nodes
		w0 = gpuarray.zeros(n_nodes_prod,np.float32)

		block = (5,5,1)
		dim = (g1.num_nodes,g2.num_nodes,1)
		grid = tuple([int(np.ceil(n/t)) for n,t in zip(dim,block)])

		compute(pssm1,pssm2,w0,g1.num_nodes,g2.num_nodes,block=block,grid=grid)
		self.W0 = w0.get()
		driver.Context.synchronize()
		print('GPU - W0   : %f' %(time()-t0))

		

if __name__ == "__main__":

	import argparse 
	parser = argparse.ArgumentParser(description=' test graphRank')
	parser.add_argument('--cuda',action='store_true')
	args = parser.parse_args()

	GR = graphRank()
	GR.import_from_mat()

	

	print('')
	print('-'*20)
	print('- timing')
	print('-'*20)
	print('')

	if args.cuda:
		GR.compile_kernel('./cuda_kernel.c')
		GR.compute_kron_mat_cuda(GR.test_graphs['2OZA'],GR.train_graphs['1IRA'])
		GR.compute_px_cuda(GR.test_graphs['2OZA'],GR.train_graphs['1IRA'])
		GR.compute_W0_cuda(GR.test_graphs['2OZA'],GR.train_graphs['1IRA'])
	else:
		GR.compute_kron_mat(GR.test_graphs['2OZA'],GR.train_graphs['1IRA'])
		GR.compute_px(GR.test_graphs['2OZA'],GR.train_graphs['1IRA'])
		GR.compute_W0(GR.test_graphs['2OZA'],GR.train_graphs['1IRA'])		

	K = GR.compute_K(lamb=1,walk=4)
	Kcheck = spio.loadmat('../kernelMAT/K_testID.mat')['K'][0][0]

	print('')
	print('-'*20)
	print('- Accuracy')
	print('-'*20)
	print('')

	print('K      :  ' + '  '.join(list(map(lambda x: '{:1.3}'.format(x),K))))
	print('Kcheck :  ' + '  '.join(list(map(lambda x: '{:1.3}'.format(x),Kcheck))))
	#assert np.allclose(K,Kcheck)







# setup = '''
# from __main__ import graphRank
# GR = graphRank()
# '''
# stmt = '''
# GR.import_from_mat()
# '''
# n = 100
# t = timeit.Timer(setup=setup,stmt=stmt).timeit(n)
# print('Import of the graph takes %1.3f s. ' %(t/n))



# setup = '''
# from __main__ import graphRank
# GR = graphRank()
# GR.import_from_mat()
# '''
# stmt = '''
# GR.compute_kron_mat(GR.train_graphs['1IRA'],GR.test_graphs['2OZA'])
# '''
# n = 10
# t = timeit.Timer(setup=setup,stmt=stmt).timeit(n)
# print('Compute the bi-graph takes %1.3f s. ' %(t/n))

