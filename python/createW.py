import numpy as np 
from time import time

from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit

NDATA = 20
N = 50
M = 50

class edge:
	def __init__(self,nodes=(),weight=0,data=[]):
		self.nodes = nodes
		self.data = data
		self.weight = weight

class Graph:
	def __init__(self):
		self.edges = []
		self.adjmat = []

	def extract_data(self):
		nedges = len(self.edges)
		mat = []
		for i in range(nedges):
			mat.append(self.edges[i].data)
		return mat

	def __len__(self):
		return len(self.edges)



def create_graph(N):

	data = []
	for i in range(N):
		data.append([np.random.randint(-15,15) for i in range(NDATA)])
	adj = np.random.random_integers(0,1,size=(N,N))

	g = Graph()
	for i in range(N-1):
		idata = data[i]
		for j in range(i+1,N):
			if adj[i,j] == 1:
				jdata = data[j]
				g.edges.append( edge(nodes=(i,j),weight = 1.0,data = idata+jdata ) )	
	return g


def sim(idata,jdata):
	return np.sum([i+j for i,j in zip(idata,jdata)])
	return np.sum(np.array([np.exp(-np.abs(i-j)) for i,j in zip(idata,jdata)]))/2/NDATA


def createW(g1,g2):
	n1,n2 = len(g1.edges),len(g2.edges)
	W = Graph()
	W.adjmat = np.zeros((n1,n2))
	for i in range(n1):
		for  j in range(n2):
			nodes = (g1.edges[i].nodes,g2.edges[j].nodes)
			weight = sim(g1.edges[i].data,g2.edges[j].data)
			W.edges.append(  edge( nodes=nodes,weight=weight )  )
			W.adjmat[i,j] = weight
	return W




g1 = create_graph(N)
g2 = create_graph(M)
n1 = g1.__len__()
n2 = g2.__len__()
print(n1,n2)
t0 = time()
W = createW(g1,g2)
print('CPU : %f' %(time()-t0))


kernel_code_template = """
#include <math.h>
// the sim function
__host__ __device__ float eval(float a, float b)
{
	float sim;
	sim = exp(-fabs(a-b))/%(lendata)s/2;
	sim = a+b;
	return sim;
}

__global__ void createW( float *g1, float *g2, float *W)
{

	int tx = threadIdx.x + blockDim.x * blockIdx.x;
	int ty = threadIdx.y + blockDim.y * blockIdx.y;

	int len = 2*%(lendata)s;

	int start_1 = tx*len;
	int start_2 = ty*len;
	
	float sim = 0;
	for(int i=0;i<len;i++){
		sim += eval(g1[start_1+i],g2[start_2+i]);
	}

	W[tx * %(leng)s + ty] = sim;

}
"""

kernel_code = kernel_code_template % {'lendata' : NDATA, 'leng' : n2 }
mod = compiler.SourceModule(kernel_code)
createW_gpu = mod.get_function('createW')

mat1 = gpuarray.to_gpu(np.array(g1.extract_data()).astype(np.float32))
mat2 = gpuarray.to_gpu(np.array(g2.extract_data()).astype(np.float32))
Wgpu = gpuarray.zeros((n1,n2), np.float32)

block = (5,5,1)
dim = (n1,n2,1)
grid = tuple([int(np.ceil(n/t)) for n,t in zip(dim,block)])

t0 = time()
driver.Context.synchronize()
createW_gpu(mat1,mat2,Wgpu,block=block,grid=grid)
driver.Context.synchronize()
print('GPU : %f' %(time()-t0))
print('-'*10)
print(W.adjmat-Wgpu.get())
print('-'*10)
print('-'*10)
print(np.linalg.norm(W.adjmat-Wgpu.get()))