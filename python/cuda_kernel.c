#include <math.h>
// the sim function
__host__ __device__ float rbf_kernel(float a, float b)
{
	float sim;
	float d;
	float sigma = 10.;
	float beta = 0.5/sigma/sigma;

	d = a-b; 
	sim = exp(-beta*(d*d));
	return sim;
}

__global__ void create_kron_mat( int *edges_index_1, int *edges_index_2, 
                                 float *edges_pssm_1, float *edges_pssm_2, 
                                 int *edges_index_product, float *edges_weight_product,
                                 int n_edges_1, int n_edges_2,
                                 int n_nodes_2)
{

	int tx = threadIdx.x + blockDim.x * blockIdx.x;
	int ty = threadIdx.y + blockDim.y * blockIdx.y;
	int ind,len;

	if ( (tx < n_edges_1) && (ty < n_edges_2) ){ 

		
		len = 40;
		ind = tx * n_edges_2 + ty;

		float sim = 0;
		for(int i=0;i<len;i++){
			sim += rbf_kernel(edges_pssm_1[tx*len + i],edges_pssm_2[ty*len + i]);
		}			

		edges_weight_product[ind] = sim;
		edges_index_product[2*ind]      = edges_index_1[2*tx]   * n_nodes_2   + edges_index_2[2*ty];
		edges_index_product[2*ind + 1]  = edges_index_1[2*tx+1] * n_nodes_2   + edges_index_2[2*ty+1];
	}
}


__global__ void create_nodesim_mat(float *nodes_pssm_1, float *nodes_pssm_2, float *W0, int n_nodes_1, int n_nodes_2)
{

	int tx = threadIdx.x + blockDim.x * blockIdx.x;
	int ty = threadIdx.y + blockDim.y * blockIdx.y;
	int len,ind;

	if ( (tx<n_nodes_1) && (ty<n_nodes_2))
	{

		len = 20;
		ind = tx * n_nodes_2 + ty;

		float sim = 0;
		for (int i=0;i<len;i++){
			sim += rbf_kernel(nodes_pssm_1[ tx*len + i ],nodes_pssm_2[ty*len + i]);
		}
		W0[ind] = sim;
	}
}

__global__ void create_p_vect(float *node_info1, float* node_info2, float *p, int n_nodes_1, int n_nodes_2)
{

	int tx = threadIdx.x + blockDim.x * blockIdx.x;
	int ty = threadIdx.y + blockDim.y * blockIdx.y;
	float cutoff = 0.5;

	if ( (tx < n_nodes_1) && (ty < n_nodes_2) )
	{ 
		int ind = tx * n_nodes_2 + ty;
		if ( (node_info1[tx] < cutoff) && (node_info2[ty] < cutoff))
			p[ind] = 0;
		else
			p[ind] = node_info1[tx] * node_info2[ty];
	}
}