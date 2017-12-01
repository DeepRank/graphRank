# graphRank

To do anything with the code first go in the python directory
```
cd python
```

# Usage
You can check all the options of the code using 

``` 
python graphRank.py --help
```

```
usage: graphRank.py [-h] [--tune_kernel] [--test] [--lamb LAMB] [--walk WALK]
                    [--outfile OUTFILE] [--func FUNC] [--cuda]
                    [--gpu_block GPU_BLOCK [GPU_BLOCK ...]]

test graphRank

optional arguments:
  -h, --help            show this help message and exit
  --tune_kernel         Only tune the CUDA kernel if present
  --test                Only test the functions on a single pair pair of graph
                        if present
  --lamb LAMB           Lambda parameter in the Kernel calculations
  --walk WALK           Max walk length in the Kernel calculations
  --outfile OUTFILE     Output file containing the Kernel
  --func FUNC           Which functions to tune in the kernel (defaut all
                        functions)
  --cuda                Use CUDA kernel if present
  --gpu_block GPU_BLOCK [GPU_BLOCK ...]
                        number of gpu block to use (default 8 8 1)
```

# Test 

You can test the code using a precomputed example using

## CPU version
```
python graphRank.py --test
```

## GPU version
```
python graphRank.py --test --cuda
```

which should output (GPU version)

```
--------------------
- timing
--------------------

GPU - Kern : 0.111562
GPU - Mem  : 0.190918 	 (block size:8x8)
GPU - Kron : 0.081629 	 (block size:8x8)
GPU - Px   : 0.002048 	 (block size:8x8)
GPU - W0   : 0.001714 	 (block size:8x8)
CPU - K    : 0.024109

--------------------
- Accuracy
--------------------

K      :  1.57e-05  4.61e-05  0.000175  0.000491  0.00192
Kcheck :  1.57e-05  4.61e-05  0.000175  0.000491  0.00192
```

# Kernel Tuner

You can tune the gpu block/grid size using the kernel tuner. Simply type

```
python graphRank.py --tune_kernel [--func=<func_name>]
```

If you don't specify a function name (present in cuda_kernel.c) the code will tune all the functios. For each function it should output something like:

```
Tuning function create_kron_mat from ./cuda_kernel.c
----------------------------------------
Using: GeForce GTX 1080 Ti
block_size_x=2, block_size_y=2, time=0.905830395222
block_size_x=2, block_size_y=4, time=0.545791995525
block_size_x=2, block_size_y=8, time=0.355219191313
block_size_x=2, block_size_y=16, time=0.30387840271
block_size_x=2, block_size_y=32, time=0.27014400363
block_size_x=2, block_size_y=64, time=0.259091204405
block_size_x=2, block_size_y=128, time=0.250815996528
......
best performing configuration: block_size_x=8, block_size_y=8, time=0.161958396435
```

# Run

You can run the calculation on the entire training/test set using

```
python graphRank.py [--cuda] [--lamb=X] [--walk=X] [--outfile=name] [--gpu_block=i j k]
```

In the GPU case the code will first output the timing of the kernel compilation and GPU memory assignement 

```
GPU - Kern : 0.106779
GPU - Mem  : 0.146905
```


Then for each pair of graph present in the train/test set the code will output the following

```
2OZA 1IRA
--------------------
GPU - Mem  : 0.003217 	 (block size:8x8)
GPU - Kron : 0.079183 	 (block size:8x8)
GPU - Px   : 0.001758 	 (block size:8x8)
GPU - W0   : 0.001631 	 (block size:8x8)
CPU - K    : 0.023726
--------------------
K      :  1.57e-05  4.61e-05  0.000175  0.000491  0.00192
```




