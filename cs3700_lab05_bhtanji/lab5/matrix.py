import numpy as np
from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit

kernel_template_sum = """
__global__ void sum(float *d_a, float *d_b, float *d_c) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int m_loc = row * %(MATRIX_SIZE)s + col;

    d_c[m_loc] = d_a[m_loc] + d_b[m_loc];
}

"""
kernel_template_mult = """
__global__ void product(float *d_a, float *d_b, float *d_c) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0;

    // %(MATRIX_SIZE)s is a substitution expression
    // it gets replaced by the actual size of the matrix
    // by the Host before this kernel is called
    for (int i = 0; i < %(MATRIX_SIZE)s; ++i) {
        float aEle = d_a[row * %(MATRIX_SIZE)s + i];
        float bEle = d_b[i * %(MATRIX_SIZE)s + col];
        sum += aEle * bEle;
    }

    d_c[row * %(MATRIX_SIZE)s + col] = sum;
}
"""

WIDTH = 6

# create two host random square matrices
host_a = np.random.randint(1, 10, size=(WIDTH, WIDTH)).astype(np.float32)
host_b = np.random.randint(1, 10, size=(WIDTH, WIDTH)).astype(np.float32)

# transfer host (CPU) memory to device (GPU) memory 
device_a = gpuarray.to_gpu(host_a) 
device_b = gpuarray.to_gpu(host_b)

# create empty gpu array for the result (C = A * B)
device_c = gpuarray.empty((WIDTH, WIDTH), np.float32)

# get the kernel code from the template 
# by specifying the constant MATRIX_SIZE
kernel_code = kernel_template_mult % {
   'MATRIX_SIZE': WIDTH 
}
# compile the kernel code 
mod = compiler.SourceModule(kernel_code)
# get the kernel function from the compiled module
product = mod.get_function("product")

# get the kernel code from the template 
# by specifying the constant MATRIX_SIZE
kernel_code = kernel_template_sum % {
   'MATRIX_SIZE': WIDTH 
}
# compile the kernel code 
mod = compiler.SourceModule(kernel_code)
# get the kernel function from the compiled module
sum_m = mod.get_function("sum")

# print the results
print("-" * 60)
print("Matrix A (GPU):")
print(device_a.get())

print("-" * 60)
print("Matrix B (GPU):")
print(device_b.get())

# call the kernel on the card
# configures WIDTH X WIDTH blocks with one thread in each block
product(device_a, device_b, device_c, 
    block = (1, 1, 1),
    grid = (WIDTH, WIDTH))
print("-" * 60)
print("A X B (GPU):")
print(device_c.get())


sum_m(device_a, device_b, device_c, 
    block = (1, 1, 1),
    grid = (WIDTH, WIDTH))
print("-" * 60)
print("A + B (GPU):")
print(device_c.get())
