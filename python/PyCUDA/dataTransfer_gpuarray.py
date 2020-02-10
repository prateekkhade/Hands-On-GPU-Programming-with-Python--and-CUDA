'''
Transferring data to and from the GPU with gpuarray, doing some arithmetic(multiplting by 2 in this case).
'''
import numpy as np
import pycuda.autoinit # Here we use autoinit
from pycuda import gpuarray

### Host_data
host_data= np.arange(10, dtype= np.float32)

### Device_data
device_data= gpuarray.to_gpu(host_data)

### Device_data_x2
device_data_x2= 2*device_data

### Host_data_x2
host_data_x2= device_data_x2.get()

### Let us look at the data
print("Before transformation: {}\nAfter transformation: {}".format(host_data, host_data_x2))