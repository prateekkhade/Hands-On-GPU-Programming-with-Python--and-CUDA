'''
My own version of deviceQuery.py written as per the book - Hand On GPU programming with PyCuda
'''
import pycuda.driver as drv
drv.init()

# Print detected device
print('Detected {} CUDA Capable device(s)'.format(drv.Device.count()))

# Loop over devices
for i in range(drv.Device.count()):
	gpu_device= drv.Device(1)
	print('Device {}: {}'.format(i, gpu_device.name()))
	compute_capability= float(gpu_device.compute_capability())

	print('Compute capability: {}\nTotal memory: {}'.format(compute_capability))
