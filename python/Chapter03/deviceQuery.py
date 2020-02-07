'''
My own version of deviceQuery.py written as per the book - Hand On GPU programming with Python and CUDA
'''
import pycuda.driver as drv
drv.init()

# Print detected device
print('Detected {} CUDA Capable device(s)'.format(drv.Device.count()))

# Loop over devices
for i in range(drv.Device.count()):
	gpu_device= drv.Device(i)
	print('Device {}: {}'.format(i, gpu_device.name()))
	compute_capability= gpu_device.compute_capability()[0]

	print('Compute capability: {}\nTotal memory: {} MBs'.format(compute_capability, gpu_device.total_memory()//(1024**2)))
	
	device_attribute_tuples= gpu_device.get_attributes().items()
	device_attributes= {}

	for k, v in device_attribute_tuples:
		device_attributes[str(k)]= v

	# Number of multiprocessors
	num_mp= device_attributes['MULTIPROCESSOR_COUNT']

	# Lookup for cuda cores per mp
	cuda_cores_per_mp= {5.0: 128, 5.1: 128, 5.2: 128, 6.0: 64, 6.1: 128, 6.2: 128}[compute_capability]

	# Printing the number of multiprocessors and cuda cores per mp
	print('{} Multiprocessors\t{} CUDA Cores/Multiprocessor\t{} CUDA Cores'.format(num_mp, cuda_cores_per_mp, num_mp*cuda_cores_per_mp))

	# All attributes
	device_attributes.pop('MULTIPROCESSOR_COUNT')

	for k in device_attributes.keys():
		print('\t {}: {}'.format(k, device_attributes[k]))



