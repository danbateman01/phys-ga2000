import numpy as np

# Using float32
float32_num = np.float32(1.0)
float32_smallest = np.nextafter(float32_num, np.float32(np.inf)) - float32_num
print("Smallest number added to 1.0 in float32:" ,float32_smallest)

# Using float64
float64_num = np.float64(1.0)
float64_smallest = np.nextafter(float64_num, np.float64(np.inf)) - float64_num
print("Smallest number added to 1.0 in float64:" ,float64_smallest)


min32 = np.float32(1e-45)
max32 = np.float32(1e38)
min64 = np.float64(1e-323)
max64 = np.float64(1e308)

print("minimum value represented by 32 floating bit=",min32)
print("maximum value represented by 32 floating bit=",min32)
print("minimum value represented by 32 floating bit=",min64)
print("maximum value represented by 32 floating bit=",max64)