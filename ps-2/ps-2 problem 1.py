import numpy as np
def get_bits(number):
    bytes = number.tobytes()
    bits = []
    for byte in bytes:
        bits = bits + np.flip(np.unpackbits(np.uint8(byte)), np.uint8(0)).tolist()
    return list(reversed(bits))

value = np.float32(100.98763)
bitlist = get_bits(value)

exponent = bitlist[1:9]
mantissa = bitlist[9:32]
signbit = bitlist[0]

print("100.98763 Represented as 32bit floating point:",bitlist[0:32])
print("Exponent:", exponent)
print("Mantissa:", mantissa)
print("Signbit:",signbit)
print("Difference =", value - 100.98763)

