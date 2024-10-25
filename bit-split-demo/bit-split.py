# This script illustrates how to multiply 2 16-bit integers by first splitting
# them into high and low bits, then foiling the results and recombining them

# 16 bit numbers to multiply
number1 = 355
number2 = 399

A = (number1 >> 8) & 0xFF  # Higher 8 bits
B = number1 & 0xFF         # Lower 8 bits

C = (number2 >> 8) & 0xFF  # Higher 8 bits
D = number2 & 0xFF         # Lower 8 bits

# foil
AC = A * C
AD = A * D
BC = B * C
BD = B * D
high_term = (BC + AD) << 8       # mul by 256
low_term = ((AC << 8) << 8) + BD # mul by 256 * 256

# recombine
product_split = high_term + low_term

# actual product
product_og = number1 * number2  

# check
print(product_split, product_og, (product_split == product_og))
assert product_split == product_og
