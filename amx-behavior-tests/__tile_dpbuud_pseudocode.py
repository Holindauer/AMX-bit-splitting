import numpy as np  

"""
This script implements the pseudocode for the amx instruction __tile_dpbuud.
"""

def DPBD(c, x, y) -> np.int32:
    # Zero-extend the 8-bit integers (no need for explicit casting in numpy since it's already uint8)
    tmp1 = np.uint32(x[0]) * np.uint32(y[0])
    tmp2 = np.uint32(x[1]) * np.uint32(y[1])
    tmp3 = np.uint32(x[2]) * np.uint32(y[2])
    tmp4 = np.uint32(x[3]) * np.uint32(y[3])
    return c + tmp1 + tmp2 + tmp3 + tmp4

def write_row_and_zero(dst, m, tmp):
    # Write the computed values to row m of dst directly
    dst[m, :] = tmp[:]

def zero_upper_rows(dst, rows):
    # Zero out any remaining rows in dst
    dst[rows:, :] = 0

def __tile_dpbuud(dst, src0, src1):
    for m in range(dst.shape[0]):
        tmp = np.copy(dst[m])  # copy to not overwrite the original 
        
        for k in range(0, src0.shape[1], 4):  # Iterate in steps of 4
            for n in range(dst.shape[1]):  # Iterate through all columns
                x = src0[m, k:k+4]  # Extract 4 elements from src0
                y = src1[k:k+4, n]  # Extract 4 elements from src1
                tmp[n] = DPBD(tmp[n], x, y)
        
        write_row_and_zero(dst, m, tmp)
    
    zero_upper_rows(dst, dst.shape[0])

    return dst

# Test case: same memory layout, but different shapes
def test_1():

    # create (16,64) matrix where each row contains only 0's for row 0, 1's for row 1, etc.
    src1 = np.tile(np.arange(16, dtype=np.uint8)[:, np.newaxis], (1, 64))

    # src2 has same contiguous memory, but treated as (64, 16) matrix
    src2 = src1.copy().reshape(64, 16)

    dst = np.zeros((16, 16), dtype=np.uint32)

    __tile_dpbuud(dst, src1, src2)

    print(dst)

def test_2():

    A = np.arange(1024).reshape(16, 64)
    B = np.arange(1024).reshape(64, 16)
    true_result = np.dot(A, B)


    # perform matmul w/ __tile_dpbuud
    amx_result = np.zeros((16, 16), dtype=np.uint32)
    __tile_dpbuud(amx_result, A, B)

    # modular reduction by 257
    true_result = true_result % 257
    amx_result = amx_result % 257

    print(amx_result)
    print(true_result)




if __name__ == "__main__":
    # test_1()
    test_2()  
