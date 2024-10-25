#include <stdio.h>
#include <stdint.h>

int main() {

    // two 16-bit numbers to multiply
    uint16_t number1 = 355;
    uint16_t number2 = 399;

    // number 1
    uint8_t A = (number1 >> 8) & 0xFF; // Higher 8 bits
    uint8_t B = number1 & 0xFF;        // Lower 8 bits

    // number 2 
    uint8_t C = (number2 >> 8) & 0xFF; // Higher 8 bits
    uint8_t D = number2 & 0xFF;        // Lower 8 bits

    // foil (TMUL unit will perform this type casting under the hood)
    uint32_t AC = (uint32_t)A * (uint32_t)C;
    uint32_t AD = (uint32_t)A * (uint32_t)D;
    uint32_t BC = (uint32_t)B * (uint32_t)C;
    uint32_t BD = (uint32_t)B * (uint32_t)D;

    // combine low-low and high-high products
    uint32_t high_term = (BC + AD) << 8;    // Multiply by 256 (shift left by 8)
    uint32_t low_term = ((AC << 16) + BD);  // Multiply by 256 * 256 (shift left by 16)

    // recombine low w/ high
    uint32_t product_split = high_term + low_term;

    // Calculate the multiplication of the original numbers for comparison
    uint32_t product_og = (uint32_t)number1 * (uint32_t)number2;

    // Output the results
    printf("Product using split method: %u\n", product_split);
    printf("Product using original method: %u\n", product_og);
    printf("Are the products equal? %s\n", (product_split == product_og) ? "Yes" : "No");

    return 0;
}
