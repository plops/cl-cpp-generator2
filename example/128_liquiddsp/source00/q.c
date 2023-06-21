// liquidsdr.org/blog/lms-equalizer/
// gcc -Wall -O2 -lm -lc -o q.c -o equalizer
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>

int main() {
    unsigned int M           = 1 << 3;  // phase-shift keying mod. order
    unsigned int num_symbols = 1200;    // number of symbols to simulate
    unsigned int w_len       = 13;      // equalizer length
    float        mu          = 0.05f;   // equalizer learning rate
    float        alpha       = 0.60f;   // channel filter bandwidth

    // create and initialize arrays
    unsigned int  n, i, buf_index=0;
    float complex w[w_len];             // equalizer coefficients
    float complex b[w_len];             // equalizer buffer
    for (i=0; i<w_len; i++) w[i] = (i == w_len/2) ? 1.0f : 0.0f;
    for (i=0; i<w_len; i++) b[i] = 0.0f;

    float complex x_prime = 0;
    for (n=0; n<num_symbols; n++) {
        // 1. generate random transmitted phase-shift keying symbol
        float complex x = cexpf(_Complex_I*2*M_PI*(float)(rand() % M)/M);

        // 2. compute received signal and store in buffer
        float complex y = sqrt(1-alpha)*x + alpha*x_prime;
        x_prime = y;
        b[buf_index] = y;
        buf_index = (buf_index+1) % w_len;

        // 3. compute equalizer output
        float complex r = 0;
        for (i=0; i<w_len; i++)
            r += b[(buf_index+i)%w_len] * conjf(w[i]);

        // 4. compute 'expected' signal (blind), skip first w_len symbols
        float complex e = n < w_len ? 0.0f : r - r/cabsf(r);

        // 5. adjust equalization weights
        for (i=0; i<w_len; i++)
            w[i] = w[i] - mu*conjf(e)*b[(buf_index+i)%w_len];

        // print resulting symbols to screen
        printf("%8.3f %8.3f",   crealf(y), cimagf(y)); // received
        printf("%8.3f %8.3f\n", crealf(r), cimagf(r)); // equalized
    }
    return 0;
}
