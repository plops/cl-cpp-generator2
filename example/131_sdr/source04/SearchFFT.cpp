/**********************************************************
* SearchFFT.cpp - Extract GPS signal alignment details
* from a raw GPS sample bitstream
*
* Requires data file : gps.samples.1bit.I.fs5456.if4092.bin
*
* Requires Library FFTW - see http://www.fftw.org/
*
* Original author Andrew Holme
* Copyright (C) 2013 Andrew Holme
* http://www.holmea.demon.co.uk/GPS/Main.htm
*
**********************************************************/
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <iostream>

#include "fftw3.h"

#ifdef USE_FFTW_SPREC
 #define fftw_complex fftwf_complex
 #define fftw_malloc fftwf_malloc
 #define fftw_plan fftwf_plan
 #define fftw_plan_dft_1d fftwf_plan_dft_1d
 #define fftw_execute fftwf_execute
 #define fftw_destroy_plan fftwf_destroy_plan
 #define fftw_free fftwf_free
#endif

/******************************************
* Details of the space vehicles  
******************************************/
int SVs[] = { // PRN, Navstar, taps
     1,  63,  2,  6,
     2,  56,  3,  7,
     3,  37,  4,  8,
     4,  35,  5,  9,
     5,  64,  1,  9,
     6,  36,  2, 10,
     7,  62,  1,  8,
     8,  44,  2,  9,
     9,  33,  3, 10,
    10,  38,  2,  3,
    11,  46,  3,  4,
    12,  59,  5,  6,
    13,  43,  6,  7,
    14,  49,  7,  8,
    15,  60,  8,  9,
    16,  51,  9, 10,
    17,  57,  1,  4,
    18,  50,  2,  5,
    19,  54,  3,  6,
    20,  47,  4,  7,
    21,  52,  5,  8,
    22,  53,  6,  9,
    23,  55,  1,  3,
    24,  23,  4,  6,
    25,  24,  5,  7,
    26,  26,  6,  8,
    27,  27,  7,  9,
    28,  48,  8, 10,
    29,  61,  1,  6,
    30,  39,  2,  7,
    31,  58,  3,  8,
    32,  22,  4,  9,
};


/**********************************************************
* Class to generate the C/A gold codes that are transmitted
* by each Satallite. Combines the output of the G1 and G2
* LFSRs to produce the 1023 symbol sequence
**********************************************************/
struct CACODE { // GPS coarse acquisition (C/A) gold code generator.  Sequence length = 1023.
   char g1[11], g2[11], *tap[2];

   CACODE(int t0, int t1) {
      tap[0] = g2+t0;
      tap[1] = g2+t1;
      memset(g1+1, 1, 10);
      memset(g2+1, 1, 10);
   }

   int Chip() {
      return g1[10] ^ *tap[0] ^ *tap[1];
   }

   void Clock() {
      g1[0] = g1[3] ^ g1[10];
      g2[0] = g2[2] ^ g2[3] ^ g2[6] ^ g2[8] ^ g2[9] ^ g2[10];
      memmove(g1+1, g1, 10);
      memmove(g2+1, g2, 10);
   }
};


int main(int argc, char *argv[]) {
    /****************************
    * Detals of the input file
    ****************************/
    const int fc = 0; // 1023000; // 4092000; // or 1364000
    const int fs = 1023000; // 5456000;
    const char *in = "gps.samples.1bit.I.fs5456.if4092.bin";
    const int ms  = 1;    // Length of data to process (milliseconds)
    const int Start = 0;

    /**************************************
    * Derived values
    **************************************/
    const int Len = ms*fs/1000;

    ///////////////////////////////////////////////////////////////////////////////////////////////

    fftw_complex *code = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * Len);
    fftw_complex *data = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * Len);
    fftw_complex *prod = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * Len);

    fftw_plan p;

    /***************************************
    * Read in the file file . Data is
    * packed LSB first (sample 0 in bit 0)
    ****************************************/

    int i;
    /*

    static char sample_data[Len+7];
    int i, j, ch;

    FILE *fp = fopen(in, "r");
    if (!fp) { perror(in); return 0; }
    fseek(fp, Start, SEEK_SET);

    for (i=0; i<Len; i+=8) {
        ch = fgetc(fp);
        for (j=0; j<8; j++) {
            sample_data[i+j] = ch&1;
            ch>>=1;
        }
    }

    fclose(fp);
    */

    printf("PRN Nav Doppler   Phase   MaxSNR\n");
    /********************************************
    *  Process all space vehicles in turn
    ********************************************/
    for (int sv=0; sv<sizeof(SVs)/sizeof(int); ) {

        int PRN     = SVs[sv++];
        int Navstar = SVs[sv++];
        int T1      = SVs[sv++];
        int T2      = SVs[sv++];

        if (!PRN) break;

        /*************************************************
        * Generate the C/A code for the window of the
        * data (at the sample rate used by the data stream
        *************************************************/

        CACODE ca(T1, T2);

        double ca_freq = 1023000, ca_phase = 0, ca_rate = ca_freq/fs;

        for (i=0; i<Len; i++) {

            code[i][0] = ca.Chip() ? -1:1;
            code[i][1] = 0;

            ca_phase += ca_rate;

            if (ca_phase >= 1) {
                ca_phase -= 1;
                ca.Clock();
            }
        }

        for (i=0;i<Len; i++){
            std::cout << PRN << " " << Navstar << " " <<  i << " " << code[i][0] << std::endl;
        }
        /******************************************
        * Now run the FFT on the C/A code stream  *
        ******************************************/
        p = fftw_plan_dft_1d(Len, code, code, FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_execute(p);
        fftw_destroy_plan(p);


        /*************************************************
        * Now generate the same for the sample data, but
        * removing the Local Oscillator from the samples.
        *************************************************/
//
//        const int lo_sin[] = {1,1,0,0};
//        const int lo_cos[] = {1,0,0,1};
//
//        double lo_freq = fc, lo_phase = 0, lo_rate = lo_freq/fs*4;
//
//        for (i=0; i<Len; i++) {
//
//            data[i][0] = (sample_data[i] ^ lo_sin[int(lo_phase)]) ? -1:1;
//            data[i][1] = (sample_data[i] ^ lo_cos[int(lo_phase)]) ? -1:1;
//            //printf("%i %i\n",lo_sin[int(lo_phase)],lo_cos[int(lo_phase)]);
//            lo_phase += lo_rate;
//            if (lo_phase >= 4) lo_phase -= 4;
//        }
//
//        p = fftw_plan_dft_1d(Len, data, data, FFTW_FORWARD, FFTW_ESTIMATE);
//        fftw_execute(p);
//        fftw_destroy_plan(p);

        /***********************************************
        * Generate the execution plan for the Inverse
        * FFT (which will be reused multiple times
        ***********************************************/
//
//        p = fftw_plan_dft_1d(Len, prod, prod, FFTW_BACKWARD, FFTW_ESTIMATE);
//
//        double max_snr=0;
//        int max_snr_dop, max_snr_i;

        /************************************************
        * Test at different doppler shifts (+/- 5kHz)
        ************************************************/
//        for (int dop=-5000*Len/fs; dop<=5000*Len/fs; dop++) {
//            double max_pwr=0, tot_pwr=0;
//            int max_pwr_i;
//
//            /*********************************************
//            * Complex muiltiply the C/A code spectrum
//            * with the spectrum that came from the data
//            ********************************************/
//            for (i=0; i<Len; i++) {
//                int j=(i-dop+Len)%Len;
//                prod[i][0] = data[i][0]*code[j][0] + data[i][1]*code[j][1];
//                prod[i][1] = data[i][0]*code[j][1] - data[i][1]*code[j][0];
//            }
//
//
//            /**********************************
//            * Run the inverse FFT
//            **********************************/
//            fftw_execute(p);
//
//            /*********************************
//            * Look through the result to find
//            * the point of max absolute power
//            *********************************/
//            for (i=0; i<fs/1000; i++) {
//                double pwr = prod[i][0]*prod[i][0] + prod[i][1]*prod[i][1];
//                if (pwr>max_pwr) max_pwr=pwr, max_pwr_i=i;
//                tot_pwr += pwr;
//            }
//            /*****************************************
//            * Normalise the units and find the maximum
//            *****************************************/
//            double ave_pwr = tot_pwr/i;
//            double snr = max_pwr/ave_pwr;
//            if (snr>max_snr) max_snr=snr, max_snr_dop=dop, max_snr_i=max_pwr_i;
//        }
//        fftw_destroy_plan(p);
//
//
//        /*****************************************
//        * Display the result
//        *****************************************/
//        printf("%-2d %4d %7.0f %8.1f %7.1f    ",
//            PRN,
//            Navstar,
//            max_snr_dop*double(fs)/Len,
//            (max_snr_i*1023.0)/(fs/1000),
//            max_snr);
//
//        for (i=int(max_snr)/10; i--; ) putchar('*');
//        putchar('\n');
//
//
   }

    fftw_free(code);
    fftw_free(data);
    fftw_free(prod);
    return 0;
}