./imgui_soapysdr  -f 1575.42e6 -b 64512 -F 1000000 -r 10000000 -B 5e6

# -B 1.536e+06

# The 1575.42 MHz ± 12 MHz (10.23 MHz × 154) frequency is used to
# transmit the Global Positioning System (GPS)
# radionavigation-satellite service L1 signal for military, aviation,
# space, and commercial applications.


# My SDR only has a maximum sampling rate of 10MHz and 8MHz bandwidth
# is that enough to decode the C/A PRN codes and the navigation
# message of the GPS signal?

# The C/A code is transmitted on the L1 frequency as a 1.023 MHz
# signal using a bi-phase shift keying (BPSK) modulation
# technique. The P(Y)-code is transmitted on both the L1 and L2
# frequencies as a 10.23 MHz signal using the same BPSK modulation,
# however the P(Y)-code carrier is in quadrature with the C/A carrier
# (meaning it is 90° out of phase).

# The P(Y)-code is the encrypted high precision code (restricted use,
# not for public)

# The C/A (coarse/acqusition code) PRN codes are Gold codes with a
# period of 1023 chips transmitted at 1.023 Mchip/s, causing the code
# to repeat every 1 millisecond. They are exclusive-ored with a 50
# bit/s navigation message and the result phase modulates the carrier
# as previously described. These codes only match up, or strongly
# autocorrelate when they are almost exactly aligned.
