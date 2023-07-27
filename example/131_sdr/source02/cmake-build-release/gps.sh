./imgui_soapysdr  -f 1575.42e6 -b 64512 -F 1000000 -r 10000000

# The 1575.42 MHz ± 12 MHz (10.23 MHz × 154) frequency is used to
# transmit the Global Positioning System (GPS)
# radionavigation-satellite service L1 signal for military, aviation,
# space, and commercial applications.

# My sdr only has a maximum sampling rate of 10MHz and 8MHz bandwidth
# is that enough?


# The C/A code is transmitted on the L1 frequency as a 1.023 MHz
# signal using a bi-phase shift keying (BPSK) modulation
# technique. The P(Y)-code is transmitted on both the L1 and L2
# frequencies as a 10.23 MHz signal using the same BPSK modulation,
# however the P(Y)-code carrier is in quadrature with the C/A carrier
# (meaning it is 90° out of phase).
