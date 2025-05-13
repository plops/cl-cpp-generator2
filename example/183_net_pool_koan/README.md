i need help with c++ architecture. i have three pools that can contain images (e.g. vector<byte> but the size is constant (128x128) throughout the runtime of the program (the vectors can be allocated for the pool once when the program starts (10 slots should be enough) and the vectors never need to be resized) ) metadata (struct Meta { int i; float a; }, 12 slots are required) or measurement (struct Measurement { double q; double p; }, 20 slots are required)

these 3 types of packages arrive over network as tcp packets (the packet has an u8 indicating id (image id=0, metadata id=1, measurement id=2). a vector on the network has a uint16 prefix indicating the length. i have a  single thread that parses the package and stores it in elements of the three pools and places the elements in three corresponding queues.  (there is ever only one producer thread that fills the 3 queues)

three consumer threads consume the data from their respective queues. consumers only read the data.

create an implementation, use modern c++ and libraries like boost (if that seems reasonable)
use c++20 and prefer modern c++ (jthread)
prefer references to pointers (if possible)
