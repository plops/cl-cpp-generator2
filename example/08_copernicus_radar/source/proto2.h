int main ()  ;  
size_t get_filesize (const char* filename)  ;  
void destroy_mmap ()  ;  
void init_mmap (const char* filename)  ;  
void destroy_collect_packet_headers ()  ;  
void init_collect_packet_headers ()  ;  
void init_process_packet_headers ()  ;  
void init_sequential_bit_function (sequential_bit_t* seq_state, size_t byte_pos)  ;  
void consume_padding_bits (sequential_bit_t* s)  ;  
inline int get_bit_rate_code (sequential_bit_t* s)  ;  
inline int decode_huffman_brc0 (sequential_bit_t* s)  ;  
inline int decode_huffman_brc1 (sequential_bit_t* s)  ;  
inline int decode_huffman_brc2 (sequential_bit_t* s)  ;  
inline int decode_huffman_brc3 (sequential_bit_t* s)  ;  
inline int decode_huffman_brc4 (sequential_bit_t* s)  ;  
int init_decode_packet (int packet_idx, std::complex<float>* output)  ;  
inline int get_data_type_a_or_b (sequential_bit_t* s)  ;  
int init_decode_packet_type_a_or_b (int packet_idx, std::complex<float>* output)  ;  
void init_sub_commutated_data_decoder ()  ;  
bool feed_sub_commutated_data_decoder (uint16_t word, int idx, int space_packet_count)  ;  
inline int get_baq3_code (sequential_bit_t* s)  ;  
inline int get_baq4_code (sequential_bit_t* s)  ;  
inline int get_baq5_code (sequential_bit_t* s)  ;  
int init_decode_type_c_packet_baq3 (int packet_idx, std::complex<float>* output)  ;  
int init_decode_type_c_packet_baq4 (int packet_idx, std::complex<float>* output)  ;  
int init_decode_type_c_packet_baq5 (int packet_idx, std::complex<float>* output)  ;  
