|       |     |                                                                                                              |
| gen01 | c++ | read network packets individually with recv                                                                  |
| gen02 | c++ | read network packets with rx ring buffer packet mmap (similar to what tcpdump does) TPACKET_v3 (not working) |
| gen03 | c++ | packet mmap TPACKET_v2                                                                                       |
|       |     |                                                                                                              |


https://gist.github.com/austinmarton/2862515


- reading network with recv can be slow. packet mmap with an rx ring buffer has more throughput

http://paul.chavent.free.fr/packet_mmap.html explains how to use tpacket_v2
