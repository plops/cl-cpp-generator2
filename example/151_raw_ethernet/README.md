|       |     |                                                                                                              |
| gen01 | c++ | read network packets individually with recv                                                                  |
| gen02 | c++ | read network packets with rx ring buffer packet mmap (similar to what tcpdump does) TPACKET_v3 (not working) |
| gen03 | c++ | packet mmap TPACKET_v2, print loss statistics                                                                |
| gen04 | c++ | 03 but unit-testable                                                                                         |
|       |     |                                                                                                              |


https://gist.github.com/austinmarton/2862515


- reading network with recv can be slow. packet mmap with an rx ring buffer has more throughput

http://paul.chavent.free.fr/packet_mmap.html explains how to use tpacket_v2


# timestamps

https://github.com/jclark/rpi-cm4-ptp-guide

The PTP support involves the Ethernet PHY having its own clock, called the PTP hardware clock (PHC), and being able to use this clock to timestamp incoming and outgoing network packets. This enables the CM4 to make use of PTP, but it is not by itself particularly exciting: similar functionality is available on many NICs. The exciting part is that the CM4 provides a pin that allows the PHC to be synchronized with an external pulse per second (PPS) signal

- my laptop has no hardware clock
``` 
 $ ethtool -T wlan0
Time stamping parameters for wlan0:
Capabilities:
        software-receive
        software-system-clock
PTP Hardware Clock: none
Hardware Transmit Timestamp Modes: none
Hardware Receive Filter Modes: none
```

- hetzner has no hardware clock:
```
$ ethtool -T ens3
Time stamping parameters for ens3:
Capabilities:
        software-transmit
        software-receive
        software-system-clock
PTP Hardware Clock: none
Hardware Transmit Timestamp Modes: none
Hardware Receive Filter Modes: none

```

- azure has no hardware clock:
```
# ethtool -T eth0
Time stamping parameters for eth0:
Capabilities:
        software-transmit
        software-receive
        software-system-clock
PTP Hardware Clock: none
Hardware Transmit Timestamp Modes: none
Hardware Receive Filter Modes: none

```

- hp desktop has hardware clock timestamping but no precision time protocol:
```
$ ethtool -T enp65s0
Time stamping parameters for enp65s0:
Capabilities:
        hardware-transmit
        software-transmit
        hardware-receive
        software-receive
        software-system-clock
        hardware-raw-clock
PTP Hardware Clock: 0
Hardware Transmit Timestamp Modes:
        off
        on
Hardware Receive Filter Modes:
        none
        all

```

- this NIC uses the igc kernel module for this PCIe device: Intel
  Corporation Ethernet Controller I226-LM (rev 04)
- according to the intel specs this controller should support PTP:
```
IEEE 1588, also knows as the Precision Time Protocol (PTP) is a protocol used to synchronize clocks throughout a computer network. On a local area network it achieves clock accuracy in the sub-microsecond range, making it suitable for measurement and control systems.
```
