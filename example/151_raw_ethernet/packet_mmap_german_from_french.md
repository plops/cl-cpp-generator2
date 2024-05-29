This article was translated from french to german using gemini 1.5 pro
The original source is: http://paul.chavent.free.fr/packet_mmap.html

## Programmierung von Layer-2-Netzwerken unter Linux

Dieser Artikel beschreibt die Programmierschnittstelle für den direkten Zugriff auf Layer 2 des OSI-Modells. Wir werden insbesondere sehen, wie man die Leistung durch einen Zero-Copy-Mechanismus optimieren kann, der vom Linux-Kernel bereitgestellt wird.

**Dieser Artikel befindet sich noch in Bearbeitung.**

### Einleitung

Wenn man über Netzwerkprogrammierung unter Linux spricht, denkt man als Erstes an "Sockets". Sockets stellen eine einheitliche Schnittstelle zwischen Ihren Programmen und der Netzwerkschicht des Kernels dar. Die unterstützten Protokollfamilien sind auf der Manpage von `socket(2)` aufgeführt. Sie können beispielsweise auf UNIX-Sockets für die lokale Kommunikation auf dem Rechner (Familie `PF_UNIX`), auf Internet-Sockets für die Kommunikation zwischen Rechnern in einem IP-Netzwerk (Familie `PF_INET/PF_INET6`) usw. zugreifen.

Sockets ermöglichen auch den Zugriff auf die unterste Ebene des Netzwerkstacks und das Senden von Daten direkt an die Netzwerkkarte (Familie `AF_PACKET`). Aber wer könnte das jemals brauchen?

Ein solcher Zugriff wird bereits von Netzwerküberwachungstools (Wireshark) verwendet, um den gesamten Datenverkehr aufzuzeichnen, der über Ihre Netzwerkkarte läuft. Er kann auch verwendet werden, um bestimmte Pakete zu erzeugen (z. B. solche, die absichtlich Fehler enthalten), um ein Gerät zu testen oder Fehler zu diagnostizieren. Schließlich ermöglicht diese Schnittstelle das Schreiben eigener Netzwerkprotokolle im Benutzerbereich, und das ganz einfach mit Ihren gewohnten Tools.

Dieser Artikel zeigt Ihnen, wie Sie ein Programm schreiben, das eine einfache Kommunikation ermöglicht.

Wir werden sehen, wie man den Frame direkt im Puffer vorbereitet, der an das Gerät gesendet wird.

### ZeroCopy?

Die Dokumentation über Netzwerkprogrammierung ist umfangreich. Für den uns interessierenden Punkt empfehle ich die Manpages von `socket(7)` und `packet(7)` zu lesen. In den Hinweisen von `packet(7)` finden Sie eine Warnung bezüglich der Portabilität.

Im Hinblick auf die Portabilität wird empfohlen, die `PF_PACKET`-Funktionalitäten über die `pcap(3)`-Schnittstelle zu verwenden, obwohl diese nur eine Teilmenge der Möglichkeiten von `PF_PACKET` abdeckt.

Der Fokus dieses Artikels liegt jedoch gerade darauf, eine recht interessante (aber wenig dokumentierte) Möglichkeit von `PF_PACKET` unter Linux zu erkunden: Zero-Copy.

Wie der Name schon sagt, geht es bei Zero-Copy darum, das Kopieren von Daten von einem Puffer in einen anderen, vom Benutzerbereich in den Kernel, durch alle Schichten hindurch bis zur Ausgabe an das Gerät zu vermeiden.

Die Schnittstellen, mit denen wir in unseren Programmen normalerweise arbeiten, sind selten für Zero-Copy geeignet. Um beispielsweise Daten zu senden, erfolgt der Vorgang in drei Schritten:

1. Abrufen eines Zeigers auf einen zugewiesenen Speicherbereich, der einen Frame aufnehmen kann
2. Füllen des Frames
3. Senden des Frames über einen Systemaufruf wie `send`

Für Zero-Copy müssen wir nur den ersten Schritt ändern, um den Speicherbereich anzufordern, den der Kernel für die Übertragung der Netzwerkframes verwendet.

Die Schnittstelle, die dies ermöglicht, heißt Packet MMAP.

### Packet MMAP

Die Dokumentation zu MMAP ist in den Kernel-Quellen [1] enthalten. Außerdem gibt es zwei Referenz-Tutorials für das Senden [2] und Empfangen [3]. Mir scheint jedoch, dass die Erklärungen recht knapp sind und diese kleine Zusammenfassung rechtfertigen.

Die Packet-MMAP-Schnittstelle bietet einen Ringpuffer, der im Benutzerbereich zugänglich ist und das Senden oder Empfangen von Frames ermöglicht. Der Autor dieser Funktionalität hebt den Vorteil hervor, dass man mehrere Pakete mit einem einzigen Systemaufruf senden kann, aber mir scheint, dass `sendmsg` dies bereits ermöglichte. Meiner Meinung nach liegt der Vorteil vor allem in der Vermeidung von Kopien und der Reduzierung von Systemaufrufen zum Abrufen von Informationen über das Paket (insbesondere der Zeitstempel).

**FIXME**: Diesen Absatz überarbeiten: Die API von Packet MMAP befindet sich in `include/uapi/linux/if_packet.h`.

### Konfiguration

Die Option Packet MMAP hat sich in drei Versionen weiterentwickelt. In diesem Artikel wird Version 2 verwendet.

Die Aktivierung des Ringpuffers erfolgt mit einem Aufruf von `setsockopt_`:

Für das Empfangen:
```c
setsockopt(fd, SOL_PACKET, PACKET_RX_RING, (void *) &req, sizeof(req))
```

Für das Senden:
```c
setsockopt(fd, SOL_PACKET, PACKET_TX_RING, (void *) &req, sizeof(req))
```

Das Argument `req` enthält die Beschreibung der "Geometrie" des Ringpuffers. Es handelt sich um eine Struktur vom Typ `struct tpacket_req`, die in `/usr/include/uapi/linux/if_packet.h` beschrieben ist:

```c
struct tpacket_req {
	unsigned int	tp_block_size;	/* Mindestgröße eines zusammenhängenden Blocks */
	unsigned int	tp_block_nr;	/* Anzahl der Blöcke */
	unsigned int	tp_frame_size;	/* Größe eines Frames */
	unsigned int	tp_frame_nr;	/* Gesamtzahl der Frames */
};
```

Jeder Ringpuffer besteht also aus Blöcken (einem zusammenhängenden Speicherbereich), und jeder Block kann mehrere "Frames" aufnehmen.

#### Geometrie der Ringpuffer

Die Konfiguration der Ringpuffer unterliegt den folgenden Einschränkungen:

* Die maximale Anzahl der Blöcke ist festgelegt und hängt von Ihrer Architektur ab (in der Größenordnung von mehreren Tausend).
* Die Größe eines Blocks muss eine Potenz von 2 der Größe einer Seite sein.
* Ein Block enthält eine ganze Zahl von Frames.
* Ein Frame kann eine beliebige Größe haben, vorausgesetzt, die Anzahl der Blöcke stimmt mit der Anzahl der Frames überein:
   `tp_block_size / tp_frame_size * tp_block_nr = tp_frame_nr`

Detaillierte Erläuterungen zu den Beschränkungen der Anzahl und Größe von Blöcken finden Sie in der Dokumentation zu Packet MMAP [1].

Schließlich enthalten die Frames die Pakete und ihre Metadaten (Größe, Zeitstempel, ...). Jeder Frame beginnt mit einem Header (Metadaten), der durch die Struktur `tpacket_hdr` definiert ist. Für Version 2 sieht die Struktur wie folgt aus:

```c
struct tpacket2_hdr {
	__u32		tp_status;
	__u32		tp_len;
	__u32		tp_snaplen;
	__u16		tp_mac;
	__u16		tp_net;
	__u32		tp_sec;
	__u32		tp_nsec;
	__u16		tp_vlan_tci;
	__u16		tp_padding;
};
```

Achtung: Die Struktur der "Frames" ist asymmetrisch und die Daten befinden sich an einem anderen Offset, je nachdem, ob es sich um das Senden oder Empfangen handelt.

Beim Empfangen wird der Offset vom Beginn des Frames an durch das Feld `tp_net` oder `tp_mac` angegeben, je nachdem, ob der Socket-Typ `SOCK_DGRAM` oder `SOCK_RAW` ist.

#### Struktur eines RX-Frames:

```
Start (ausgerichtet auf TPACKET_ALIGNMENT=16)   TPACKET_ALIGNMENT=16                                   TPACKET_ALIGNMENT=16
v                                         v                                                      v
|                                         |                             | tp_mac                 |tp_net
|  struct tpacket_hdr  ... pad            | struct sockaddr_ll ... gap  | min(16, maclen)        |
|<--------------------------------------------------------------------->|<---------------------->|<----... 
                               tp_hdrlen = TPACKET2_HDRLEN                   if SOCK_RAW           Benutzerdaten (Nutzdaten)
```

Beim Senden ist der Offset konstant und definiert durch: `TPACKET2_HDRLEN - sizeof(struct sockaddr_ll)`.

#### Struktur eines TX-Frames:

```
Start (ausgerichtet auf TPACKET_ALIGNMENT=16)   TPACKET_ALIGNMENT=16
v                                         v
|                                         |
|  struct tpacket_hdr  ... pad            | struct sockaddr_ll ... gap
|<--------------------------------------------------------------------->| 
                               tp_hdrlen = TPACKET2_HDRLEN
                                          |<---- ... 
                                              Benutzerdaten
```

Ab Kernel-Version 3.8 kann der Benutzer den Offset des gesendeten Pakets über die Option `PACKET_TX_HAS_OFF` festlegen.

### Mapping des Ringpuffers in den Benutzerbereich

Der Benutzer erhält einen Zeiger auf einen zusammenhängenden Speicherbereich, der den/die Ringpuffer repräsentiert, indem er `mmap` verwendet.

Unabhängig davon, ob Sie einen einzigen Ringpuffer (Senden oder Empfangen) oder zwei Ringpuffer (einen von jedem) haben, muss nur ein Aufruf von `mmap` erfolgen. Die Ringpuffer folgen in der Reihenfolge RX/TX aufeinander.

### Empfangsvorgang

Beim Anlegen des Empfangsringpuffers hat der Kernel alle Header der Frames initialisiert und insbesondere den Wert des Feldes `tp_status` auf `TP_STATUS_KERNEL` gesetzt.

Das Feld `tp_status` ermöglicht es dem Kernel, dem Benutzer die Verfügbarkeit eines Frames mit dem Wert `TP_STATUS_USER` mitzuteilen. Wenn der Benutzer mit dem Lesen des Pakets fertig ist, gibt er den Frame mit dem Wert `TP_STATUS_KERNEL` an den Kernel zurück.

Der Benutzer muss den Statuswechsel der Frames nicht überwachen, sondern kann sich einfach mit der Funktion `poll` (oder einer entsprechenden Funktion) in den Wartezustand versetzen.

### Sendevorgang

Beim Anlegen des Senderingpuffers hat der Kernel alle Header der Frames initialisiert und insbesondere den Wert des Feldes `tp_status` auf `TP_STATUS_AVAILABLE` gesetzt.

Das Feld `tp_status` ermöglicht es dem Benutzer, dem Kernel die Verfügbarkeit eines Frames mit dem Wert `TP_STATUS_SEND_REQUEST` mitzuteilen. Wenn der Kernel mit dem Senden des Pakets fertig ist, gibt er den Frame mit dem Wert `TP_STATUS_AVAILABLE` an den Benutzer zurück.

Der Benutzer muss die Länge der Daten im Header angeben und dann `send` aufrufen, um dem Kernel mitzuteilen, dass Daten zum Senden bereit sind.

Es können mehrere Frames gleichzeitig gesendet werden.

### Beispiel

Hier ist ein einfaches Beispiel zum Senden und Empfangen von Frames.

#### Öffnen und Parametrieren des Sockets

Das Öffnen des Sockets erfolgt mit einem Aufruf von `socket`:

```c
int type;
uint16_t protocol;
fd = socket(AF_PACKET, socket_type, htons(protocol));
```

Die Variable `type` nimmt den Wert `SOCK_RAW` an, wenn der Ethernet-Header angegeben werden soll, oder `SOCK_DGRAM`, wenn nur die Nutzdaten des Frames bereitgestellt werden (siehe Struktur eines Frames [4]).

Die Variable `protocol` ist die Kennung des Protokolls und kann den Wert `0X88b5` oder `0X88b6` annehmen, die für lokale Experimente reserviert sind (http://standards.ieee.org/develop/regauth/ethertype/eth.txt).

Der Socket wird mit einem Aufruf von `bind` an eine Hardware-Schnittstelle gebunden:

```c
struct sockaddr_ll local_addr;
bind(fd, &local_addr, sizeof(local_addr));
```

Das Feld `local_addr.sll_ifindex` nimmt den Wert des Index der Netzwerkschnittstelle an (den man mit dem `ioctl SIOCGIFINDEX` abrufen kann, siehe `man netdevices`).

Das Feld `local_addr.sll_halen` ist die Länge einer MAC-Adresse.

Das Feld `local_addr.sll_addr` ist die MAC-Adresse der Netzwerkschnittstelle.

Geben Sie die Version der benötigten Packet-MMAP-Schnittstelle an.

```c
int version = TPACKET_V2;
setsockopt(fd, SOL_PACKET, PACKET_VERSION, &version, sizeof(version))
```

Sie können festlegen, dass der Offset der gesendeten Paketdaten angegeben werden soll (ab Kernel-Version 3.X).

```c
int tx_has_off = 1;
setsockopt(fd, SOL_PACKET,  PACKET_TX_HAS_OFF, &tx_has_off, sizeof(tx_has_off));
```

Bereiten Sie die Geometrie der Ringpuffer vor.

```c
struct tpacket_req rx_paquet_req;
rx_paquet_req.tp_block_size = sysconf(_SC_PAGESIZE) << 1; 
rx_paquet_req.tp_block_nr = 2;
rx_paquet_req.tp_frame_size = next_power_of_two(mtu + 128);
rx_paquet_req.tp_frame_nr = (rx_paquet_req.tp_block_size / rx_paquet_req.tp_frame_size) * rx_paquet_req.tp_block_nr;
```

In diesem Beispiel fordern wir zwei Blöcke an, die jeweils die Größe von zwei Seiten haben. Die Frames müssen eine Mindestgröße haben, die die größten Frames (MTU) mit ihrem Header (falls `SOCK_RAW`) plus dem Header der Packet-MMAP-Schnittstelle (ca. 80 Byte) aufnehmen können.

Sie können eine andere Geometrie für den Sende- und Empfangspuffer angeben.

Instanziieren Sie die Ringpuffer.

```c
setsockopt(fd, SOL_PACKET, PACKET_RX_RING, &rx_paquet_req, sizeof(rx_paquet_req))
setsockopt(fd, SOL_PACKET, PACKET_TX_RING, &tx_paquet_req, sizeof(tx_paquet_req))
```

Mappen Sie die Ringpuffer in den Benutzerbereich.

```c
int mmap_size = 
    rx_paquet_req.tp_block_size * rx_paquet_req.tp_block_nr +
    tx_paquet_req.tp_block_size * tx_paquet_req.tp_block_nr ;
mmap_base = mmap(0, mmap_size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
```

Diese Operation gibt einen Zeiger auf einen zusammenhängenden Speicherbereich zurück, wobei der Empfangspuffer dem Sendepuffer vorangestellt ist.

Wir können nun jeden Puffer aus Sicht des Benutzers beschreiben:

```c
rx_buffer_size = rx_paquet_req.tp_block_size * rx_paquet_req.tp_block_nr;
rx_buffer_addr = mmap_base;
rx_buffer_idx  = 0;
rx_buffer_cnt  = rx_paquet_req.tp_block_size * rx_paquet_req.tp_block_nr / rx_paquet_req.tp_frame_size;
tx_buffer_size = tx_paquet_req.tp_block_size * tx_paquet_req.tp_block_nr;
tx_buffer_addr = mmap_base + rx_buffer_size;
tx_buffer_idx  = 0;
tx_buffer_cnt  = tx_paquet_req.tp_block_size * tx_paquet_req.tp_block_nr / tx_paquet_req.tp_frame_size;
```

Die Schnittstelle ist nun einsatzbereit.

#### Empfangsvorgang

Der Empfang eines Pakets kann nicht mehr mit einem einzigen Aufruf von `recv` erfolgen, wie Sie es vielleicht gewohnt sind. Der Empfang erfolgt in zwei Schritten.

##### Anfordern eines Speicherbereichs

Man muss einen Zeiger auf den Bereich erhalten, in dem die empfangenen Daten gespeichert wurden. Dazu suchen wir nach einem Frame, den der Kernel als bereit zum Lesen durch den Benutzer markiert hat. Da die Frames aufeinanderfolgen, sind sie indizierbar und wir können einen Zeiger auf ihren Header abrufen:

```c
void * base = rx_buffer_addr + rx_buffer_idx * rx_packet_req.tp_frame_size;
volatile struct tpacket2_hdr * header = (struct tpacket2_hdr *)base;
```

Der Header eines Frames enthält ein Feld `tp_status`, das Informationen über seinen Zustand liefert: Er ist verfügbar, wenn sein Zustand ungleich `TP_STATUS_KERNEL` ist.

Wir können also den Ringpuffer nach einem verfügbaren Frame durchsuchen. Ist kein Frame verfügbar, muss man sich mit einem Aufruf von `poll` in den Wartezustand für ein Signal vom Kernel versetzen.

```c
struct pollfd;
pollfd.fd = fd;
pollfd.events = POLLIN|POLLRDNORM|POLLERR;
pollfd.revents = 0;
ppoll(&pollfd, 1, NULL, NULL);
```

Wenn `poll` mit gesetztem Bit `POLLIN` von `pollfd.revents` zurückkehrt, ist ein Frame zum Empfang verfügbar.

Die Informationen, die uns im Header des Frames zur Verfügung stehen, sind der Offset der Daten, die erfasste Größe und der Zeitpunkt des Eintreffens der Daten:

```c
void * data = base + header->tp_net;
unsigned data_len = header->tp_snaplen;
struct timespec ts;
ts.sec  = header->tp_sec;
ts.nsec = header->tp_nsec;
```

##### Nutzung des Speicherbereichs

Die Daten können "an Ort und Stelle" verarbeitet werden, ohne dass sie in einen lokalen Puffer kopiert werden müssen. Die Adresse ist an `TPACKET_ALIGNMENT` (16 Byte) ausgerichtet und sollte es ermöglichen, jede beliebige Struktur darauf abzubilden.

##### Zurückgeben des Speicherbereichs

Nach der Verarbeitung muss der Frame an den Kernel "zurückgegeben" werden, damit dieser die nächsten eingehenden Frames darin speichern kann.

Dies geschieht einfach, indem man seinen Status auf `TP_STATUS_KERNEL` setzt.

```c
header->tp_status = TP_STATUS_KERNEL;
```

##### Hinweise

Man beachte, dass, wenn die Schnittstelle einen konstanten Datenstrom empfängt, keine Systemaufrufe für den Empfang erforderlich sind. Der einzige Blockierungspunkt ist der Aufruf von `poll`, der bei Engpässen auf der Schnittstelle erfolgt.

#### Sendevorgang

Das Senden eines Pakets kann nicht mehr mit einem einzigen Aufruf von `send` erfolgen, wie Sie es vielleicht gewohnt sind. Das Senden erfolgt in zwei Schritten.

##### Anfordern eines Speicherbereichs

Man muss einen Zeiger auf den Bereich erhalten, in dem man die zu sendenden Daten speichern kann. Dazu muss man einen freien Frame finden. Da die Frames aufeinanderfolgen, sind sie indizierbar und wir können einen Zeiger auf ihren Header auf die gleiche Weise wie beim Empfang abrufen.

Der Header eines Frames enthält ein Feld `tp_status`, das Informationen über seinen Zustand liefert: Er ist verfügbar, wenn sein Zustand ungleich `TP_STATUS_AVAILABLE` ist.

Der für den Benutzer verfügbare Speicherbereich befindet sich dann am Offset:

```c
tx_buffer_payload_offset = TPACKET2_HDRLEN - sizeof(struct sockaddr_ll);
```

Wenn Sockets vom Typ `SOCK_RAW` verwendet werden, sind die Nutzdaten um 12 Byte versetzt:

```c
tx_buffer_payload_offset += sizeof(ether_header_t);
```

Wenn es möglich ist, den Offset der gesendeten Daten anzugeben (`PACKET_TX_HAS_OFF`):

```c
tx_buffer_payload_offset = TPACKET_ALIGN(tx_buffer_payload_offset);
```

##### Nutzung des Speicherbereichs

Die Daten können "an Ort und Stelle" verarbeitet werden, ohne dass sie aus einem lokalen Puffer kopiert werden müssen. Die Adresse kann an `TPACKET_ALIGNMENT` (16 Byte) ausgerichtet werden und sollte es ermöglichen, jede beliebige Struktur darauf abzubilden.

##### Übergeben des Speicherbereichs

Sobald der Frame gefüllt ist, muss er an den Kernel "übergeben" werden, damit dieser ihn an die Schnittstelle senden kann.

Dies geschieht, indem man seinen Status auf `TP_STATUS_SEND_REQUEST` setzt und dann `sendto` aufruft.

```c
header->tp_status = TP_STATUS_SEND_REQUEST;
sendto(itf->sock_fd, NULL, 0, 0, (const struct sockaddr *)&remote_addr, remote_addr));
```

##### Hinweise

Man beachte, dass man das Senden mehrerer Frames in einem einzigen Systemaufruf zusammenfassen kann. Außerdem erfordert der Systemaufruf kein Kopieren von Daten.

### Ressourcen

* Code des Artikels.
* Programme, die die Schnittstelle verwenden: libpcap, netsniff-ng, ...
* [1] packet_mmap.txt
* [2] http://wiki.ipxwarzone.com/index.php5?title=Linux_packet_mmap#Example
* [3] http://www.scaramanga.co.uk/code-fu/lincap.c
* [4] Struktur eines Ethernet-Frames

