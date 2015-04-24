#! /usr/bin/env python
# -*- coding: utf8 -*-



import socket
m=raw_input()
while m:
	s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
	s.connect(('localhost',5555))
	s.send(m)
	response=s.recv(1024)
	s.close()
	print response
	m=raw_input()
