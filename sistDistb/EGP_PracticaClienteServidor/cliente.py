# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 18:28:33 2015

@author: mag (emanuelgp)
"""

import socket
import os

miSocket = socket.socket( socket.AF_INET, socket.SOCK_STREAM )

miSocket.connect( (socket.gethostname(), 4545 ) )

while True:
	msj = raw_input(">> ")
	miSocket.send(msj)
	if msj.upper() == "EOF":
		miSocket.close()
		os.system('clear')
		break
	data, server = miSocket.recvfrom( 100 )
	print data