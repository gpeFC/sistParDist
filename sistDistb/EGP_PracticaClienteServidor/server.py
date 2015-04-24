# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 18:29:55 2015

@author: mag (emanuelgp)
"""

import socket
import os

puerto = 4545;

miSocket = socket.socket( socket.AF_INET, socket.SOCK_STREAM )

miSocket.bind( ( socket.gethostname(), puerto ) )

miSocket.listen( 1 )

channel, details = miSocket.accept()

while True:
    datos, serv = channel.recvfrom(100)   
    print datos
    if datos.upper() == "EOF":
    	channel.close()
    	os.system('clear')
    	break
    data = raw_input(">> ")
    channel.send(data)