#! /usr/bin/env python
# -*- coding: utf8 -*-

import threading
import SocketServer

class ThreadedEchoRequestHandler(SocketServer.BaseRequestHandler):

    def handle(self):
        # Echo the back to the client
        data = self.request.recv(1024)
        cur_thread = threading.currentThread()
        response = '%s: %s' % (cur_thread.getName(), data)
        self.request.send(response)
        return

class ThreadedEchoServer(SocketServer.ThreadingMixIn, SocketServer.TCPServer):
    pass

address=('localhost',5555)
server=ThreadedEchoServer(address,ThreadedEchoRequestHandler)
print server.server_address
server.serve_forever()