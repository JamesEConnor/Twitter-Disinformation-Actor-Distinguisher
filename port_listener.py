# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 19:30:03 2020

@author: James
"""


import socket;
from twitter_actors import get_actor;

from datetime import datetime;


host = "localhost";
port = 8080;
backlog = 5;

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM);
sock.bind((host, port));
sock.listen(backlog);




# Get transmitted block of bytes
def _get_block(s, count):
    # If count is invalid, return empty string.
    if count <= 0:
        return '';
    
    # Create the byte buffer
    buffer = '';
    while len(buffer) < count:
        # Receive the bytes
        buffer2 = s.recv(count - len(buffer));
        
        # If no bytes received, either the connection has been ended or the count was invalid.
        if not buffer2:
            if buffer:
                raise RuntimeError("underflow");
            else:
                return '';
            
        buffer += buffer2.decode("utf-8");
            
    # Return the result
    return buffer;

def _send_block(s, data):
    # Transmit each byte on the socket
    while data:
        data = data[s.send(data):]

def _get_count(s):
    # Create the temporary buffer
    buf = '';
    while True:
        # Receive one byte
        c = s.recv(1);
        if not c:
            # If no bytes received, either the connection has been ended or the count was invalid.
            if buf:
                raise RuntimeError("underflow");
            else:
                return -1;
            
        # If character is '|', the full count of the message length has been found.
        if c.decode("utf-8") == '|':
            return int(buf);
        # Otherwise, add and continue
        else:
            buf += c.decode("utf-8");
            log(buf);
            
    return -1;

def get_msg(s):
    return _get_block(s, _get_count(s));

def send_msg(s, data):
    _send_block(s, data.encode("utf-8"));

def close_socket():
    if sock:
        sock.shutdown(socket.SHUT_RDWR);
        sock.close();
    
def log(msg):
    print("[" + datetime.now().strftime("%Y/%m/%d %H:%M:%S") + "]\t" + msg);
    
log("Loaded algorithms.");
while True:
    try:
        conn, addr = sock.accept();
        log("Connection with " + str(addr) + " accepted.");
        
        msg_len = _get_count(conn);
        log("Received message length: " + str(msg_len));
        
        msg_data = _get_block(conn, msg_len); #The handle
        log("Received message: " + str(msg_data));
        
        result = get_actor(msg_data);
        log("Got result: " + result);
        
        send_msg(conn, result);
        log("Wrote to connection.");

        conn.shutdown(socket.SHUT_RDWR);
        conn.close();
    except KeyboardInterrupt:
        if conn:
            conn.shutdown(socket.SHUT_RDWR);
            conn.close();
            
        close_socket();
        break;