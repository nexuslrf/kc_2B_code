import bluetooth

bd_addr = "00:10:05:25:00:01"
# bd_addr = "98:D3:36:80:CE:ED"
port = 1

sock=bluetooth.BluetoothSocket( bluetooth.RFCOMM )
sock.connect((bd_addr, port))
sock.send("T")
print sock.recv(1024)
sock.send("T")
print sock.recv(1024)
ss=""
while ss!= 'q':
    ss = raw_input()
    sock.send(ss)
    data=sock.recv(1024)
    print data
    if data[0]=="F":
        print "TRUE"

sock.close()
