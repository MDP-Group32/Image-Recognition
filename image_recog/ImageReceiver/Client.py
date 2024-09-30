import socket

class PCClient:
    def __init__(self, ip, port):
        self.server_ip = ip
        self.server_port = port
        self.client_socket = None

    def connect(self):
        # Create a TCP/IP socket
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            # Connect to the Raspberry Pi server
            self.client_socket.connect((self.server_ip, self.server_port))
            print(f"Connected to Raspberry Pi at {self.server_ip}:{self.server_port}")
        except socket.error as e:
            print(f"Failed to connect: {e}")
            self.client_socket.close()
            return False
        return True

    def send(self, message):
        try:
            self.client_socket.sendall(message.encode('utf-8'))
            print(f"Sent: {message}")
        except socket.error as e:
            print(f"Failed to send message: {e}")

    def receive(self):
        try:
            data = self.client_socket.recv(1024)
            return data.decode('utf-8')
        except socket.error as e:
            print(f"Error receiving message: {e}")
            return None

    def close(self):
        self.client_socket.close()
        print("Connection closed.")