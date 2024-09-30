from Client import PCClient


pc_client = PCClient(ip="192.168.32.1", port=5000)  # Use the RPi's IP
if pc_client.connect():
    # while True:
        # Send a message to RPi
    pc_client.send("Hello from PC!")

    #     # Receive a response from RPi
    # response = pc_client.receive()
    # if response:
    #     print(f"Received from RPi: {response}")

    #     # Add a condition to break the loop
    # if response == "exit":
    #     # break

pc_client.close()
# pc_client.connect()

# message = pc_client.receive()

# print(message)

# pc_client.close()
