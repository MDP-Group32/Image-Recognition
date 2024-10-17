from pc.task1 import ImageReceiver as task1
from pc.task2 import ImageReceiver as task2

image_receiver = task1()
# image_receiver = task2()
image_receiver.receive_image()

# Troubleshoot: 
# 1. Ensure connection to RPI
# 2. Check RPI's PC IP Address (PC to call ifconfig)
