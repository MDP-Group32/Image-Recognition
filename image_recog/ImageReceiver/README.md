# Image Recognition Receiver

## Getting Started

### Prerequisites
1. Python should be installed, you can check via
```
python3 --version
```
or
```
python --version
```

### MacOS
1. Create a virtual environment (if you haven't already)
```
python -m venv venv
```
or 
```
python3 -m venv venv
```
2. Enter the virtual environment
```
source venv/bin/activate
```
3. Verify that you are in the correct virtual env using
```
which python
```
- It should show a path like `Image-Recognition/venv/bin/python`
4. Install dependencies:
```
pip install -r requirements.txt
```

### Windows
1. Create a virtual environment (if you haven't already)
```
python -m venv venv
```
or 
```
python3 -m venv venv
```
2. Enter the virtual environment
```
venv\Scripts\activate
```
3. Verify that you are in the correct virtual env using
- Your path should be prefixed with `(venv)`
4. Install dependencies:
```
pip install -r requirements.txt
```
NOTE: If running on windows comment out: `uvloop==0.20.0` dependency because it is currently not supported on Windows.

### Running the image recognition receiver
1. selecting Task 1 vs Task 2 image recognition
```
open pcRecieveImg.py

- for task 1 (comment out task2()) - 
image_receiver = task1()
# image_receiver = task2()

- for task 2 (comment out task1()) - 
# image_receiver = task1()
image_receiver = task2()
```
2. run pcRecieveImg.py
```
cd image_recog/ImageReceiver
python3 pcRecieveImg.py
```
3. you should see the following messages
```
'init
init finish
Initiating Image Recognition
Waiting to receive image from rpi'
```
4. images from rpi will be received and saved under images directory
5. a stitched result containing all received and recognised images will be auto-generated upon receiving all expected images from rpi 