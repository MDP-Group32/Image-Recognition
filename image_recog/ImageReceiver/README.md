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
1. navigate to and run pcRecieveImg.py
```
cd image_recog/ImageReceiver
python3 pcRecieveImg.py
```
2. you should see the following messages
```
'init
init finish
Initiating Image Recognition
Waiting to receive image from rpi'
```
3. images from rpi will be received and saved under images directory
4. to terminate the receiver, press any key in the terminal (temp solution, to be integrated with rpi to note last image transmission)
5. a stitched result containing all received and recognised images will be auto-generated upon terminating the receiver