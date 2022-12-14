## Keyless Challenge

Your task is to convert a PyTorch model to TFLite and reproduce the same inference scores obtained in python in this C++ project. For this challenge, we will use a computer vision deep learning model for anti-spoofing from the [Silent-Face-Anti-Spoofing github repo](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing).

The above repo contains two deep learning models inside the [resources/anti_spoof_models](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/tree/master/resources/anti_spoof_models) folder, but we are only interested in the `2.7_80x80_MiniFASNetV2.pth`. The other should be ignored.

This C++ project is organized as follows:

- `assets/models` is the folder where you should place your converted TFLite model, namely `2.7_80x80_MiniFASNetV2.tflite`.
- `bin` contains some shell scripts to help you build and test the project. Both macOS and linux are supported (Big Sur 11.2.3 and Ubuntu 20.04 have been tested).
- `build/{OS_NAME}` is the folder in which the project will be built.
- `cmake` contains a CMake helper file required when building for linux.
- `src` contains the source code you will work on. <font color='red'>I added two sub folder to `src` folder.</font>
  - <font color='red'>`src/cpp` contains c++ codes</font>
  - <font color='red'>`src/python` contains Python codes</font>
- `test` contains test code and a fixtures folder where you should place your test images.
- `vendors` contains external dependencies (opencv2 and tflite) required by the project.

To better organize the challenge and provide you with some guidance, we will break the task into smaller ones. 

Whenever you find a question, it should be answered in place. All questions are there for a purpose.

### 1. Building the project

- Prepare your environment with CMake, a C++ 17 compiler and pthread. As far as I can remember, these are the only requirements.
- You can build the project by running `./bin/build_macos.sh` or `./bin/build_linux.sh`, but it will probably fail. What is the cause?
- Edit the `src/CMakeLists.txt` - and the `cmake/generate_ar_input_file.cmake` if you are on linux - to fix it.
- <font color='red'>I modified some paths to adopt new directory structure</font>

### 2. Understanding Silent-Face-Anti-Spoofing

- You should start from the `test.py` file.
- Take a look at the networks' architecture. What is the last layer? 
  - <font color='red'>Linear(in_features=128, out_features=3, bias=False). One Linear layer which has 128 input feature and 3 output feature</font>
- What preprocessing operations an image undergoes before being inputted to the network? 
  - <font color='red'>Checking image width/height ratio. 
  - Converting array to tensors with using ToTensor() function</font>
- Does the input image have channel-first or channel-last format?
- What is the input image colorspace?
  - <font color='red'>3 - RGB Colorspace</font>
- How many classes image can be classified into?
  - <font color='red'>3 classes which are: 2D Fake,Real and 3D Fake <link>https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/issues/46</link> </font>
- What is the index for the genuine (real face) classification score?
  - <font color='red'>Index: 1</link> </font>
- Apart from the anti-spoofing models, does the code use any other ML model?
  - <font color='red'>Yes it use RetinaFace model to detect face</link> </font>

### 3. Testing Silent-Face-Anti-Spoofing

- [X] Modify the `test.py` script to only use the `2.7_80x80_MiniFASNetV2.pth` model.
- [X] Modify the `test.py` script to output only the genuine score.
- [X] Run the `test.py` script for `image_F1.jpg`, `image_F2.jpg` and `image_T1.jpg` images.
  - ```python src/python/test.py --image_name image_F1.jpg```
  - ```python src/python/test.py --image_name image_F2.jpg```
  - ```python src/python/test.py --image_name image_T1.jpg```
- [X] What are the genuine scores for each one of them?
  - - ```python src/python/test.py --image_name image_F1.jpg```
    - Image 'image_F1.jpg' is Fake Face. Score: 0.46.
    - Prediction cost 0.07 s
  - ```python src/python/test.py --image_name image_F2.jpg```
    - Image 'image_F2.jpg' is Fake Face. Score: 0.50.
    - Prediction cost 0.07 s
  - ```python src/python/test.py --image_name image_T1.jpg```
    - Image 'image_T1.jpg' is Real Face. Score: 0.50.
    - Prediction cost 0.07 s
- [X] You will have to reproduce the scores from the previous step later when using TFLite.

### 4. Converting the model to TFLite

- Is it possible to convert the model directly from PyTorch to TFLite?
- If not, which are the intermediates required for this conversion?
- Convert the `2.7_80x80_MiniFASNetV2.pth` model to TFLite and place it inside `assets/models`.

### 5. Generating test images
 
- You should generate the test images for your C++ code and place them inside `test/fixtures`.
- Note from `test/test_main.cpp` that they must be in `bmp` format. Any hunch why we use `bmp` instead of `jpg` here?

### 6. Implementing the C++ code

- Your task here is to complete the `model.cpp` file. There are 3 methods to be implemented:
  - `init`, which should load the TFLite model from disk and make it ready to be used.
  - `inference`, which should load an image from disk and pass it through the network.
  - `convert_image`, which should convert the image to the correct format before sending it to the network.
- The opencv2 lib is available for you to read image files from disk.
  - Does the input image have channel-first or channel-last format?
  - What is the input image colorspace?

### 7. Testing your solution

- Build your solution
- Test your solution by running `bin/test_macos.sh` or `bin/test_linux.sh`
- What are the genuine scores for each test image?