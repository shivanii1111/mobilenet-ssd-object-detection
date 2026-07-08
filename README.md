# MobileNet-SSD C++ Object Detection Pipeline with Custom NMS

A from-scratch C++ inference pipeline that runs MobileNet-SSD (Caffe) through
OpenCV's DNN module, with a hand-written IoU-based Non-Maximum Suppression
(no `cv::dnn::NMSBoxes`), bounding-box visualization, and per-frame latency
measurement.

## What it demonstrates
- Loading and running a pretrained CNN detector with OpenCV DNN (no PyTorch/TF
  runtime needed)
- Manual image preprocessing into a network "blob" (resize, mean subtraction,
  scaling)
- Parsing raw SSD output tensors
- Implementing NMS yourself (sort by confidence, suppress by IoU) instead of
  relying on a library call
- Measuring and reporting inference latency / FPS
- Clean, modular C++ (CMake build, no notebook hacks)

## Project layout
```
mobilenet-ssd-cpp/
├── CMakeLists.txt
├── download_model.sh        # fetches the pretrained Caffe weights
├── models/                  # deploy.prototxt + .caffemodel go here
└── src/
    └── main.cpp              # the whole pipeline
```

## Step-by-step setup

### 1. Install dependencies
You need a C++17 compiler, CMake, and OpenCV (with the `dnn` module, which is
included in standard OpenCV >= 4.x builds).

Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libopencv-dev
```

macOS (Homebrew):
```bash
brew install cmake opencv
```

### 2. Download the pretrained model
MobileNet-SSD is distributed as a Caffe model (`deploy.prototxt` +
`mobilenet_iter_73000.caffemodel`), trained on the 20 PASCAL VOC classes.

```bash
cd mobilenet-ssd-cpp
chmod +x download_model.sh
./download_model.sh
```

This creates:
```
models/deploy.prototxt
models/mobilenet_iter_73000.caffemodel
```

If the download script's URLs ever go stale, search GitHub for
"chuanqi305/MobileNet-SSD" — that repo is the canonical source for these two
files and mirrors exist under the same filenames.

### 3. Build
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j
```
This produces the `mobilenet_ssd_detect` binary in `build/`.

### 4. Run

Single image:
```bash
./mobilenet_ssd_detect --image ../test.jpg
```
This writes an annotated `output.jpg` and pops up a window with the result.

Video file:
```bash
./mobilenet_ssd_detect --video ../test.mp4
```

Webcam:
```bash
./mobilenet_ssd_detect --camera 0
```

Press `ESC` to quit the video/camera preview.

## How the pipeline works (matches the code in `src/main.cpp`)

1. **Preprocess** — `cv::dnn::blobFromImage` resizes the frame to 300x300,
   subtracts a mean of 127.5 per channel, and scales pixel values by
   `1/127.5`, matching how the network was trained.
2. **Inference** — `net.forward()` runs the forward pass and returns a
   `[1, 1, N, 7]` tensor: one row per candidate detection
   `(imageId, classId, confidence, xmin, ymin, xmax, ymax)`, all box
   coordinates normalized to `[0, 1]`.
3. **Parse** — rows below `confThreshold` (default `0.4`) are dropped; the
   rest are converted from normalized coordinates to pixel-space `cv::Rect`s.
4. **Custom NMS** (`customNMS` in `main.cpp`) —
   - Sort all surviving detections by confidence, descending.
   - Walk the sorted list; keep the first unsuppressed box.
   - Suppress every remaining box **of the same class** whose IoU with the
     kept box exceeds `iouThreshold` (default `0.45`).
   - Repeat until every detection is either kept or suppressed.
   - This is the same greedy algorithm as `cv::dnn::NMSBoxes`, but it's
     implemented by hand, entirely in terms of `computeIoU`, so it's easy to
     explain in an interview or extend (e.g. soft-NMS, class-agnostic NMS).
5. **Draw** — surviving boxes are rendered with class label + confidence.
6. **Latency** — `std::chrono::high_resolution_clock` brackets steps 1–5 per
   frame; the pipeline prints per-frame latency and an average FPS for
   video/camera runs.

## Tuning
- `confThreshold` in `main.cpp` — raise it to cut false positives, lower it
  to catch more (weaker) detections.
- `iouThreshold` — lower it for more aggressive suppression of overlapping
  boxes; raise it to keep more overlapping boxes (useful for crowded scenes).

