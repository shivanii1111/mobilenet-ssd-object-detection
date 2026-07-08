# MobileNet-SSD C++ Object Detection Project — Full Explanation

This document walks through everything about this project: what the model is,
how the code works piece by piece, and the exact steps you personally ran to
get it working on your Mac in VS Code.

---

## 1. What is this project, in plain English?

It's a program that:
1. Takes an image or video as input
2. Runs it through a pretrained neural network that finds objects in the frame (people, cars, dogs, etc.)
3. Draws a green box around each object it finds, with a label and confidence score
4. Tells you how long that took (latency) and how many frames-per-second it could process

The "smart" part (recognizing objects) is done by a pretrained model someone
else already trained. Your C++ code doesn't do any learning — it just loads
that pretrained brain and asks it "what's in this picture?", then processes
the answer.

---

## 2. What is MobileNet-SSD? (the model)

**SSD = Single Shot Detector.** It's an object detection architecture — a
type of neural network design specifically built to look at an image once and
output a list of "I think there's an object here, this size, this
confidence" boxes in a single forward pass (as opposed to older, slower
methods that scanned the image many times).

**MobileNet** is the "backbone" — the part of the network that actually looks
at pixels and extracts features (edges, shapes, textures). MobileNet is
specifically designed to be small and fast (fewer calculations) so it can run
on phones or CPUs without a GPU, at the cost of being slightly less accurate
than bigger backbones like ResNet.

**MobileNet-SSD** = MobileNet backbone + SSD detection head. It's a
lightweight, CPU-friendly object detector — not the most accurate one that
exists today, but a good balance of "fast enough to run in real time" and
"accurate enough to be useful."

**Trained on PASCAL VOC** — the specific pretrained weights you downloaded
were trained on a dataset called PASCAL VOC, which only has 20 object
categories (see section 8 below). That's why the model only recognizes those
20 things and nothing else — it literally never saw a "laptop" or "phone"
during training, so it has no concept of them.

**File format — Caffe.** The model ships as two files:
- `deploy.prototxt` — a text file describing the network's *architecture*
  (how many layers, what type, how they connect). No actual learned numbers
  here, just the blueprint.
- `mobilenet_iter_73000.caffemodel` — a binary file containing the actual
  *learned weights* (millions of numbers) from training. `iter_73000` means
  it was saved after 73,000 training iterations.

Caffe is an older deep learning framework (from before PyTorch/TensorFlow
became dominant). OpenCV's `dnn` module can load Caffe models directly
without needing Caffe itself installed — that's what makes this whole project
possible with just OpenCV.

---

## 3. What is OpenCV DNN?

OpenCV normally does classic computer vision (blurring, edge detection,
resizing, etc.). Its `dnn` module is a lightweight *neural network inference
engine* bolted on — it can **load and run** pretrained models (Caffe,
TensorFlow, ONNX, Darknet) but it **cannot train** them. That's exactly what
we need here: we're not training anything, just running (inferencing) an
already-trained model.

This is why the project didn't need PyTorch, TensorFlow, or any Python at
all — OpenCV's C++ `dnn` module handles loading the model and doing the
math (matrix multiplications through all the layers) by itself.

---

## 4. The overall pipeline (what happens, in order)

```
Image/Video frame
      │
      ▼
[1] PREPROCESS   → resize to 300x300, subtract mean, scale pixel values
      │
      ▼
[2] INFERENCE    → feed into MobileNet-SSD, get raw detection tensor out
      │
      ▼
[3] PARSE        → convert raw numbers into (class, confidence, box) list
      │
      ▼
[4] CUSTOM NMS   → remove duplicate/overlapping boxes for the same object
      │
      ▼
[5] DRAW         → paint boxes + labels onto the image
      │
      ▼
[6] MEASURE      → record how long steps 1-5 took (latency/FPS)
```

Each of these is a distinct block in `src/main.cpp`. Let's go through the
code itself now.

---

## 5. Code walkthrough — `src/main.cpp`

### 5.1 The class list (`VOC_CLASSES`)

```cpp
static const std::vector<std::string> VOC_CLASSES = {
    "background", "aeroplane", "bicycle", ... "tvmonitor"
};
```

The model's raw output only gives you a *number* for each detected object
(e.g. `15`), not a human-readable word. This array is just a lookup table:
index 15 → `"person"`. The order matters — it must exactly match the order
the model was trained with (this is a standard, well-known ordering for VOC
models, not something you can change).

### 5.2 The `Detection` struct

```cpp
struct Detection {
    int classId;
    float confidence;
    cv::Rect box;
};
```

A simple container to carry around one detected object's info together —
which class it is, how confident the model is (0.0 to 1.0), and where its
bounding box is (`cv::Rect` = x, y, width, height in pixels).

### 5.3 `computeIoU()` — Intersection over Union

```cpp
static float computeIoU(const cv::Rect& a, const cv::Rect& b) { ... }
```

IoU is a single number (0 to 1) that says "how much do these two boxes
overlap?" — 0 means no overlap at all, 1 means identical boxes.

Formula: `IoU = (overlapping area) / (total area covered by both boxes combined)`

This is the core building block of NMS (next section) — it's how we
mathematically decide "these two boxes are probably pointing at the same
object" vs. "these are two separate objects that happen to be near each
other."

### 5.4 `customNMS()` — Non-Maximum Suppression, written by hand

This is the most important part of the project to understand, since it's
what makes this a "custom NMS" project instead of just calling a library
function.

**The problem it solves:** SSD-style models tend to output *many* overlapping
boxes for the same real-world object (e.g. 5 slightly different boxes all
around the same dog, each with a different confidence score). We only want
one final box per real object.

**The algorithm (classic greedy NMS):**
1. Sort every candidate detection by confidence, highest first.
2. Take the very first (most confident) box → it's a keeper, add it to results.
3. Look at every *remaining* box of the **same class**. If its IoU with the
   box we just kept is above a threshold (here, `0.45`), throw it away — it's
   almost certainly a duplicate of the same object.
4. Move to the next unprocessed box in the sorted list and repeat step 3.
5. Stop when every box has either been kept or thrown away.

```cpp
std::sort(detections.begin(), detections.end(),
    [](const Detection& a, const Detection& b) {
        return a.confidence > b.confidence;
    });

for (size_t i = 0; i < detections.size(); ++i) {
    if (suppressed[i]) continue;
    results.push_back(detections[i]);          // keep this one

    for (size_t j = i + 1; j < detections.size(); ++j) {
        if (suppressed[j]) continue;
        if (detections[j].classId != detections[i].classId) continue;

        float iou = computeIoU(detections[i].box, detections[j].box);
        if (iou > iouThreshold) suppressed[j] = true;  // discard duplicate
    }
}
```

Why "custom"? OpenCV has a built-in `cv::dnn::NMSBoxes()` function that does
the exact same thing in one line. We deliberately didn't use it, and wrote
the sort + suppress logic ourselves — this is the part worth being able to
explain in an interview, since it shows you understand *how* NMS works
internally, not just that you can call a function that does it.

### 5.5 `processFrame()` — the main pipeline function

This function is called once per image (or once per video frame) and does
steps 1 through 6 from the diagram above.

**Preprocessing:**
```cpp
cv::Mat blob = cv::dnn::blobFromImage(
    frame,
    0.007843,                        // scale = 1/127.5
    cv::Size(300, 300),               // resize target
    cv::Scalar(127.5, 127.5, 127.5),  // mean subtraction
    false, false);
```
Neural networks expect input in a very specific numeric range and size — this
is non-negotiable, it must match exactly how the model was trained. MobileNet-SSD
was trained on 300×300 images with pixel values normalized to roughly
`-1.0` to `1.0` (achieved by subtracting 127.5 then dividing by 127.5). Get
this wrong and the model's predictions become garbage, even though the code
won't throw any error.

**Inference:**
```cpp
net.setInput(blob);
cv::Mat output = net.forward();
```
This is the actual "thinking" step — the blob gets pushed through every
layer of the network, and `output` comes out the other side.

**Parsing the raw output:**
```cpp
cv::Mat detectionMat(output.size[2], output.size[3], CV_32F, output.ptr<float>());
```
SSD's output is a tensor shaped `[1, 1, N, 7]` — N candidate detections,
each described by 7 numbers: `[imageId, classId, confidence, xmin, ymin,
xmax, ymax]`. The coordinates come out *normalized* (between 0 and 1, as a
fraction of image width/height), so we multiply by the actual frame's
width/height to get real pixel coordinates.

```cpp
if (confidence < confThreshold) continue;
```
Anything below the confidence threshold (default `0.4`, i.e. 40% sure) is
thrown away immediately, before NMS even runs — no point comparing boxes the
model wasn't confident about in the first place.

**Drawing:**
```cpp
cv::rectangle(frame, d.box, cv::Scalar(0, 255, 0), 2);
cv::putText(frame, text, ...);
```
Standard OpenCV drawing calls — green rectangle, plus a small filled label
box with the class name and confidence text on top.

**Latency measurement:**
```cpp
auto t0 = std::chrono::high_resolution_clock::now();
// ... all of steps 1-5 ...
auto t1 = std::chrono::high_resolution_clock::now();
double latencyMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
```
`std::chrono` is C++'s standard high-precision timer. We snapshot the clock
right before preprocessing and right after drawing, subtract, and that gap
(in milliseconds) is the "latency" for that one frame. FPS is just
`1000 / latencyMs`.

### 5.6 `main()` — command-line handling and program entry point

```cpp
cv::dnn::Net net = cv::dnn::readNetFromCaffe(protoPath, modelPath);
```
This is where the model actually gets loaded off disk — `protoPath` is
`models/deploy.prototxt` (the blueprint) and `modelPath` is
`models/mobilenet_iter_73000.caffemodel` (the learned weights). **This is
exactly the line that failed with "Can't open models/deploy.prototxt"** when
you ran the program from inside the `build/` folder — the path is relative
to your terminal's *current directory*, not to where the `.exe`/binary file
sits. That's why the fix was to always run the program from the project's
top-level folder.

The rest of `main()` just checks which flag you passed (`--image`,
`--video`, or `--camera`) and either:
- reads one image with `cv::imread` and calls `processFrame()` once, or
- opens a video/camera with `cv::VideoCapture` and calls `processFrame()`
  in a loop, once per frame, until the video ends or you press ESC.

---

## 6. The other project files

### `CMakeLists.txt`
CMake is a *build system generator* — instead of typing a long, error-prone
`g++` command by hand with all the right flags and library paths, you
describe your project once in this file, and CMake figures out the actual
compiler commands for whatever machine it's running on (Mac, Linux, Windows).

Key line:
```cmake
find_package(OpenCV REQUIRED)
target_link_libraries(mobilenet_ssd_detect PRIVATE ${OpenCV_LIBS})
```
This tells CMake "go find the OpenCV library you installed via Homebrew, and
link it into our program" — without this, the code would compile (since it's
valid C++) but fail to *link*, because none of OpenCV's actual functions
would be available to call.

### `download_model.sh`
A small shell script that just runs two `curl` commands to fetch
`deploy.prototxt` and the `.caffemodel` file from GitHub into a `models/`
folder. Nothing clever — it's just automating a manual download so you don't
have to hunt for the URLs yourself.

---

## 7. Steps you actually ran, in order (your real session)

This is the exact sequence from our conversation, including the two hiccups
and how they got fixed:

1. **Installed Xcode command line tools** (`xcode-select --install`) — turned
   out to already be installed on your Mac.
2. **Installed Homebrew and then `cmake` + `opencv`** via
   `brew install cmake opencv` — this gives your Mac the compiler tools and
   the OpenCV library the code depends on.
3. **Ran `download_model.sh`** to fetch `deploy.prototxt` and
   `mobilenet_iter_73000.caffemodel` into `models/`.
4. **First build attempt failed:**
   ```
   CMake Error ... Cannot find source file: src/main.cpp
   ```
   Cause: when downloading the files individually (rather than as one
   folder/zip), `main.cpp` ended up sitting in the top-level project folder
   instead of inside a `src/` subfolder, which is where `CMakeLists.txt`
   expects to find it.
   Fix: created a `src/` folder and moved `main.cpp` into it
   (`mkdir -p src && mv main.cpp src/`), then deleted the old `build/` folder
   and re-ran CMake from scratch so it wouldn't reuse a broken cache.
5. **Build succeeded:**
   ```
   [100%] Built target mobilenet_ssd_detect
   ```
6. **First run attempt failed:**
   ```
   Can't open "models/deploy.prototxt"
   ```
   Cause: you ran the program from *inside* `build/`
   (`./mobilenet_ssd_detect --image ../test.jpg`), but the program looks for
   `models/deploy.prototxt` relative to your terminal's current folder, and
   `build/` doesn't contain a `models/` folder — only the project root does.
   Fix: run the program from the **project root** instead, referencing the
   binary inside `build/`:
   ```bash
   cd ..                      # back to project root
   ./build/mobilenet_ssd_detect --image test.jpg
   ```
7. **Successfully ran detection on an image**, then later on **video**
   (`--video myvideo.mp4`) and learned the **webcam** option
   (`--camera 0`) also exists.

**The one rule to remember going forward:** always run the compiled program
from the project's top-level folder (where `models/` lives), using
`./build/mobilenet_ssd_detect` as the command — never `cd` into `build/`
first.

---

## 8. Classes the model can detect (20 total)

| Category | Classes |
|---|---|
| People & animals | person, bird, cat, cow, dog, horse, sheep |
| Vehicles | aeroplane, bicycle, boat, bus, car, motorbike, train |
| Indoor / furniture | chair, diningtable, pottedplant, sofa, tvmonitor |
| Other | bottle |

No fine-grained subtypes (e.g. it says "dog," never a breed), and no modern
everyday objects like laptops, phones, or backpacks — those simply don't
exist in this model's training data.

---

## 9. Key takeaways to be able to explain this project

If someone asks you about this project, the things worth being able to
speak to confidently are:

- **Why preprocessing matters:** the exact resize/mean/scale values must
  match training, or predictions silently degrade.
- **What the raw model output looks like** and how you parse it
  (`[1,1,N,7]` tensor → class, confidence, box).
- **What NMS is solving** (duplicate overlapping boxes) and **how the greedy
  algorithm works** (sort by confidence → keep top → suppress same-class
  overlaps above an IoU threshold → repeat).
- **Why you wrote NMS by hand** instead of using `cv::dnn::NMSBoxes` — to
  demonstrate you understand the mechanics, not just the API.
- **What latency/FPS measurement tells you** — this is the kind of metric
  that matters for real-time or embedded deployment (ties in nicely with
  your FPGA/YOLOv5 thesis work, where latency is a first-class concern).
- **The limits of this specific model** — lightweight, CPU-friendly, but only
  20 coarse-grained classes, trained on an older/smaller dataset than
  something like COCO.
