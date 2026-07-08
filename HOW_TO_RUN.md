# How to Run This Project — Quick Reference (Mac)

Keep this file for next time. These are the exact steps that worked for you.

## One-time setup (only needed once ever, already done for you)
- Xcode command line tools installed
- Homebrew installed
- `cmake` and `opencv` installed via Homebrew

If you're on a **new/different Mac**, run this once:
```bash
xcode-select --install
brew install cmake opencv
```

## Every time you want to run the project

### 1. Open the project folder
Open VS Code → **File → Open Folder** → select your `mobilenet-ssd-cpp`
(or whatever you named it) project folder — the one containing
`CMakeLists.txt`, `src/`, `models/`, `build/`.

### 2. Open a terminal in VS Code
`Terminal → New Terminal` (or `Ctrl + \``)

Make sure you're at the **project root** (not inside `build/`):
```bash
pwd
```
should show your project folder path, and:
```bash
ls
```
should show `CMakeLists.txt`, `src`, `models`, `build`, `README.md`, etc.

### 3. Build (only needed if you changed the code, or first time on this machine)
```bash
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j
cd ..
```
If `build/` already has a working `mobilenet_ssd_detect` file inside it and
you haven't changed `main.cpp`, you can skip this step entirely.

### 4. Run it — always from the project root, never from inside `build/`

**On an image:**
```bash
./build/mobilenet_ssd_detect --image test.jpg
```
(replace `test.jpg` with your actual image filename, sitting in the project root)

**On a video:**
```bash
./build/mobilenet_ssd_detect --video myvideo.mp4
```

**On your webcam:**
```bash
./build/mobilenet_ssd_detect --camera 0
```

### 5. What to expect
- A window pops up showing green boxes + labels on detected objects
- Terminal prints latency (ms) and FPS
- For images: `output.jpg` gets saved into your project root
- For video/camera: press **ESC** with the window focused to stop early

---

## The one rule that matters most
**Always run the command from the project root**, using
`./build/mobilenet_ssd_detect` — never `cd build` first and run it from
in there. The program looks for `models/deploy.prototxt` relative to
wherever you typed the command, and that folder only exists at the project
root.

## If something breaks
- `command not found: cmake` → Homebrew/cmake isn't installed on this
  machine, run `brew install cmake opencv`
- `Cannot find source file: src/main.cpp` → `main.cpp` isn't inside a `src/`
  folder; check with `ls src/`
- `Can't open "models/deploy.prototxt"` → you're running from the wrong
  folder; `cd` back to the project root first
- Model files missing → re-run `./download_model.sh` from the project root
