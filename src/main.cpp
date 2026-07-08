// MobileNet-SSD Object Detection Pipeline with Custom NMS
// -------------------------------------------------------
// Pipeline stages:
//   1. Load Caffe MobileNet-SSD model via OpenCV DNN
//   2. Preprocess frame (resize, mean-subtract, scale) into a blob
//   3. Run forward inference
//   4. Parse raw detections
//   5. Apply CUSTOM IoU-based Non-Maximum Suppression (no cv::dnn::NMSBoxes)
//   6. Draw bounding boxes + labels + confidence
//   7. Measure and report per-frame latency / FPS
//
// Build: see CMakeLists.txt
// Usage:
//   ./mobilenet_ssd_detect --image path/to/image.jpg
//   ./mobilenet_ssd_detect --video path/to/video.mp4
//   ./mobilenet_ssd_detect --camera 0

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>

// -----------------------------------------------------------------------
// VOC class labels used by the standard chuanqi305/MobileNet-SSD Caffe model
// (index 0 = background, which we never draw)
// -----------------------------------------------------------------------
static const std::vector<std::string> VOC_CLASSES = {
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor"
};

// -----------------------------------------------------------------------
// Simple struct representing one detected object before/after NMS
// -----------------------------------------------------------------------
struct Detection {
    int classId;
    float confidence;
    cv::Rect box;
};

// -----------------------------------------------------------------------
// Intersection-over-Union between two boxes
// -----------------------------------------------------------------------
static float computeIoU(const cv::Rect& a, const cv::Rect& b) {
    int x1 = std::max(a.x, b.x);
    int y1 = std::max(a.y, b.y);
    int x2 = std::min(a.x + a.width,  b.x + b.width);
    int y2 = std::min(a.y + a.height, b.y + b.height);

    int interW = std::max(0, x2 - x1);
    int interH = std::max(0, y2 - y1);
    float interArea = static_cast<float>(interW) * interH;

    float unionArea = static_cast<float>(a.area()) + static_cast<float>(b.area()) - interArea;
    if (unionArea <= 0.0f) return 0.0f;
    return interArea / unionArea;
}

// -----------------------------------------------------------------------
// Custom Non-Maximum Suppression (class-aware, IoU-threshold based)
//
// Standard greedy NMS:
//   1. Sort all candidate detections by confidence, descending
//   2. Take the highest-confidence box, keep it
//   3. Suppress every remaining box of the SAME class whose IoU with the
//      kept box exceeds iouThreshold
//   4. Repeat until no boxes remain
// -----------------------------------------------------------------------
static std::vector<Detection> customNMS(std::vector<Detection> detections, float iouThreshold) {
    std::vector<Detection> results;

    std::sort(detections.begin(), detections.end(),
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });

    std::vector<bool> suppressed(detections.size(), false);

    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) continue;

        results.push_back(detections[i]);

        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppressed[j]) continue;
            if (detections[j].classId != detections[i].classId) continue;

            float iou = computeIoU(detections[i].box, detections[j].box);
            if (iou > iouThreshold) {
                suppressed[j] = true;
            }
        }
    }
    return results;
}

// -----------------------------------------------------------------------
// Run the full pipeline (preprocess -> inference -> parse -> NMS -> draw)
// on a single frame. Returns latency in milliseconds.
// -----------------------------------------------------------------------
static double processFrame(cv::dnn::Net& net, cv::Mat& frame,
                            float confThreshold, float iouThreshold) {
    auto t0 = std::chrono::high_resolution_clock::now();

    // ---- 1. Preprocess: MobileNet-SSD expects 300x300 BGR input,
    //         mean-subtraction of 127.5 on each channel, scale 1/127.5 ----
    cv::Mat blob = cv::dnn::blobFromImage(
        frame,
        0.007843,                       // scale factor = 1/127.5
        cv::Size(300, 300),              // network input size
        cv::Scalar(127.5, 127.5, 127.5), // mean subtraction
        false,                            // swapRB (BGR already matches training)
        false);                           // crop

    net.setInput(blob);

    // ---- 2. Inference ----
    cv::Mat output = net.forward();

    // ---- 3. Parse raw detections ----
    // Output shape: [1, 1, N, 7] where each row is:
    // [imageId, classId, confidence, xmin, ymin, xmax, ymax] (normalized 0-1)
    cv::Mat detectionMat(output.size[2], output.size[3], CV_32F, output.ptr<float>());

    std::vector<Detection> rawDetections;
    int frameW = frame.cols;
    int frameH = frame.rows;

    for (int i = 0; i < detectionMat.rows; ++i) {
        float confidence = detectionMat.at<float>(i, 2);
        if (confidence < confThreshold) continue;

        int classId = static_cast<int>(detectionMat.at<float>(i, 1));
        int xmin = static_cast<int>(detectionMat.at<float>(i, 3) * frameW);
        int ymin = static_cast<int>(detectionMat.at<float>(i, 4) * frameH);
        int xmax = static_cast<int>(detectionMat.at<float>(i, 5) * frameW);
        int ymax = static_cast<int>(detectionMat.at<float>(i, 6) * frameH);

        // Clamp to frame bounds
        xmin = std::max(0, std::min(xmin, frameW - 1));
        ymin = std::max(0, std::min(ymin, frameH - 1));
        xmax = std::max(0, std::min(xmax, frameW - 1));
        ymax = std::max(0, std::min(ymax, frameH - 1));

        if (xmax <= xmin || ymax <= ymin) continue;

        Detection d;
        d.classId = classId;
        d.confidence = confidence;
        d.box = cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin);
        rawDetections.push_back(d);
    }

    // ---- 4. Custom IoU-based NMS ----
    std::vector<Detection> finalDetections = customNMS(rawDetections, iouThreshold);

    // ---- 5. Draw results ----
    for (const auto& d : finalDetections) {
        cv::rectangle(frame, d.box, cv::Scalar(0, 255, 0), 2);

        std::string label = (d.classId >= 0 && d.classId < static_cast<int>(VOC_CLASSES.size()))
                                 ? VOC_CLASSES[d.classId]
                                 : "unknown";
        char confText[16];
        std::snprintf(confText, sizeof(confText), "%.2f", d.confidence);
        std::string text = label + " " + confText;

        int baseline = 0;
        cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        int textY = std::max(d.box.y, textSize.height + 4);

        cv::rectangle(frame,
                      cv::Point(d.box.x, textY - textSize.height - 4),
                      cv::Point(d.box.x + textSize.width, textY),
                      cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(frame, text, cv::Point(d.box.x, textY - 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double latencyMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::string perf = cv::format("Latency: %.1f ms | FPS: %.1f", latencyMs, 1000.0 / latencyMs);
    cv::putText(frame, perf, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 0, 255), 2);

    return latencyMs;
}

// -----------------------------------------------------------------------
// CLI argument helper
// -----------------------------------------------------------------------
static std::string getArg(int argc, char** argv, const std::string& flag) {
    for (int i = 1; i < argc - 1; ++i) {
        if (flag == argv[i]) return std::string(argv[i + 1]);
    }
    return "";
}

int main(int argc, char** argv) {
    std::string protoPath  = "models/deploy.prototxt";
    std::string modelPath  = "models/mobilenet_iter_73000.caffemodel";
    float confThreshold = 0.4f;
    float iouThreshold  = 0.45f;

    std::string imagePath  = getArg(argc, argv, "--image");
    std::string videoPath  = getArg(argc, argv, "--video");
    std::string cameraArg  = getArg(argc, argv, "--camera");

    if (imagePath.empty() && videoPath.empty() && cameraArg.empty()) {
        std::cout << "Usage:\n"
                  << "  " << argv[0] << " --image path/to/image.jpg\n"
                  << "  " << argv[0] << " --video path/to/video.mp4\n"
                  << "  " << argv[0] << " --camera 0\n";
        return 0;
    }

    // ---- Load network ----
    cv::dnn::Net net = cv::dnn::readNetFromCaffe(protoPath, modelPath);
    if (net.empty()) {
        std::cerr << "ERROR: could not load network from " << protoPath
                  << " / " << modelPath << "\n";
        return 1;
    }
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    if (!imagePath.empty()) {
        // ---- Single image mode ----
        cv::Mat frame = cv::imread(imagePath);
        if (frame.empty()) {
            std::cerr << "ERROR: could not read image " << imagePath << "\n";
            return 1;
        }

        double latency = processFrame(net, frame, confThreshold, iouThreshold);
        std::cout << "Inference latency: " << latency << " ms\n";

        cv::imwrite("output.jpg", frame);
        std::cout << "Saved result to output.jpg\n";

        cv::imshow("MobileNet-SSD Detection", frame);
        cv::waitKey(0);
    } else {
        // ---- Video / camera mode ----
        cv::VideoCapture cap;
        if (!videoPath.empty()) {
            cap.open(videoPath);
        } else {
            cap.open(std::stoi(cameraArg));
        }

        if (!cap.isOpened()) {
            std::cerr << "ERROR: could not open video source\n";
            return 1;
        }

        cv::Mat frame;
        std::vector<double> latencies;

        while (cap.read(frame)) {
            if (frame.empty()) break;

            double latency = processFrame(net, frame, confThreshold, iouThreshold);
            latencies.push_back(latency);

            cv::imshow("MobileNet-SSD Detection", frame);
            if (cv::waitKey(1) == 27) break; // ESC to quit
        }

        if (!latencies.empty()) {
            double avg = 0;
            for (double l : latencies) avg += l;
            avg /= latencies.size();
            std::cout << "Average latency: " << avg << " ms over "
                      << latencies.size() << " frames ("
                      << (1000.0 / avg) << " FPS avg)\n";
        }
    }

    return 0;
}
