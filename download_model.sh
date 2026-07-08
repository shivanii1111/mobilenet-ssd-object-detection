#!/usr/bin/env bash
# Downloads the pretrained MobileNet-SSD (Caffe, VOC-20-class) model files
# from the widely-used chuanqi305/MobileNet-SSD repository mirror.
set -e

mkdir -p models
cd models

echo "Downloading deploy.prototxt ..."
curl -L -o deploy.prototxt \
  "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"

echo "Downloading mobilenet_iter_73000.caffemodel ..."
curl -L -o mobilenet_iter_73000.caffemodel \
  "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel"

echo "Done. Model files are in ./models/"
