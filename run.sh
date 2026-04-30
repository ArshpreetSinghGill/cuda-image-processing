#!/bin/bash

INPUT1=data/input/textures
INPUT2=data/input/misc

OUTPUT_EDGE=data/output/edge
OUTPUT_EQ=data/output/equalized
OUTPUT_SHARP=data/output/sharpen

mkdir -p $OUTPUT_EDGE
mkdir -p $OUTPUT_EQ
mkdir -p $OUTPUT_SHARP

echo "=============================="
echo "Running EDGE DETECTION..."
echo "=============================="

./app $INPUT1 $OUTPUT_EDGE --mode=edge
./app $INPUT2 $OUTPUT_EDGE --mode=edge

echo "=============================="
echo "Running HISTOGRAM EQUALIZATION..."
echo "=============================="

./app $INPUT1 $OUTPUT_EQ --mode=equalize
./app $INPUT2 $OUTPUT_EQ --mode=equalize

echo "=============================="
echo "Running SHARPENING..."
echo "=============================="

./app $INPUT1 $OUTPUT_SHARP --mode=sharpen
./app $INPUT2 $OUTPUT_SHARP --mode=sharpen

echo "=============================="
echo "ALL PROCESSING COMPLETE"
echo "=============================="