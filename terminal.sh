#!/bin/sh

g++ -std=c++14 Project_Stereo_Alpha.cpp -I. stereocore.cpp OccupancyGrid.cpp -o stereoProject_gpu `pkg-config --cflags --libs opencv`

./stereoProject_gpu
