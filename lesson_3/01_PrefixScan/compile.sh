#!/bin/bash 
DIR=`dirname $0` 
 
nvcc -w -std=c++11 -arch=sm_62 "$DIR"/PrefixScan.cu "$DIR"/src/Timer.cpp "$DIR"/src/Timer.cu -I"$DIR"/include -o prefix_scan
