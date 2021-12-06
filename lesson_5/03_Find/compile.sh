#!/bin/bash
DIR=`dirname $0`

g++ -std=c++11 -O3 -fopenmp "$DIR"/Find.cpp -I"$DIR"/include/ -o find_omp

./find_omp

