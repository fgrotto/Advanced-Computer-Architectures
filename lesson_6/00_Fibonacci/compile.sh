#!/bin/bash
DIR=`dirname $0`

g++ -std=c++11 -fopenmp "$DIR"/Fibonacci.cpp -I"$DIR"/include/ -o fibonacci
./fibonacci
