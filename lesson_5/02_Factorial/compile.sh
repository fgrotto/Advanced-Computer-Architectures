#!/bin/bash
DIR=`dirname $0`

g++ -std=c++14 -O0 -fopenmp "$DIR"/Factorial.cpp -I"$DIR"/include -o factorial
./factorial
