#!/bin/bash

# 定義された値の配列
window_lengths=(10 30 50 70 90 110 130)
look_forwards=(10 30 50 70 90 110 130)
cvs=(0 1 2 3 4 5)

# それぞれの組み合わせで実行
for window_length in "${window_lengths[@]}"
do
    for look_forward in "${look_forwards[@]}"
    do
        for cv in "${cvs[@]}"
        do
            echo "Running with window_length=${window_length}, look_forward=${look_forward}, cv=${cv}"
            python forecast_model.py tilt $window_length $look_forward $cv
        done
    done
done
