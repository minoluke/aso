#!/bin/bash

# コマンドライン引数を変数に代入
mode=$1

# 定義された値の配列
window_lengths=(30 60 90 120 150)
look_forwards=(30 60 90 120 150)
cvs=(0 1 2 3 4)

# それぞれの組み合わせで実行
for window_length in "${window_lengths[@]}"
do
    for look_forward in "${look_forwards[@]}"
    do
        for cv in "${cvs[@]}"
        do
            echo "Running with mode=${mode}, window_length=${window_length}, look_forward=${look_forward}, cv=${cv}"
            python forecast_model.py $mode $window_length $look_forward $cv
        done
    done
done
