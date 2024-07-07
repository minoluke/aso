#!/bin/bash

# 定義された値の配列
window_lengths=(30 60 90 120)
look_forwards=(30 60 90 120)


# それぞれの組み合わせで実行
for window_length in "${window_lengths[@]}"
do
    for look_forward in "${look_forwards[@]}"
    do
        combination="${window_length}_${look_forward}"
        if [[ " ${skip_combinations[@]} " =~ " ${combination} " ]]; then
            echo "Skipping window_length=${window_length} and look_forward=${look_forward}"
        else
            echo "Running with window_length=${window_length} and look_forward=${look_forward}"
            python forecast_model.py $window_length $look_forward
        fi
    done
done
