mode=$1

# 定義された値の配列
window_lengths=(10 30 60 90 120 150 180)
look_forwards=(10 30 60 90 120 150 180)
cvs=(0 1 2 3 4)

# すでに計算済みのペア
calculated_pairs=(
    "10-10"
    "10-30"
    "10-60"
    "10-90"
    "10-120"
    "30-10"
    "30-30"
    "30-60"
    "30-90"
    "30-120"
    "60-10"
    "60-30"
    "60-60"
    "60-90"
    "60-120"
    "90-10"
    "90-30"
    "90-60"
    "90-90"
    "90-120"
    "120-10"
    "120-30"
    "120-60"
    "120-90"
    "120-120"
)

# それぞれの組み合わせで実行
for window_length in "${window_lengths[@]}"
do
    for look_forward in "${look_forwards[@]}"
    do
        # 組み合わせが既に計算されているかを確認
        pair="${window_length}-${look_forward}"
        if [[ ! " ${calculated_pairs[@]} " =~ " ${pair} " ]]; then
            for cv in "${cvs[@]}"
            do
                echo "Running with mode=${mode}, window_length=${window_length}, look_forward=${look_forward}, cv=${cv}"
                python forecast_model.py $mode $window_length $look_forward $cv
            done
        else
            echo "Skipping already calculated pair: ${pair}"
        fi
    done
done
