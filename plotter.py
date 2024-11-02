from modules import *

data_streams_dict = {
        'tremor': ['rsam', 'mf', 'hf', 'dsar'],
        'gas': ['gas_max', 'gas_min', 'gas_mean', 'gas_number'],
        'magnetic': ['magnetic'],
        'kakou': ['kakouwall_temp'],
        'tilt': ['tilt1_NS', 'tilt1_EW', 'tilt2_NS', 'tilt2_EW'],
        'yudamari': ['yudamari_number', 'yudamari_temp'],
        'all': ['rsam', 'mf', 'hf', 'dsar', 'gas_max', 'gas_min', 'gas_mean', 'gas_number','magnetic', 'kakouwall_temp', 'tilt1_NS', 'tilt1_EW', 'tilt2_NS', 'tilt2_EW','yudamari_number', 'yudamari_temp']
    }

start_period = '2010-01-01'
end_period = '2022-12-31'

overlap = 0.85
look_backward = 90
look_forward = 120
cv=0
od='magnetic'

data_streams = data_streams_dict.get(od)


plotmodel = PlotModel(ti=start_period, tf=end_period, look_backward=look_backward, overlap=overlap, look_forward=look_forward, data_streams=data_streams, od=od, cv=cv)

#plotmodel.plot_learning_curve(max_models=100, metrics='AUC')

#feature_name = 'yudamari_number__fft_coefficient__attr_"real"__coeff_13'

#plotmodel.plot_feature_histogram(feature_name, x_min = -1, x_max = 1, xlabel= "Gas_max Linear Trend (slope)", months = 1, log_scale=False)

'''
min_window = 30              # 最小のウィンドウサイズ
delta_window = 15             # ウィンドウサイズの増分
grid_number = 11    
plotmodel.plot_AUC_colormap(min_window, delta_window, grid_number)


# 実行部分
basepath = 'save/rawdata/consensus'
smooth_window_sizes = [30, 150, 135, 120, 180, 30, 165]

total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0
# 各 n でループ
for cv in range(plotmodel.data.eruption_number):
    model_paths = {
        'gas 1': f'{basepath}/gas/60.0_30.0/{cv}_consensus.csv',
        'gas 2': f'{basepath}/gas/150.0_150.0/{cv}_consensus.csv',
        'kakou': f'{basepath}/kakou/180.0_135.0/{cv}_consensus.csv',
        'magnetic': f'{basepath}/magnetic/60.0_120.0/{cv}_consensus.csv',
        'tilt': f'{basepath}/tilt/180.0_180.0/{cv}_consensus.csv',
        'tremor': f'{basepath}/tremor/30.0_30.0/{cv}_consensus.csv',
        'yudamari': f'{basepath}/yudamari/165.0_165.0/{cv}_consensus.csv',
    }
    tp, fp, tn, fn, mcc, precision, recall = plotmodel.plot_time_series_with_alarm(model_paths, smooth_window_sizes, cv, threshold=0.65, m_threshold=4)

    # 累積計算
    total_tp += tp
    total_fp += fp
    total_tn += tn
    total_fn += fn

# 全体の結果を計算
total_tn = total_tn - total_tp
total_mcc, total_precision, total_recall = plotmodel.calculate_metrics(total_tp, total_fp, total_tn, total_fn)

# 全体の結果を表示
print(f"Total TP: {total_tp}, Total FP: {total_fp}, Total TN: {total_tn}, Total FN: {total_fn}")
print(f"Total MCC: {total_mcc}, Total Precision: {total_precision}, Total Recall: {total_recall}")
'''

window_params = {
        'vlocanicgas': ['gas',180,165],
        'creator wall': ['kakou',180,165],
        'magnetic': ['magnetic',45,165],
        'tilt': ['tilt',180,165],
        'hot water': ['yudamari',165,165],
}

alpha = 150
window_params = {
        'vlocanic gas': ['gas',alpha,alpha],
        'creator wall': ['kakou',alpha,alpha],
        'magnetic': ['magnetic',alpha,alpha],
        'tilt': ['tilt',alpha,alpha],
        'hot water': ['yudamari',alpha,alpha],
        'tremor': ['tremor',alpha,alpha],
}


plotmodel.plot_pearson_correlation_matrix(window_params)

