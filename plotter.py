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
min_window = 30              # 最小のウィンドウサイズ
delta_window = 15             # ウィンドウサイズの増分
grid_number = 11    
plotmodel.plot_AUC_colormap(min_window, delta_window, grid_number)