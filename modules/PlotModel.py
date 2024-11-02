from .FeatureExtractionModel import FeatureExtractionModel
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import  MinMaxScaler

makedir = lambda name: os.makedirs(name, exist_ok=True)

class PlotModel(FeatureExtractionModel):
    """
    Description:
        Class for plotting model results and features

    Attributes:
        eruptive_periods_path : str
            Path to eruptive periods file
        data_path : str
            Path to observation data file

    Methods:
        _load_eruptive_periods
            Load eruption periods from file
        plot_feature_histogram
            Plot histogram of feature
        _eruption_within_days
            Check if eruption is within days
        _split_train_test
            Split data into train and test sets
        _preprocess_data
            Preprocess data for plotting
        plot_learning_curve
            Plot learning curve
        plot_AUC_colormap
            Plot AUC colormap
        _get_alarm_periods
            Get alarm periods
        _smooth_data    
            Smooth data
        
    """

    def __init__(self, look_backward, overlap, look_forward, data_streams, ti=None, tf=None, root=None, od=None, cv=None):
        super().__init__(look_backward, overlap, look_forward, data_streams, ti, tf, root, od, cv)
        self.eruptive_periods_path = r'{:s}/data/eruptive_periods.txt'.format(self.rootdir)
        self.data_path = r'{:s}/data/observation_data.dat'.format(self.rootdir)

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
        self.timeseries_dir = r'{:s}/save/figures/timeseries/{:s}'.format(self.rootdir,timestamp)
        self.feature_histogram_dir = r'{:s}/save/figures/feature_histogram/{:s}'.format(self.rootdir, self.od)
        self.learning_curve_dir = r'{:s}/save/figures/learning_curve'.format(self.rootdir)
        self.auc_dir = r'{:s}/save/figures/AUC_colormap'.format(self.rootdir)
           
    def _load_eruptive_periods(self, file_path=None):
        if file_path is None:
            file_path = self.eruptive_periods_path
        eruptive_periods = []
        with open(file_path, 'r') as file:
            for line in file:
                eruptive_periods.append(datetime.strptime(line.strip(), '%Y %m %d %H %M %S'))
        return eruptive_periods

    def plot_feature_histogram(self, feature_name, x_min, x_max, xlabel, months, log_scale=False):
        eruptive_periods = self._load_eruptive_periods()
        file_path = self.featfile
        data = pd.read_csv(file_path)
        time_axis = pd.to_datetime(data['time'])

        # feature_data を NumPy 配列に変換
        feature_data = np.array(data[feature_name])

        # ビンの境界を設定（対数スケールの場合はlogspaceを使用）
        bin_edges = np.logspace(np.log10(x_min), np.log10(x_max), 30) if log_scale else np.linspace(x_min, x_max, 30)

        # 全期間のヒストグラムと噴火前のプロットを同じグラフに描画
        fig, ax = plt.subplots(figsize=(18, 6))  # 横長に調整

        # 全体区間の特徴量データをヒストグラムに描画
        n, bins, patches = ax.hist(feature_data, bins=bin_edges, alpha=0.5, label='All Period')
        vertical_offset = max(n) * 0.9
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel(xlabel, fontsize=25)

        # y軸の最大値を取得し、相対サイズの基準を作る
        ymax = ax.get_ylim()[1]

        # 噴火前の特徴量データをまとめるリスト
        pre_eruption_all_features = []

        for eruption_date in eruptive_periods:
            months = months
            pre_eruption_start = eruption_date - pd.DateOffset(months=months)

            # 噴火前nヶ月間のデータを取得
            pre_eruption_mask = (time_axis >= pre_eruption_start) & (time_axis <= eruption_date)
            pre_eruption_features = np.array(feature_data)[pre_eruption_mask]
            pre_eruption_times = time_axis[pre_eruption_mask]

            # プロットする頻度を下げる
            indices_to_plot = np.linspace(0, len(pre_eruption_features) - 1, min(5, len(pre_eruption_features)), dtype=int)
            pre_eruption_features = pre_eruption_features[indices_to_plot]
            pre_eruption_times = pre_eruption_times.iloc[indices_to_plot]

            # 噴火に近づくにつれてばつ印を大きくするためのサイズリストを作成
            days_until_eruption = (eruption_date - pre_eruption_times).dt.days
            sizes = ymax * 0.9 - 0.02 * ymax * days_until_eruption / months  # 噴火に近づくにつれてサイズが大きくなる

            # 噴火前1ヶ月の特徴量データの位置を太いばつ印でプロット
            ax.scatter(pre_eruption_features, [vertical_offset] * len(pre_eruption_features),
                    label=f'{months} month before {eruption_date.strftime("%Y-%m-%d")}',
                    marker='x', s=sizes, linewidths=5)

            # 噴火前の全特徴量データをまとめる
            pre_eruption_all_features.extend(pre_eruption_features)

            # オフセットを微調整しながら次の噴火を描画
            vertical_offset -= ymax * 0.19

        # U検定を実行してp-valueを計算
        stat, p_value = mannwhitneyu(feature_data, pre_eruption_all_features, alternative='two-sided')

        # p-valueをグラフの左の真ん中に固定して表示
        ax.text(0.05, 0.3, f'p = {p_value:.1e}', fontsize=24, color='black', style='italic', transform=ax.transAxes)

        ax.set_yticks([])  # y軸の目盛りを非表示に
        ax.tick_params(axis='x', labelsize=14)

        # 横軸を対数表示するかどうか
        if log_scale:
            ax.set_xscale('log')  # 横軸をログスケールに変更

        # 枠線を非表示にする
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # 凡例を左上に固定して表示
        ax.legend(loc='upper left', fontsize=14, frameon=False, bbox_to_anchor=(0.02, 0.98), borderaxespad=0)

        plot_dir = self.feature_histogram_dir
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plt.savefig(f'{plot_dir}/{feature_name}.png')
        #plt.show()

    def _eruption_within_days(self,date, eruptive_periods, days):
        for eruption_date in eruptive_periods:
            if date <= eruption_date <= date + timedelta(days=days):
                return True
        return False

    def _split_train_test(self, df, eruptive_periods, start_test, end_test, exclusion_index):
        df['is_test_period'] = df['time'].apply(lambda x: start_test <= x <= end_test and x < eruptive_periods[exclusion_index])
        test_df = df[df['is_test_period']]
        train_df = df[~df['is_test_period']]
        return test_df, train_df

    def _preprocess_data(self, file_path, eruptive_periods, lookforward_days, start_test, end_test, exclusion_index):
        df = pd.read_csv(file_path)
        df['time'] = pd.to_datetime(df['time'])
        df['eruption_within_days'] = df['time'].apply(lambda x: self._eruption_within_days(x, eruptive_periods, lookforward_days))
        test_df, train_df = self._split_train_test(df, eruptive_periods, start_test, end_test, exclusion_index)
        return test_df, train_df

    def plot_learning_curve(self,max_models=10, metrics='AUC'):
        pred_path = self.preddir
        eruptive_periods = self._load_eruptive_periods()
        metric_test_scores = []
        metric_train_scores = []
        model_range = range(1, max_models + 1)
        lookforward_days = self.look_forward

        exclusion_index = int(self.cv)
        start_test = eruptive_periods[exclusion_index] - timedelta(days=180) - timedelta(days=lookforward_days)
        end_test = eruptive_periods[exclusion_index] 

        for num_models in model_range:
            all_y_true_test = []
            all_y_scores_test = []
            all_y_true_train = []
            all_y_scores_train = []

            for i in range(num_models):
                file_path = f'{pred_path}/DecisionTreeClassifier_{i:04d}.csv'
                test_df, train_df = self._preprocess_data(file_path, eruptive_periods, lookforward_days, start_test, end_test, exclusion_index)
                if test_df is None:
                    continue

                # Test data
                all_y_true_test.extend(test_df['eruption_within_days'].astype(int))
                all_y_scores_test.extend(test_df[f'pred{i:04d}'])

                # Train data
                all_y_true_train.extend(train_df['eruption_within_days'].astype(int))
                all_y_scores_train.extend(train_df[f'pred{i:04d}'])

            if metrics == 'AP':
                metric_test = average_precision_score(all_y_true_test, all_y_scores_test) if all_y_true_test else 0
                metric_train = average_precision_score(all_y_true_train, all_y_scores_train) if all_y_true_train else 0
            elif metrics == 'AUC':
                metric_test = roc_auc_score(all_y_true_test, all_y_scores_test) if all_y_true_test else 0
                metric_train = roc_auc_score(all_y_true_train, all_y_scores_train) if all_y_true_train else 0
            else:
                raise ValueError("metrics must be 'AP' or 'AUC'")

            metric_test_scores.append(metric_test)
            metric_train_scores.append(metric_train)

        # Plotting AP for test and train data against the number of models
        plt.figure(figsize=(10, 6))
        plt.plot(model_range, metric_test_scores, label=f'Test Data {metrics}', marker='o')
        plt.plot(model_range, metric_train_scores, label=f'Train Data {metrics}', marker='x')
        plt.xlabel('Number of Models')
        plt.ylabel(metrics)
        plt.title(f'{metrics} for Test and Train Data (Look Forward {lookforward_days} Days, CV={self.cv})')
        plt.legend()
        plt.grid()

    
        plot_dir = self.learning_curve_dir
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plt.savefig(f'{plot_dir}/{self.root}.png')

        #plt.show()

    def plot_AUC_colormap(self, min_window, delta_window, grid_number):
        eruption_number = self.eruption_number
        eruptive_periods = self._load_eruptive_periods()
        observation_data = self.od
        # AUCを計算する関数
        def _calculate_auc(file_path, eruptive_periods, lookforward_days, exclusion_index):
            exclusion_index = int(self.cv)
            start_test = eruptive_periods[exclusion_index] - timedelta(days=180) - timedelta(days=lookforward_days)
            end_test = eruptive_periods[exclusion_index]
            test_df, train_df = self._preprocess_data(file_path, eruptive_periods, lookforward_days, start_test, end_test, exclusion_index)

            # 真のラベルと予測スコア
            y_true = test_df['eruption_within_days'].astype(int)
            y_scores = test_df['consensus']
            if len(np.unique(y_true)) < 2:
                print("y_trueには1つのクラスしか存在しないため、AUCの計算は行われません。")
                return 0.5
            else:
                return roc_auc_score(y_true, y_scores)
            
        # look_backward と look_forward のグリッドポイントを生成
        look_backward_values = [min_window + delta_window * i for i in range(grid_number)]
        look_forward_values = [min_window + delta_window * i for i in range(grid_number)]

        # 各グリッドポイントでAUCを計算
        full_auc_matrix = np.zeros((grid_number, grid_number))
        exclude_auc_matrix = np.zeros((grid_number, grid_number,eruption_number))
        for i, lb_val in enumerate(look_backward_values):
            for j, lf_val in enumerate(look_forward_values):
                full_auc_values = 0
                exclude_auc_values = np.zeros(eruption_number)
                for cv_val in range(eruption_number):
                    self.look_backward = lb_val
                    self.look_forward = lf_val
                    self.cv = cv_val
                    ob_folder = os.path.join(self.consensusdir, self.od)
                    wl_lfl_folder = os.path.join(ob_folder, f"{lb_val}.0_{lf_val}.0")
                    consensus_file = os.path.join(wl_lfl_folder, f"{cv_val}_consensus.csv")
                    auc_value = _calculate_auc(consensus_file, eruptive_periods, lf_val, cv_val)
                    full_auc_values += auc_value
                    exclude_auc_values = [exclude_auc_values[k] + auc_value if k != cv_val else exclude_auc_values[k] for k in range(eruption_number)]
                    
                full_auc_matrix[i, j] = full_auc_values / eruption_number
                exclude_auc_matrix[i, j,:] = np.array(exclude_auc_values) / (eruption_number - 1)


            # カラーマップで表示

        plt.rcParams.update({
            'font.size': 18,         # 全体のフォントサイズ
            'axes.titlesize': 22,    # タイトルのフォントサイズ
            'axes.labelsize': 20,    # 軸ラベルのフォントサイズ
            'xtick.labelsize': 17,   # x軸目盛りのフォントサイズ
            'ytick.labelsize': 17,   # y軸目盛りのフォントサイズ
            'legend.fontsize': 16    # 凡例のフォントサイズ
        })
        
        plt.figure(figsize=(10, 8))
        plt.imshow(full_auc_matrix, cmap='viridis', interpolation='none', aspect='auto', vmin=0.5, vmax=1.0)
        plt.colorbar(label='Mean AUC')
        plt.xticks(ticks=np.arange(len(look_forward_values)), labels=look_forward_values)
        plt.yticks(ticks=np.arange(len(look_backward_values)), labels=look_backward_values)
        plt.xlabel('Look Forward Length (days)')
        plt.ylabel('Look Backward Length (days)')
        plt.title(f'Mean AUC for {observation_data}')
        plt.gca().invert_yaxis()

        auc_dir = r'{:s}/{:s}'.format(self.auc_dir, observation_data)
        if not os.path.exists(auc_dir):
            os.makedirs(auc_dir)
        plt.savefig(f'{auc_dir}/Mean_AUC_{observation_data}.png')

        for cv_val in range(eruption_number):
            plt.figure(figsize=(10, 8))
            plt.imshow(exclude_auc_matrix[:,:,cv_val], cmap='viridis', interpolation='none', aspect='auto', vmin=0.5, vmax=1.0)
            plt.colorbar(label='Mean AUC')
            plt.xticks(ticks=np.arange(len(look_forward_values)), labels=look_forward_values)
            plt.yticks(ticks=np.arange(len(look_backward_values)), labels=look_backward_values)
            plt.xlabel('Look Forward Length (days)')
            plt.ylabel('Look Backward Length (days)')
            plt.title(f'Mean AUC for {observation_data} (Excluding CV={cv_val})')
            plt.gca().invert_yaxis()
            if not os.path.exists(auc_dir):
                os.makedirs(auc_dir)
            plt.savefig(f'{auc_dir}/Mean_AUC_{observation_data}_exclude_{cv_val}.png')

        #plt.show()

    def _get_alarm_periods(self, model_data, eruptive_periods, start_time, end_time, solo=False, threshold=0.65, m_threshold=3):
        alarm_periods = []
        is_alarm = False
        last_alert_time = None
        alarm_end_time = None
        tp, fp, tn, fn = 0, 0, 0, 0

        duration = 1

        eruption_dates = [pd.to_datetime(e) for e in eruptive_periods if start_time <= pd.to_datetime(e) <= end_time]

        times = model_data[next(iter(model_data))]['time']

        # アラーム期間を特定
        for t in times:
            count_above_threshold = 0
            for model in model_data.values():
                filtered = model[model['time'] == t]
                if not filtered.empty and filtered['smoothed'].values[0] > threshold:
                    count_above_threshold += 1

            model_threshold =  m_threshold

            if count_above_threshold >= model_threshold:
                if not is_alarm:
                    last_alert_time = t
                    alarm_end_time = t + pd.Timedelta(days=duration)
                    is_alarm = True
                else:
                    alarm_end_time = max(alarm_end_time, t + pd.Timedelta(days=duration))

            # 警報終了の確認
            if is_alarm and t > alarm_end_time:
                # 警報終了後の期間について噴火の確認
                eruption_in_alarm = any(last_alert_time <= ed <= alarm_end_time for ed in eruption_dates)
                if eruption_in_alarm:
                    tp += 1  # 警報期間中に噴火があった場合
                else:
                    fp += 1  # 警報期間中に噴火がなかった場合
                alarm_periods.append((last_alert_time, alarm_end_time))
                is_alarm = False

        # 最後の警報期間も確認
        if is_alarm:
            eruption_in_alarm = any(last_alert_time <= ed <= alarm_end_time for ed in eruption_dates)
            if eruption_in_alarm:
                tp += 1
            else:
                fp += 1
            alarm_periods.append((last_alert_time, alarm_end_time))

        # 非警報期間の噴火確認
        last_end = start_time
        for start, end in alarm_periods:
            # 警報期間が始まるまでの非警報期間を確認
            eruption_in_non_alarm = any(last_end <= ed <= start for ed in eruption_dates)
            if eruption_in_non_alarm:
                fn += 1  # 非警報期間中に噴火があった場合
            else:
                tn += 1  # 非警報期間中に噴火がなかった場合
            last_end = end

        # 最後の警報終了後の期間も確認
        if last_end < end_time:
            eruption_in_non_alarm = any(last_end <= ed <= end_time for ed in eruption_dates)
            if eruption_in_non_alarm:
                fn += 1
            else:
                tn += 1

        return alarm_periods, tp, fp, tn, fn
        
    def _smooth_data(self, df, window_size):
        # 平滑化処理
        df['smoothed'] = df['consensus'].rolling(window=window_size, min_periods=1).mean()

        # 日付ごとに再サンプリングし、欠損値を補間
        df = df.set_index('time').resample('D').asfreq()  # 'time'列をインデックスにして日単位で再サンプリング
        df['smoothed'] = df['smoothed'].interpolate(method='linear')  # 欠損値を線形補間
        
        # インデックスをリセットして元の形式に戻す
        df = df.reset_index()

        return df

    def _scale_data(self, df):
        scaler = MinMaxScaler()
        df[['smoothed']] = scaler.fit_transform(df[['smoothed']])
        return df

    def calculate_metrics(self, tp, fp, tn, fn):
        # Precisionの計算 (分母が0の場合は0にする)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        # Recallの計算 (分母が0の場合は0にする)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # MCCの計算 (分母が0の場合は0にする)
        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        mcc = numerator / denominator if denominator > 0 else 0

        return mcc, precision, recall

    def _filter_post_eruption(self, df, eruptive_periods, cv, months=6):
        if cv <= -1 or cv > len(eruptive_periods):
            raise ValueError("n must be between 1 and the length of eruptive_periods")

        eruption_date = pd.to_datetime(eruptive_periods[cv])  
        start_date = eruption_date - pd.DateOffset(months=months)

        df = df[(df['time'] >= start_date) & (df['time'] <= (eruption_date + pd.Timedelta(days=1)))]
        return df

    def plot_time_series_with_alarm(self, model_paths, smooth_window_sizes, cv, threshold = 0.65, m_threshold=3):
        model_data = {}
        eruptive_periods = self._load_eruptive_periods()

        # 各モデルのデータを読み込み
        for (name, path), window_size in zip(model_paths.items(), smooth_window_sizes):
            df = pd.read_csv(path)
            df['time'] = pd.to_datetime(df['time'])

            # 平滑化処理
            df = self._smooth_data(df, window_size)

            df =  self._filter_post_eruption(df, eruptive_periods, cv)

            # データの標準化（MinMaxScalerでスケーリング）
            df = self._scale_data(df)

            model_data[name] = df

        # プロットする期間をデータの期間に合わせる
        start_time = max([data['time'].min() for data in model_data.values()])
        end_time = min([data['time'].max() for data in model_data.values()]) + pd.Timedelta(days=60)

        # 警報期間の取得
        alarm_periods, tp, fp, tn, fn = self._get_alarm_periods(model_data, eruptive_periods, start_time, end_time, threshold=threshold, m_threshold=m_threshold)

        mcc, precision, recall = self.calculate_metrics(tp, fp, tn, fn)
        # プロットの比率を5:1に設定
        plt.figure(figsize=(15, 3))
        
        # 各モデルのプロット
        lines = []
        for name, df in model_data.items():
            label_name = name.replace('gas', 'volcanic gas').replace('magnetic', 'magnetic force').replace('kakou', 'crater wall').replace('yudamari', 'hot spring').replace('tremor', 'tremor 1 day').replace('short', 'tremor 10 min')
            line, = plt.plot(df['time'], df['smoothed'], label=f'{label_name} model')
            lines.append(line)

        # 警報期間を薄い赤で塗りつぶす
        for alarm_start, alarm_end in alarm_periods:
            plt.axvspan(alarm_start, alarm_end, color='lightcoral', alpha=0.3)

        # 噴火の日付部分に赤い縦の点線を追加
        for eruption in eruptive_periods:
            eruption_date = pd.to_datetime(eruption)
            if start_time <= eruption_date <= end_time:
                plt.axvline(x=eruption_date, color='red', linestyle='--')

        # プロットの範囲をデータの期間に合わせる
        plt.xlim(start_time, end_time)

        # グラフのラベルとタイトル
        plt.xlabel('Time', fontsize=14)
        plt.ylabel('Prediction Index', fontsize=14)
        plt.title('Prediction Index Time Series with Alarm Periods', fontsize=16)

        # 凡例を追加
        red_patch = plt.Line2D([0], [0], color='lightcoral', lw=4, label='alarm period')
        plt.legend(handles=[red_patch] + lines, loc='upper right', fontsize=12)

        # グリッド表示
        plt.grid(True)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        #plt.show()

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
        timeseries_dir = self.timeseries_dir
        if not os.path.exists(timeseries_dir):
            os.makedirs(timeseries_dir)
        plt.savefig(f'{timeseries_dir}/timeseries_{cv}cv.png')

        return tp, fp, tn, fn, mcc, precision, recall
        
    
    def _calculate_pearson_correlation(self, df1, df2):
        # 2つのデータフレームの時系列データを結合
        merged = pd.merge(df1, df2, on='time', suffixes=('_1', '_2'))

        # 2つのデータフレームの相関係数を計算
        correlation = merged['consensus_1'].corr(merged['consensus_2'])

        return correlation
    
    def _find_consensus_paths(self, window_params, cv):
        consensus_paths = {}
        for name, params in window_params.items():
            window = f'{params[0]}/{params[1]:.1f}_{params[2]:.1f}'
            consensus_paths[name] = f'{self.consensusdir}/{window}/{cv}_consensus.csv'
        return consensus_paths
    

    def plot_pearson_correlation_matrix(self, window_params):
        # データの読み込み
        average_correlation_matrix = np.zeros((len(window_params), len(window_params)))
        for cv in range(self.eruption_number-1):
            consensus_paths = self._find_consensus_paths(window_params, cv)
            df_dict = {}
            for name, path in consensus_paths.items():
                df = pd.read_csv(path)
                df['time'] = pd.to_datetime(df['time'])
                df_dict[name] = df

            # データフレームを１日ごとに再サンプリング
            for name, df in df_dict.items():
                df = df.set_index('time').resample('D').asfreq()
                df = df.interpolate(method='linear')
                df = df.reset_index()
                df_dict[name] = df

            # 噴火前の六ヶ月間でデータをフィルタリング
            for name, df in df_dict.items():
                df_dict[name] = self._filter_post_eruption(df, self._load_eruptive_periods(), cv, months=10)
            

            # データフレームの相関係数を計算
            correlation_matrix = np.zeros((len(df_dict), len(df_dict)))
            for i, (name1, df1) in enumerate(df_dict.items()):
                for j, (name2, df2) in enumerate(df_dict.items()):
                    correlation_matrix[i, j] = self._calculate_pearson_correlation(df1, df2)
            
            abs_correlation_matrix = np.abs(correlation_matrix)
            average_correlation_matrix += abs_correlation_matrix

            plt.figure(figsize=(10, 8))
            plt.imshow(abs_correlation_matrix, cmap='viridis', interpolation='none', aspect='auto', vmin=0, vmax=1)
            plt.colorbar(label='Pearson Correlation Coefficient')

            # x軸とy軸のラベル
            plt.xticks(ticks=np.arange(len(df_dict)), labels=df_dict.keys())
            plt.yticks(ticks=np.arange(len(df_dict)), labels=df_dict.keys())
            plt.xlabel('Observation Data')
            plt.ylabel('Observation Data')
            plt.title(f'Pearson Correlation Coefficient Matrix {cv} of eruption number')

            # 数値を各セルに表示
            for i in range(correlation_matrix.shape[0]):
                for j in range(correlation_matrix.shape[1]):
                    plt.text(j, i, f"{abs_correlation_matrix[i, j]:.2f}",
                            ha='center', va='center', color='white' if abs(correlation_matrix[i, j]) > 0.5 else 'black')

            # グラフの設定
            plt.gca().invert_yaxis()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

        average_correlation_matrix /= (self.eruption_number-1)


        plt.figure(figsize=(10, 8))
        plt.imshow(average_correlation_matrix, cmap='viridis', interpolation='none', aspect='auto', vmin=0, vmax=1)
        plt.colorbar(label='Pearson Correlation Coefficient')
        plt.xticks(ticks=np.arange(len(df_dict)), labels=df_dict.keys())
        plt.yticks(ticks=np.arange(len(df_dict)), labels=df_dict.keys())
        plt.xlabel('Observation Data')
        plt.ylabel('Observation Data')
        plt.title('Average Pearson Correlation Coefficient Matrix')

        for i in range(average_correlation_matrix.shape[0]):
            for j in range(average_correlation_matrix.shape[1]):
                plt.text(j, i, f"{average_correlation_matrix[i, j]:.2f}",
                        ha='center', va='center', color='white' if abs(average_correlation_matrix[i, j]) > 0.5 else 'black')
        plt.gca().invert_yaxis()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

                
        

