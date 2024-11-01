from .FeatureExtractionModel import FeatureExtractionModel
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score, roc_auc_score

makedir = lambda name: os.makedirs(name, exist_ok=True)

class PlotModel(FeatureExtractionModel):
    """
    

    Methods:
        _load_eruptive_periods
            Load eruption periods from file
        plot_feature_histogram
            Plot histogram of feature
        
    """

    def __init__(self, look_backward, overlap, look_forward, data_streams, ti=None, tf=None, root=None, od=None, cv=None):
        super().__init__(look_backward, overlap, look_forward, data_streams, ti, tf, root, od, cv)
        self.eruptive_periods_path = r'{:s}/data/eruptive_periods.txt'.format(self.rootdir)
        self.data_path = r'{:s}/data/observation_data.dat'.format(self.rootdir)
           
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

        plot_dir = r'{:s}/save/figures/feature_histogram/{:s}'.format(self.rootdir, self.od)
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

        plot_dir = r'{:s}/save/figures/learning_curve'.format(self.rootdir)
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plt.savefig(f'{plot_dir}/{self.root}.png')

        #plt.show()

    def plot_AUC_colormap(self, min_window, delta_window, grid_number):
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
        exclude_auc_matrix = np.zeros((grid_number, grid_number,5))
        for i, lb_val in enumerate(look_backward_values):
            for j, lf_val in enumerate(look_forward_values):
                full_auc_values = 0
                exclude_auc_values = np.zeros(5)
                for cv_val in range(5):
                    self.look_backward = lb_val
                    self.look_forward = lf_val
                    self.cv = cv_val
                    ob_folder = os.path.join(self.consensusdir, self.od)
                    wl_lfl_folder = os.path.join(ob_folder, f"{lb_val}.0_{lf_val}.0")
                    consensus_file = os.path.join(wl_lfl_folder, f"{cv_val}_consensus.csv")
                    auc_value = _calculate_auc(consensus_file, eruptive_periods, lf_val, cv_val)
                    full_auc_values += auc_value
                    exclude_auc_values = [exclude_auc_values[k] + auc_value if k != cv_val else exclude_auc_values[k] for k in range(5)]
                    
                full_auc_matrix[i, j] = full_auc_values / 5
                exclude_auc_matrix[i, j,:] = np.array(exclude_auc_values) / 4


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
        auc_dir = r'{:s}/save/figures/AUC_colormap/{:s}'.format(self.rootdir, observation_data)
        if not os.path.exists(auc_dir):
            os.makedirs(auc_dir)
        plt.savefig(f'{auc_dir}/Mean_AUC_{observation_data}.png')

        for cv_val in range(5):
            plt.figure(figsize=(10, 8))
            plt.imshow(exclude_auc_matrix[:,:,cv_val], cmap='viridis', interpolation='none', aspect='auto', vmin=0.5, vmax=1.0)
            plt.colorbar(label='Mean AUC')
            plt.xticks(ticks=np.arange(len(look_forward_values)), labels=look_forward_values)
            plt.yticks(ticks=np.arange(len(look_backward_values)), labels=look_backward_values)
            plt.xlabel('Look Forward Length (days)')
            plt.ylabel('Look Backward Length (days)')
            plt.title(f'Mean AUC for {observation_data} (Excluding CV={cv_val})')
            plt.gca().invert_yaxis()
            auc_dir = r'{:s}/save/figures/AUC_colormap/{:s}'.format(self.rootdir, observation_data)
            if not os.path.exists(auc_dir):
                os.makedirs(auc_dir)
            plt.savefig(f'{auc_dir}/Mean_AUC_{observation_data}_exclude_{cv_val}.png')


        #plt.show()