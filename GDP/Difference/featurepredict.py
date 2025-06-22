

# 随机森林预测 log-GDP 和 log-Employment 的水平与一/五/十年差分变化
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

# 读取数据
file_path = ("E://FYP//fypcode//Data//Label//FinalLabel.csv")
df = pd.read_csv(file_path)

# 特征列
features = [
    'Population', 'Personalincome', 'Percapitapersonalincome', 'econs',
    'aland', 'awater', 'shape_area', 'shape_leng',
    'Natural Gas Delivered to Consumers in California (Including Vehicle Fuel)  Million Cubic Feet'
]

# 排序并生成滞后特征
df_clean = df.dropna(subset=features + ['GDP', 'Total Employment', 'Year', 'GeoFips'])
df_sorted = df_clean.sort_values(by=['GeoFips', 'Year']).copy()

for lag in [1, 5, 10]:
    lagged = df_sorted.groupby('GeoFips')[features].shift(lag)
    lagged.columns = [f'prev{lag}_' + col for col in lagged.columns]
    df_sorted = pd.concat([df_sorted, lagged], axis=1)

# 计算 log 差分与水平值
df_sorted['log_GDP'] = np.log1p(df_sorted['GDP'])
df_sorted['log_Employment'] = np.log1p(df_sorted['Total Employment'])
df_sorted['log_GDP_diff_1'] = df_sorted['log_GDP'] - df_sorted.groupby('GeoFips')['log_GDP'].shift(1)
df_sorted['log_GDP_diff_5'] = df_sorted['log_GDP'] - df_sorted.groupby('GeoFips')['log_GDP'].shift(5)
df_sorted['log_GDP_diff_10'] = df_sorted['log_GDP'] - df_sorted.groupby('GeoFips')['log_GDP'].shift(10)
df_sorted['log_Employment_diff_1'] = df_sorted['log_Employment'] - df_sorted.groupby('GeoFips')['log_Employment'].shift(1)
df_sorted['log_Employment_diff_5'] = df_sorted['log_Employment'] - df_sorted.groupby('GeoFips')['log_Employment'].shift(5)
df_sorted['log_Employment_diff_10'] = df_sorted['log_Employment'] - df_sorted.groupby('GeoFips')['log_Employment'].shift(10)

# 准备输入输出数据
lagged_cols = features + [f'prev1_{f}' for f in features] + [f'prev5_{f}' for f in features] + [f'prev10_{f}' for f in features]
df_final = df_sorted.dropna(subset=lagged_cols + [
    'log_GDP', 'log_GDP_diff_1', 'log_GDP_diff_5', 'log_GDP_diff_10',
    'log_Employment', 'log_Employment_diff_1', 'log_Employment_diff_5', 'log_Employment_diff_10'
])

scaler = MinMaxScaler()
X = scaler.fit_transform(df_final[lagged_cols])
y_gdp_lvl = df_final['log_GDP'].values
y_gdp_1 = df_final['log_GDP_diff_1'].values
y_gdp_5 = df_final['log_GDP_diff_5'].values
y_gdp_10 = df_final['log_GDP_diff_10'].values
y_emp_lvl = df_final['log_Employment'].values
y_emp_1 = df_final['log_Employment_diff_1'].values
y_emp_5 = df_final['log_Employment_diff_5'].values
y_emp_10 = df_final['log_Employment_diff_10'].values

# 数据打乱并划分训练测试集
X, y_gdp_lvl, y_gdp_1, y_gdp_5, y_gdp_10, y_emp_lvl, y_emp_1, y_emp_5, y_emp_10 = shuffle(
    X, y_gdp_lvl, y_gdp_1, y_gdp_5, y_gdp_10, y_emp_lvl, y_emp_1, y_emp_5, y_emp_10, random_state=42
)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_gdp_lvl_train, y_gdp_lvl_test = y_gdp_lvl[:split], y_gdp_lvl[split:]
y_gdp_1_train, y_gdp_1_test = y_gdp_1[:split], y_gdp_1[split:]
y_gdp_5_train, y_gdp_5_test = y_gdp_5[:split], y_gdp_5[split:]
y_gdp_10_train, y_gdp_10_test = y_gdp_10[:split], y_gdp_10[split:]
y_emp_lvl_train, y_emp_lvl_test = y_emp_lvl[:split], y_emp_lvl[split:]
y_emp_1_train, y_emp_1_test = y_emp_1[:split], y_emp_1[split:]
y_emp_5_train, y_emp_5_test = y_emp_5[:split], y_emp_5[split:]
y_emp_10_train, y_emp_10_test = y_emp_10[:split], y_emp_10[split:]

# 定义随机森林训练函数
def train_eval_rf(X_tr, y_tr, X_te, y_te):
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    return r2_score(y_te, y_pred), np.sqrt(np.mean((y_te - y_pred) ** 2)), np.mean(np.abs(y_te - y_pred))

# 执行训练与评估
results = {}
targets = {
    "log_GDP_level": (y_gdp_lvl_train, y_gdp_lvl_test),
    "log_GDP_diff_1": (y_gdp_1_train, y_gdp_1_test),
    "log_GDP_diff_5": (y_gdp_5_train, y_gdp_5_test),
    "log_GDP_diff_10": (y_gdp_10_train, y_gdp_10_test),
    "log_Employment_level": (y_emp_lvl_train, y_emp_lvl_test),
    "log_Employment_diff_1": (y_emp_1_train, y_emp_1_test),
    "log_Employment_diff_5": (y_emp_5_train, y_emp_5_test),
    "log_Employment_diff_10": (y_emp_10_train, y_emp_10_test)
}

for name, (y_tr, y_te) in targets.items():
    r2, rmse, mae = train_eval_rf(X_train, y_tr, X_test, y_te)
    results[name] = {'R2': r2, 'RMSE': rmse, 'MAE': mae}

# 输出评估结果
for k, v in results.items():
    print(f"{k}: R2 = {v['R2']:.3f}, RMSE = {v['RMSE']:.4f}, MAE = {v['MAE']:.4f}")