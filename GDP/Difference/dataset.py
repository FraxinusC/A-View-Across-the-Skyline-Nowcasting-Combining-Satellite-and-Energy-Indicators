import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import datetime
import random
import os
import rasterio
from rasterio.enums import Resampling
from tqdm import tqdm
import pickle

# ---------------------------
# 1. 预处理函数
# ---------------------------
def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['Time'] = df['Year'].apply(lambda x: datetime.datetime(x, 1, 1))

    cols_to_log_transform = ['GDP', 'Population', 'Personalincome', 'Percapitapersonalincome', 'econs',
                             'aland', 'awater','shape_area','shape_leng',
                             'Natural Gas Delivered to Consumers in California (Including Vehicle Fuel)  Million Cubic Feet']
    df[cols_to_log_transform] = df[cols_to_log_transform].apply(np.log)

    df.sort_values(by=['GeoFIPS', 'Time'], inplace=True)
    df['previous_image_path'] = df.groupby('GeoFIPS')['ImagePath'].shift(1)
    for col in ['GDP', 'Population', 'Personalincome', 'Percapitapersonalincome', 'econs',
                'Natural Gas Delivered to Consumers in California (Including Vehicle Fuel)  Million Cubic Feet']:
        df[f'PreviousYear{col}'] = df.groupby('GeoFIPS')[col].shift(1)

    df.rename(columns={'ImagePath': 'current_image_path'}, inplace=True)
    df['IsContinuous'] = (df['Year'] - df.groupby('GeoFIPS')['Year'].shift(1)) == 1
    df_filtered = df[df['IsContinuous'] & df['previous_image_path'].notna()].copy()

    df_filtered['GDPDifference'] = df_filtered['GDP'] - df_filtered['PreviousYearGDP']
    df_filtered['PopulationDifference'] = df_filtered['Population'] - df_filtered['PreviousYearPopulation']
    df_filtered['PersonalincomeDifference'] = df_filtered['Personalincome'] - df_filtered['PreviousYearPersonalincome']
    df_filtered['econsDifference'] = df_filtered['econs'] - df_filtered['PreviousYearecons']
    df_filtered['Natural Gas Delivered to Consumers in California (Including Vehicle Fuel)  Million Cubic Feet Difference'] = \
        df_filtered['Natural Gas Delivered to Consumers in California (Including Vehicle Fuel)  Million Cubic Feet'] - \
        df_filtered['PreviousYearNatural Gas Delivered to Consumers in California (Including Vehicle Fuel)  Million Cubic Feet']

    return df_filtered.drop(['IsContinuous'], axis=1)

# ---------------------------
# 2. 自定义数据集
# ---------------------------
class EconomicDataset(Dataset):
    def __init__(self, dataframe, root_dir, economic_features, img_size, img_augmented_size,
                 scale_factor=0.8, transform=None, scaler=MinMaxScaler()):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.img_size = img_size
        self.img_augmented_size = img_augmented_size
        self.transform = transform
        self.economic_features = economic_features
        self.scale_factor = scale_factor
        self.scaler = scaler

        self.columns_to_scale = ['econs','Population', 'Personalincome', 'Percapitapersonalincome',
                                 'aland', 'awater','shape_area','shape_leng',
                                 'Natural Gas Delivered to Consumers in California (Including Vehicle Fuel)  Million Cubic Feet',
                                 'intptlat','intptlon']

    def read_and_resize_image(self, image_path):
        with rasterio.open(image_path) as src:
            data = src.read(
                out_shape=(
                    src.count,
                    int(src.height * self.scale_factor),
                    int(src.width * self.scale_factor)
                ),
                resampling=Resampling.bilinear
            )
            data = data.astype('float32')
            data[np.isnan(data)] = 0
            return data

    def random_transform(self, image):
        if random.random() > 0.5:
            image = np.flip(image, axis=2)
        _, h, w = image.shape
        crop_size = min(h, w)
        top = (h - crop_size) // 2
        left = (w - crop_size) // 2
        image = image[:, top:top + crop_size, left:left + crop_size].copy()
        image = torch.from_numpy(image).float()
        image = F.interpolate(image.unsqueeze(0), size=(self.img_size, self.img_size),
                              mode='bilinear', align_corners=False).squeeze(0)
        return torch.clamp(image, 0, 1)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        cur_img_path = os.path.join(self.root_dir, row['current_image_path'])
        prev_img_path = os.path.join(self.root_dir, row['previous_image_path'])

        cur_img = self.random_transform(self.read_and_resize_image(cur_img_path))
        prev_img = self.random_transform(self.read_and_resize_image(prev_img_path))
        cur_econ = self.economic_features[idx].float()

        prev_vals = np.array([
            row['PreviousYearecons'], row['PreviousYearPopulation'], row['PreviousYearPersonalincome'],
            row['PreviousYearPercapitapersonalincome'], row['aland'], row['awater'],
            row['shape_area'], row['shape_leng'],
            row['PreviousYearNatural Gas Delivered to Consumers in California (Including Vehicle Fuel)  Million Cubic Feet'],
            row['intptlat'], row['intptlon']
        ]).reshape(1, -1).astype(np.float32)

        prev_df = pd.DataFrame(prev_vals, columns=self.columns_to_scale)
        prev_econ = torch.tensor(self.scaler.transform(prev_df), dtype=torch.float32).squeeze(0)

        econ_diff = cur_econ - prev_econ
        GDP_diff = row['GDPDifference']

        return cur_img, prev_img, cur_econ, prev_econ, econ_diff, GDP_diff

# ---------------------------
# 3. 数据集构建函数
# ---------------------------
def create_datasets(df_cleaned, img_size, img_augmented_size):
    grouped = df_cleaned.groupby('GeoFIPS')
    train_df_list, val_df_list, test_df_list = [], [], []

    for _, group in grouped:
        train, temp = train_test_split(group, test_size=0.3, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)
        train_df_list.append(train)
        val_df_list.append(val)
        test_df_list.append(test)

    train_df = pd.concat(train_df_list)
    val_df = pd.concat(val_df_list)
    test_df = pd.concat(test_df_list)

    columns_to_scale = ['econs','Population', 'Personalincome', 'Percapitapersonalincome',
                        'aland', 'awater','shape_area','shape_leng',
                        'Natural Gas Delivered to Consumers in California (Including Vehicle Fuel)  Million Cubic Feet',
                        'intptlat','intptlon']

    scaler = MinMaxScaler()
    scaler.fit(train_df[columns_to_scale])

    train_features = torch.tensor(scaler.transform(train_df[columns_to_scale]), dtype=torch.float32)
    val_features = torch.tensor(scaler.transform(val_df[columns_to_scale]), dtype=torch.float32)
    test_features = torch.tensor(scaler.transform(test_df[columns_to_scale]), dtype=torch.float32)

    train_dataset = EconomicDataset(train_df, "<train_image_path>", train_features, img_size, img_augmented_size, scaler=scaler)
    val_dataset = EconomicDataset(val_df, "<validation_image_path>", val_features, img_size, img_augmented_size, scaler=scaler)
    test_dataset = EconomicDataset(test_df, "<test_image_path>", test_features, img_size, img_augmented_size, scaler=scaler)

    return train_dataset, val_dataset, test_dataset

# ---------------------------
# 4. 数据保存函数
# ---------------------------
def process_and_save_dataset(dataset, save_path):
    processed_data = []
    for i in tqdm(range(len(dataset))):
        cur_img, prev_img, cur_econ, prev_econ, econ_diff, label = dataset[i]

        cur_econ[torch.isnan(cur_econ)] = 0
        prev_econ[torch.isnan(prev_econ)] = 0
        econ_diff[torch.isnan(econ_diff)] = 0

        processed_data.append((cur_img, prev_img, cur_econ, prev_econ, econ_diff, label))

    with open(save_path, 'wb') as f:
        pickle.dump(processed_data, f)

    print(f"Saved processed data to {save_path}")

# ---------------------------
# 5. 执行流程
# ---------------------------
df_cleaned = preprocess_data("E://FYP//fypcode//Data//Label//ccc.csv")
train_dataset, val_dataset, test_dataset = create_datasets(df_cleaned, 512, 512)

process_and_save_dataset(train_dataset, "5f_train_dataset.pkl")
process_and_save_dataset(val_dataset, "5f_val_dataset.pkl")
process_and_save_dataset(test_dataset, "5f_test_dataset.pkl")
