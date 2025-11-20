import pandas as pd
import numpy as np
import re
import sys
sys.stdout.reconfigure(encoding='utf-8')  

df = pd.read_csv(r'D:\机器学习2\作业一\51job_spider\职业总数据\merge4.csv', encoding='utf-8')

# 1. 查看数据基本信息
print("=== 数据基本信息 ===")
print(f"数据形状：{df.shape}（行：{df.shape[0]}，列：{df.shape[1]}）")
print(f"\n列名列表：")
for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col}（数据类型：{df[col].dtype}）")

# 2. 查看缺失值情况
print("\n=== 缺失值统计 ===")
missing_info = df.isnull().sum()
missing_rate = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({
    '缺失数量': missing_info,
    '缺失率(%)': missing_rate.round(2)
}).sort_values('缺失数量', ascending=False)
print(missing_df[missing_df['缺失数量'] > 0]) 

# 3. 处理并统计完全重复行
print("\n=== 完全重复行处理 ===")
initial_rows = len(df)  
df_clean = df.drop_duplicates() 
duplicate_count = initial_rows - len(df_clean)  
print(f"删除完全重复行后：原{initial_rows}行 → 现{len(df_clean)}行")
print(f"共删除完全重复行：{duplicate_count}行")

# 4. 查看前5行数据（了解数据内容，使用去重后的数据）
print("\n=== 去重后数据预览（前5行）===")
print(df_clean.head())

