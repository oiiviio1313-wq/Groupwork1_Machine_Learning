import sys
import pandas as pd
import numpy as np
import re
import os
sys.stdout.reconfigure(encoding='utf-8-sig')

def load_data(input_path):
    """读取数据（处理编码问题）"""
    try:
        df = pd.read_csv(input_path, encoding='utf-8-sig')
        print(f"成功读取数据：{input_path}")
    except Exception as e:
        print(f"读取数据失败：{str(e)}")
        raise

    print(f"原始数据规模：{df.shape[0]}行 × {df.shape[1]}列")
    print(f"列名列表：{list(df.columns)}")
    return df


def remove_first_col_only_rows(df):
    """删除“仅第一列有值、其余列均为空”的行"""
    df_clean = df.copy()
 
    first_col = df_clean.columns[0]
    other_cols = df_clean.columns[1:]

    df_clean[other_cols] = df_clean[other_cols].replace(['', ' ', '\t', '\n'], np.nan)
    condition = (df_clean[first_col].notna()) & (df_clean[first_col].str.strip() != '') & (df_clean[other_cols].isna().all(axis=1))

    invalid_rows_count = condition.sum()
    df_clean = df_clean[~condition].reset_index(drop=True)

    print(f"\n仅第一列（{first_col}）有值、其余列空的无效行：{invalid_rows_count}行 → 已删除")
    print(f"删除后数据规模：{df_clean.shape[0]}行 × {df_clean.shape[1]}列")
    return df_clean


def clean_duplicates(df, key_cols=None):
    """处理重复值"""
    df_clean = df.copy()
    initial_rows = len(df_clean)

    df_clean = df_clean.drop_duplicates()
    full_dup_count = initial_rows - len(df_clean)
    print(f"\n完全重复行：{full_dup_count}行 → 已删除")

    if key_cols and all(col in df_clean.columns for col in key_cols):
        before_key_dup = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=key_cols, keep='first')
        key_dup_count = before_key_dup - len(df_clean)
        print(f"按{key_cols}去重：{key_dup_count}行 → 已删除")
    elif key_cols:
        print(f"部分关键列不存在，跳过按关键列去重")

    print(f"去重后数据规模：{len(df_clean)}行")
    return df_clean


def clean_missing_values(df, fill_rules=None):
    """处理缺失值"""
    df_clean = df.copy()

    default_rules = {
        '薪资': '面议',
        '学历要求': '不限',
        '工作经验要求': '不限',
        '公司规模': '未公开',
        '职位标签': '无标签',
        '职位描述': '未提供描述'
    }
    final_rules = {**default_rules, **(fill_rules if fill_rules else {})}

    print(f"\n缺失值填充规则：{final_rules}")
    for col, fill_val in final_rules.items():
        if col in df_clean.columns:
            missing_count = df_clean[col].isnull().sum()
            if missing_count > 0:
                df_clean[col] = df_clean[col].fillna(fill_val)
                print(f"{col}：{missing_count}个缺失值 → 填充为'{fill_val}'")
            else:
                print(f"{col}：无缺失值，无需填充")
        else:
            print(f"{col}：列不存在，跳过填充")

    return df_clean


def clean_job_name(df):
    """清洗“职位名称”列"""
    df_clean = df.copy()
    df_clean['职位名称'] = df_clean['职位名称'].astype(str)
    df_clean['职位名称'] = df_clean['职位名称'].str.replace(r'[\n\t【】()（）]', '', regex=True)
    df_clean['职位名称'] = df_clean['职位名称'].str.strip()
    return df_clean


def clean_city_name(df):
    """修改城市名称，保留城市名，去掉区名"""
    df_clean = df.copy()
    df_clean['工作城市'] = df_clean['工作城市'].apply(lambda x: x[:2] if pd.notna(x) and len(str(x)) >= 2 else x)
    return df_clean


def filter_security_jobs(df):
    """筛选安全相关有效职位"""
    df_clean = df.copy()
    # 不包含"安全"的行全部保留
    condition_no_security = ~df_clean['职位名称'].str.contains('安全', na=False)
    # 包含"安全"且包含指定关键词的行保留
    condition_security_valid = (
        df_clean['职位名称'].str.contains('安全', na=False) & 
        df_clean['职位名称'].str.contains('信息|网络|算法|数据', na=False) 
    )

    df_filtered = df_clean[condition_no_security | condition_security_valid].copy()
    
    return df_filtered


def filter_job_description(df):
    """筛选职位描述≥20字的记录"""
    df_clean = df.copy()
    # 处理空值并筛选
    df_filtered = df_clean[df_clean['职位描述'].apply(lambda x: len(str(x)) >= 20 if pd.notna(x) else False)]
    print(f"\n职位描述筛选：原始{len(df_clean)}行 → 保留{len(df_filtered)}行 → 删除{len(df_clean)-len(df_filtered)}行")
    return df_filtered


def save_cleaned_data(df, output_path):
    """保存数据"""
    try:
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n清洗完成！文件已保存至：{output_path}")
        print(f"最终数据规模：{df.shape[0]}行 × {df.shape[1]}列")
    except Exception as e:
        print(f"保存数据失败：{str(e)}")
        raise


if __name__ == "__main__":
    """INPUT_DIR = 'D:/机器学习2/作业一/51job_spider/职业总数据'
    OUTPUT_DIR = 'D:/机器学习2/作业一/51job_spider/职业总数据'"""
   

    INPUT_DIR = r"D:\机器学习2\作业一\51job_spider\分职业数据\清洗前"  
    OUTPUT_DIR = r"D:\机器学习2\作业一\51job_spider\分职业数据\清洗后" 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_files = [
        "job_data_algorithm.csv",
        "job_data_bigdata.csv",
        "job_data_Blockchain.csv",
        "job_data_BusinessAnalyst.csv",
        "job_data_DataSafety.csv",
        "job_data_FundManager.csv",
        "job_data_IndustryAnalyst.csv",
        "job_data_java.csv",
        "job_data_python.csv",
        "job_data_quant.csv",
        "job_data_RiskControl.csv",
        "job_data_sql.csv",
        "job_data_Supplychain.csv"
    ]

    
    for file in csv_files:
        input_path = os.path.join(INPUT_DIR, file)
        output_path = os.path.join(OUTPUT_DIR, f"{file}_cleaned")
        
        raw_df = load_data(input_path)
        no_invalid_df = remove_first_col_only_rows(raw_df)
        dup_cleaned_df = clean_duplicates(df=no_invalid_df, key_cols=['职位名称', '公司名称'])
        missing_cleaned_df = clean_missing_values(df=dup_cleaned_df)
        job_cleaned_df = clean_job_name(df=missing_cleaned_df) 
        city_cleaned_df = clean_city_name(job_cleaned_df)  
        security_filtered_df = filter_security_jobs(city_cleaned_df)
        desc_filtered_df = filter_job_description(security_filtered_df)
        save_cleaned_data(desc_filtered_df, output_path)
        
        print(f'{file} 全流程清洗完成\n{"-"*50}')