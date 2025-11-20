import pandas as pd
import numpy as np
import re
import os

input_folder = r"D:\机器学习2\作业一\51job_spider\分职业数据\清洗前"  
output_folder = r"D:\机器学习2\作业一\51job_spider\分职业数据\清洗后" 

os.makedirs(output_folder, exist_ok=True)
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

def extract_salary_range_improved(salary_str):

    if pd.isna(salary_str) or salary_str == '':
        return np.nan, np.nan
    
    salary_str = str(salary_str).strip()
    
    # 1. 区分月薪和年薪
    is_annual = '年' in salary_str
    
    # 2. 改进的正则表达式，更灵活地匹配数字和单位
    # 匹配模式1：数字+单位-数字+单位（如"8千-1.2万"）
    pattern = r'(\d+\.?\d*)\s*([万千])\s*-\s*(\d+\.?\d*)\s*([万千])'
    match = re.search(pattern, salary_str)
    
    if not match:
        # 匹配模式2：数字-数字+单位（如"1-1.5万"，两个数字共用一个单位）
        pattern2 = r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\s*([万千])'
        match2 = re.search(pattern2, salary_str)
        
        if match2:
            min_num, max_num, unit = match2.groups()
            min_val = float(min_num)
            max_val = float(max_num)
            
            # 根据单位转换（统一为"千"）
            if unit == '万':
                min_val *= 10
                max_val *= 10
            elif unit == '千':
                min_val *= 1
                max_val *= 1
        else:
            return np.nan, np.nan
    else:
        # 模式1匹配成功，分别处理两个数字的单位
        min_num, min_unit, max_num, max_unit = match.groups()
        min_val = float(min_num)
        max_val = float(max_num)
        
        if min_unit == '万':
            min_val *= 10
        elif min_unit == '千':
            min_val *= 1
        
        if max_unit == '万':
            max_val *= 10
        elif max_unit == '千':
            max_val *= 1
    
    # 3. 年薪转换为月薪（除以12，保留1位小数）
    if is_annual:
        min_val = round(min_val / 12, 1)
        max_val = round(max_val / 12, 1)
    
    # 4. 确保最低薪资≤最高薪资
    if min_val > max_val:
        min_val, max_val = max_val, min_val
    
    return min_val, max_val

processed_files = 0
total_records = 0
total_normal_conversions = 0
for file in csv_files:
    input_path = os.path.join(input_folder, file)
    output_file = file.replace(".csv", "_cleaned.csv") 
    output_path = os.path.join(output_folder, output_file)
    
    try:
        df = pd.read_csv(input_path, encoding='utf-8-sig')
        df_improved = df.copy()
        salary_results_improved = df_improved['薪资'].apply(extract_salary_range_improved)
        df_improved['最低薪资_千/月'] = [result[0] for result in salary_results_improved]
        df_improved['最高薪资_千/月'] = [result[1] for result in salary_results_improved]
        df_improved['转换状态'] = df_improved.apply(
            lambda x: '转换正常' if pd.notna(x['最低薪资_千/月']) and pd.notna(x['最高薪资_千/月']) and x['最低薪资_千/月'] <= x['最高薪资_千/月'] else '数值无效',
            axis=1
        )
        df_improved['薪资类型'] = df_improved['薪资'].apply(
            lambda x: '年薪' if pd.notna(x) and '年' in str(x) else '月薪'
        )
        df_improved.to_csv(output_path, index=False, encoding='utf-8-sig')
        normal_count = len(df_improved[df_improved['转换状态'] == '转换正常'])
        processed_files += 1
        total_records += len(df_improved)
        total_normal_conversions += normal_count
        print(f" 处理完成: {file}")
        print(f"  - 记录总数: {len(df_improved)} 条")
        print(f"  - 转换正常: {normal_count} 条")
        print(f"  - 输出文件: {output_file}\n")
        
    except Exception as e:
        print(f"处理失败: {file}")
