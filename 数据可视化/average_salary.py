import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["font.size"] = 10

JOB_TYPE_MAPPING = {
    'Algorithm': '算法工程师',
    'DataAnalyst': '数据分析师',
    'Blockchain': '区块链工程师',
    'BusinessAnalyst': '商业分析师',
    'DataSafety': '数据安全工程师',
    'FundManager': '基金经理',
    'IndustryAnalyst': '行业分析师',
    'Java': 'Java开发工程师',
    'Python': 'Python开发工程师',
    'Quant': '量化分析师',
    'RiskControl': '风控工程师',
    'SQL': 'SQL开发工程师',
    'Supplychain': '供应链管理'
}

def calculate_avg_salary(file_path):
    try:
        df = pd.read_csv(file_path)
        
        if '最高薪资_千/月' not in df.columns or '最低薪资_千/月' not in df.columns:
            raise ValueError("文件中缺少'最高薪资_千/月'或'最低薪资_千/月'列")
        
        df['平均薪资'] = (df['最高薪资_千/月'] + df['最低薪资_千/月']) / 2
        
        overall_avg = df['平均薪资'].mean()
        return round(overall_avg, 2)
    
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        return None

def main():
    INPUT_DIR = 'D:/机器学习2/作业一/51job_spider/分职业数据/清洗后'
    
    csv_files = [
        'job_data_Algorithm_cleaned.csv',
        'job_data_DataAnalyst_cleaned.csv',
        'job_data_Blockchain_cleaned.csv',
        'job_data_BusinessAnalyst_cleaned.csv',
        'job_data_DataSafety_cleaned.csv',
        'job_data_FundManager_cleaned.csv',
        'job_data_IndustryAnalyst_cleaned.csv',
        'job_data_Java_cleaned.csv',
        'job_data_Python_cleaned.csv',
        'job_data_Quant_cleaned.csv',
        'job_data_RiskControl_cleaned.csv',
        'job_data_SQL_cleaned.csv',
        'job_data_Supplychain_cleaned.csv'
    ]
    
    job_avg_salary = {}
    
    for file in csv_files:
        job_type_en = file.replace('job_data_', '').replace('_cleaned.csv', '')
        job_type_cn = JOB_TYPE_MAPPING.get(job_type_en, job_type_en)
        file_path = os.path.join(INPUT_DIR, file)
        avg_salary = calculate_avg_salary(file_path)
        
        if avg_salary is not None:
            job_avg_salary[job_type_cn] = avg_salary
            print(f"{job_type_cn} 平均薪资: {avg_salary} 千/月")
    
    if job_avg_salary:
        sorted_data = sorted(job_avg_salary.items(), key=lambda x: x[1], reverse=True)
        job_types = [item[0] for item in sorted_data]
        avg_salaries = [item[1] for item in sorted_data]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(job_types)))
        
        bars = ax.bar(job_types, avg_salaries, color=colors, edgecolor='#1A237E', linewidth=1.2, width=0.7)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                    f'{height}k', ha='center', va='bottom', rotation=0,
                    fontweight='bold', fontsize=10, color='#0D47A1')
        
        ax.set_title('各类职位平均薪资对比', fontsize=18, fontweight='bold', pad=20, color='#0D47A1')
        ax.set_xlabel('职位类型', fontsize=14, fontweight='bold', labelpad=15, color='#1A237E')
        ax.set_ylabel('平均薪资（千/月）', fontsize=14, fontweight='bold', labelpad=15, color='#1A237E')
        
        ax.set_xticklabels(job_types, rotation=45, ha='right', fontsize=11)
        
        y_max = max(avg_salaries) * 1.15
        ax.set_ylim(0, y_max)
        y_ticks = np.arange(0, int(y_max) + 2, 2)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'{int(tick)}k' for tick in y_ticks], fontsize=11)
        
        ax.grid(axis='y', linestyle='--', alpha=0.7, linewidth=0.8, color='#BBDEFB')
        ax.set_axisbelow(True)
        
        for spine in ax.spines.values():
            spine.set_color('#1A237E')
            spine.set_linewidth(1.2)
        
        ax.tick_params(axis='x', length=0, pad=10)
        
        plt.tight_layout()
        
        plt.savefig('D:/机器学习2/作业一/51job_spider/job_salary_comparison.png', 
                    dpi=300, bbox_inches='tight', facecolor='white')
        
        plt.show()

if __name__ == "__main__":
    main()