import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["figure.facecolor"] = "white"


def classify_job_type(job_name):
    """同分析1：职位类型分类规则"""
    job_name = str(job_name).lower().strip()
    if any(keyword in job_name for keyword in ["开发", "工程师", "技术", "运维", "测试", "编程", "java", "python", "sql", "算法", "大数据", "区块链", "安全", "架构"]):
        return "技术类"
    elif any(keyword in job_name for keyword in ["分析", "研究", "策略", "数据研究", "行业研究", "市场研究", "量化", "风控", "BA", "分析师"]):
        return "分析研究类"
    elif any(keyword in job_name for keyword in ["经理", "主管", "总监", "负责人", "管理", "总监", "VP", "CEO", "厂长"]):
        return "管理类"
    elif any(keyword in job_name for keyword in ["销售", "市场", "商务", "推广", "营销", "渠道", "客户", "BD", "外贸"]):
        return "销售市场类"
    elif any(keyword in job_name for keyword in ["运营", "行政", "人事", "财务", "客服", "HR", "法务", "后勤", "供应链", "采购"]):
        return "运营职能类"
    else:
        return "其他"


def load_job_salary_data(file_path):
    df = pd.read_csv(file_path)

    required_cols = ["职位名称", "最高薪资_千/月", "最低薪资_千/月"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError("数据缺少'职位名称'或'最高薪资_千/月/最低薪资_千/月'列")

    df["平均薪资_千/月"] = (df["最高薪资_千/月"] + df["最低薪资_千/月"]) / 2

    df = df[df["平均薪资_千/月"] > 0]
    

    df["职位类型"] = df["职位名称"].apply(classify_job_type)
    

    job_type_salary = df.groupby("职位类型")["平均薪资_千/月"].agg(["mean", "count"]).reset_index()
    job_type_salary.columns = ["职位类型", "平均薪资_千/月", "职位数量"]
    job_type_salary["平均薪资_千/月"] = job_type_salary["平均薪资_千/月"].round(2)
   
    job_type_salary = job_type_salary.sort_values("平均薪资_千/月", ascending=False)
    print("各职位类型平均薪资（从高到低）：")
    print(job_type_salary.to_string(index=False))
    return job_type_salary

def plot_job_type_salary_bar(job_type_salary):
   
    fig, ax = plt.subplots(figsize=(14, 8))
    
    color_map = LinearSegmentedColormap.from_list("custom_blue", ["#64B5F6", "#1565C0"])
    max_sal = job_type_salary["平均薪资_千/月"].max()
    min_sal = job_type_salary["平均薪资_千/月"].min()

    norm = plt.Normalize(min_sal, max_sal)
    colors = [color_map(norm(sal)) for sal in job_type_salary["平均薪资_千/月"]]

    bars = ax.bar(
        job_type_salary["职位类型"],
        job_type_salary["平均薪资_千/月"],
        color=colors,
        edgecolor="#1A237E",
        linewidth=1.5,
        alpha=0.9,
        width=0.7
    )
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        count = job_type_salary["职位数量"].iloc[i]
        avg_sal = job_type_salary["平均薪资_千/月"].iloc[i]
        
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.3,
            f"{avg_sal}k",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
            color="#0D47A1"
        )
       
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height/2,
            f"n={count}",
            ha="center",
            va="center",
            fontweight="medium",
            fontsize=9,
            color="white",
            alpha=0.9
        )
    

    ax.set_xlabel("职位类型", fontsize=14, fontweight="bold", labelpad=15, color="#263238")
    ax.set_xticklabels(job_type_salary["职位类型"], fontsize=12, fontweight="medium")
    ax.tick_params(axis='x', length=0, pad=10)
    
    y_max = job_type_salary["平均薪资_千/月"].max() * 1.15
    ax.set_ylim(0, y_max)
    y_ticks = np.arange(0, int(y_max) + 2, 2)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{int(tick)}k" for tick in y_ticks], fontsize=11)
    ax.set_ylabel("平均薪资（千/月）", fontsize=14, fontweight="bold", labelpad=15, color="#263238")
    ax.grid(axis="y", linestyle="--", alpha=0.6, linewidth=0.8, color="#E0E0E0")
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_color("#90A4AE")
        spine.set_linewidth(1.2)
    ax.set_title("不同职位类型的平均薪资对比", fontsize=18, pad=25, fontweight="bold", color="#1A237E")
    ax.set_facecolor("#FAFAFA")
    
    plt.tight_layout()
    plt.savefig(r"D:\机器学习2\作业一\51job_spider\job_type_salary_bar_optimized.png", 
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()

if __name__ == "__main__":
    file_path = r"D:\机器学习2\作业一\51job_spider\职业总数据\merge4_cleaned.csv"
    try:
        job_salary_data = load_job_salary_data(file_path)
        plot_job_type_salary_bar(job_salary_data)
    except Exception as e:
        print(f"执行错误：{str(e)}")