import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10
plt.rcParams["axes.linewidth"] = 1.2
plt.figure(figsize=(12, 8))

def extract_city_main(city_str):
    city_str = str(city_str).strip()
    pattern = r"[・\-—\s/]"
    city_main = re.split(pattern, city_str)[0]
    if len(city_main) > 2 and not any(char in city_main for char in ["市", "省", "区"]):
        city_main = city_main[:2]
    return city_main

def classify_city_level(city_main):
    first_tier = ["北京", "上海", "广州", "深圳"]
    new_first_tier = ["成都", "重庆", "杭州", "武汉", "西安", "天津", "苏州", "南京", "郑州", "长沙", "东莞", "沈阳", "青岛", "合肥", "佛山"]
    second_tier = ["昆明", "福州", "无锡", "厦门", "济南", "大连", "哈尔滨", "温州", "石家庄", "南宁", "常州", "泉州", "南昌", "贵阳", "太原", "嘉兴", "烟台", "惠州", "台州", "保定"]
    
    if city_main in first_tier:
        return "一线城市"
    elif city_main in new_first_tier:
        return "新一线城市"
    elif city_main in second_tier:
        return "二线城市"
    else:
        return "其他城市"

def load_city_salary_data(file_path):
    df = pd.read_csv(file_path)
    if "工作城市" not in df.columns or not all(col in df.columns for col in ["最高薪资_千/月", "最低薪资_千/月"]):
        raise ValueError("数据缺少'工作城市'或'最高薪资_千/月/最低薪资_千/月'列")
    
    df["城市主名称"] = df["工作城市"].apply(extract_city_main)
    df["城市等级"] = df["城市主名称"].apply(classify_city_level)
    
    df["平均薪资_千/月"] = (df["最高薪资_千/月"] + df["最低薪资_千/月"]) / 2
    df = df[df["平均薪资_千/月"] > 0]
    
    sigma = 3
    mean_salary = df["平均薪资_千/月"].mean()
    std_salary = df["平均薪资_千/月"].std()
    df = df[(df["平均薪资_千/月"] >= mean_salary - sigma * std_salary) & 
            (df["平均薪资_千/月"] <= mean_salary + sigma * std_salary)]
    
    city_level_order = ["一线城市", "新一线城市", "二线城市", "其他城市"]
    level_salary_data = []
    level_job_count = []
    for level in city_level_order:
        level_df = df[df["城市等级"] == level]
        if len(level_df) > 0:
            level_salary_data.append(level_df["平均薪资_千/月"].values)
            level_job_count.append(len(level_df))
        else:
            level_salary_data.append(np.array([]))
            level_job_count.append(0)
    
    print("各城市等级职位数量与平均薪资：")
    for i, level in enumerate(city_level_order):
        if level_job_count[i] > 0:
            avg_sal = np.mean(level_salary_data[i]).round(2)
            print(f"{level}：职位{level_job_count[i]}个，平均薪资{avg_sal}k")
        else:
            print(f"{level}：无有效职位数据")
    
    valid_mask = [len(data) > 0 for data in level_salary_data]
    valid_levels = [level for i, level in enumerate(city_level_order) if valid_mask[i]]
    valid_salary_data = [data for i, data in enumerate(level_salary_data) if valid_mask[i]]
    
    if len(valid_levels) < 2:
        raise ValueError("有效城市等级数据过少，无法绘制箱线图")
    return valid_salary_data, valid_levels

def plot_city_level_salary_box(valid_salary_data, valid_levels):
    colors = ["#0D47A1", "#1565C0", "#1976D2", "#1E88E5"][:len(valid_levels)]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bp = ax.boxplot(
        valid_salary_data,
        labels=valid_levels,
        patch_artist=True,
        notch=True,
        showfliers=True,
        flierprops=dict(marker="o", markerfacecolor="#FF5252", markersize=4, alpha=0.6),
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(color="#0D47A1", linewidth=1.2),
        capprops=dict(color="#0D47A1", linewidth=1.2),
        boxprops=dict(linewidth=1.2),
        widths=0.6
    )
    
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    
    ax.set_xlabel("城市等级", fontsize=14, fontweight="bold", labelpad=15, color="#0D47A1")
    ax.set_xticklabels(valid_levels, fontsize=12, fontweight="medium", rotation=0)
    ax.tick_params(axis='x', length=0, pad=10)
    
    all_data = np.concatenate(valid_salary_data)
    y_min = max(0, np.min(all_data) * 0.95)
    y_max = np.max(all_data) * 1.05
    
    y_ticks = np.arange(int(y_min), int(y_max) + 2, 2)
    ax.set_yticks(y_ticks)
    ax.set_ylim(y_min - 1, y_max + 1)
    
    ax.set_ylabel("平均薪资（千/月）", fontsize=14, fontweight="bold", labelpad=15, color="#0D47A1")
    ax.set_yticklabels([f"{int(tick)}k" for tick in y_ticks], fontsize=11)
    
    ax.set_title("不同城市等级的薪资分布对比", fontsize=18, pad=20, fontweight="bold", color="#0D47A1")
    
    ax.grid(axis="y", linestyle="--", alpha=0.7, linewidth=0.8, color="#BBDEFB")
    ax.set_axisbelow(True)
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("#0D47A1")
    
    for i, data in enumerate(valid_salary_data):
        ax.text(i+1, y_min, f'n={len(data)}', ha='center', va='top', 
                fontsize=10, fontweight='bold', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD', alpha=0.8, edgecolor="#1976D2"))
    
    plt.tight_layout()
    plt.savefig(r"D:\机器学习2\作业一\51job_spider\city_level_salary_box_blue.png", 
                dpi=300, bbox_inches="tight", facecolor='white')
    plt.show()

if __name__ == "__main__":
    file_path = r"D:\机器学习2\作业一\51job_spider\职业总数据\merge4_cleaned.csv"
    try:
        salary_data, city_levels = load_city_salary_data(file_path)
        plot_city_level_salary_box(salary_data, city_levels)
        print("蓝色主题的城市等级薪资分布箱线图已保存！")
    except Exception as e:
        print(f"执行错误：{str(e)}")