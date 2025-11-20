import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
from sklearn.feature_selection import SelectFromModel
from statsmodels.stats.outliers_influence import variance_inflation_factor
import re
import warnings

warnings.filterwarnings('ignore')
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
sns.set_style("whitegrid")
sns.set_palette("husl")
sns.set(font_scale=1.0)
sns.set(font="SimHei")

# 加载CSV数据
df = pd.read_csv("D:/11111/quotetutorial/merge4_cleaned_security(2).csv")

# 查看数据基本信息
print("数据基本信息")
print("数据形状（行×列）：", df.shape)

# 薪资数据标准化
def standardize_salary(row):
    min_sal = row["最低薪资_千/月"]
    max_sal = row["最高薪资_千/月"]
    return np.mean([min_sal, max_sal])
df["综合薪资_千/月"] = df.apply(standardize_salary, axis=1)
# 用分位数分层
def salary_tier_quantile(sal):
    q33 = df["综合薪资_千/月"].quantile(0.33)
    q67 = df["综合薪资_千/月"].quantile(0.67)
    if sal <= q33:
        return 0  # 低薪层
    elif sal <= q67:
        return 1  # 中薪层
    else:
        return 2  # 高薪层
df["薪资分层"] = df["综合薪资_千/月"].apply(salary_tier_quantile)

# 验证分层效果
print("全行业薪资分层分布")
tier_dist = df["薪资分层"].value_counts(normalize=True).sort_index()
print(f"低薪层（0）占比：{tier_dist.get(0, 0):.2%}")
print(f"中薪层（1）占比：{tier_dist.get(1, 0):.2%}")
print(f"高薪层（2）占比：{tier_dist.get(2, 0):.2%}")
q33 = df["综合薪资_千/月"].quantile(0.33)
q67 = df["综合薪资_千/月"].quantile(0.67)
print(f"\n全行业薪资分位数阈值：")
print(f"33%分位数 = {q33:.2f}千/月")
print(f"67%分位数 = {q67:.2f}千/月")

# 城市等级分类
def city_tier(city):
    tier1_cities = ["北京", "上海", "广州", "深圳"]
    new_tier1_cities = ["成都", "重庆", "杭州", "武汉", "西安", "天津", "苏州", "南京",
                       "郑州", "长沙", "东莞", "沈阳", "青岛", "合肥", "佛山"]
    tier2_cities = ["昆明", "大连", "厦门","无锡", "济南", "哈尔滨", "温州", "石家庄","南宁","泉州",
                   "长春", "南昌", "贵阳", "金华", "常州", "南通", "嘉兴", "太原",
                   "徐州", "惠州", "珠海", "中山", "台州", "烟台", "兰州", "绍兴",
                   "海口", "扬州","镇江"]
    city = str(city)
    if any(t1 in city for t1 in tier1_cities):
        return "一线城市"
    elif any(nt1 in city for nt1 in new_tier1_cities):
        return "新一线城市"
    elif any(t2 in city for t2 in tier2_cities):
        return "二线城市"
    else:
        return "其他城市"
df["地域集群"] = df["工作城市"].apply(city_tier)

#岗位类型分类
def extract_job_type(row):
    job_keywords = {
        "技术类": ["工程师", "开发", "算法", "数据", "编程", "IT", "软件", "硬件", "测试", "运维",
                   "SQL", "Python", "Java", "大数据", "AI", "人工智能", "机器学习", "分析", "系统"],
        "销售市场类": ["销售", "市场", "业务", "客户经理", "渠道", "商务", "推广", "营销", "广告"],
        "运营职能类": ["运营", "职能", "行政", "人事", "HR", "财务", "会计", "后勤", "文员", "助理",
                       "用户", "社群", "内容", "新媒体", "管理", "协调"],
        "管理类": ["管理", "经理", "主管", "总监", "负责人", "组长", "队长", "首席"],
        "设计类": ["设计", "UI", "UX", "平面", "视觉", "插画", "美术"],
        "教育医疗类": ["教师", "教研", "培训", "讲师", "医生", "护士", "医疗", "护理", "药学"],
        "生产制造类": ["生产", "制造", "车间", "工艺", "质检", "工程", "技术员"],
        "金融风控类": ["金融", "风控", "量化", "投资", "交易", "证券", "基金", "银行", "保险", "风险"],
        "其他岗位": ["其他", "助理", "专员", "文员"]
    }
    text = str(row["职位名称"]) + " " + str(row["职位标签"]) + " " + str(row["职位描述"])
    text_lower = text.lower()
    # 优先级匹配，避免重复分类
    for job_type, kw_list in job_keywords.items():
        for kw in kw_list:
            if kw.lower() in text_lower:
                if job_type == "管理类" and kw == "管理":
                    management_indicators = ["经理", "主管", "总监", "负责人", "管理岗"]
                    if any(indicator in text for indicator in management_indicators):
                        return job_type
                else:
                    return job_type
    return "其他岗位"
df["岗位类型"] = df.apply(extract_job_type, axis=1)
print("\n全行业岗位类型分布")
print(df["岗位类型"].value_counts())

# 岗位技能要求复杂度评分
core_skills_all = [
    "Python", "SQL", "Excel", "Java", "C++", "数据分析", "编程", "运维",
    "沟通", "协调", "管理", "销售", "运营", "设计", "写作", "教学", "护理"
]
# 二进制编码通用技能
for skill in core_skills_all:
    df[f"掌握_{skill}"] = df["职位标签"].apply(lambda x: 1 if skill in str(x) else 0)
# 聚合技能集
df["技术类技能集"] = df[["掌握_Python", "掌握_SQL", "掌握_Java", "掌握_数据分析"]].sum(axis=1).apply(
    lambda x: 1 if x >= 1 else 0)
df["运营销售类技能集"] = df[["掌握_沟通", "掌握_销售", "掌握_运营"]].sum(axis=1).apply(lambda x: 1 if x >= 1 else 0)
df["职能类技能集"] = df[["掌握_Excel", "掌握_协调", "掌握_管理"]].sum(axis=1).apply(lambda x: 1 if x >= 1 else 0)
# 岗位技能复杂度映射
job_complexity = {
    "技术类": 5,
    "设计类": 4,
    "教育医疗类": 4,
    "金融风控类": 4,
    "管理类": 4,
    "运营职能类": 3,
    "销售市场类": 3,
    "生产制造类": 2,
    "其他岗位": 1
}
df["岗位技能要求复杂度"] = df["岗位类型"].map(job_complexity)

# 经验-学历组合评分（满分5）
def extract_years(exp_text):
    exp_text = str(exp_text).strip()
    # 匹配“X年”“X-Y年”“X年及以上”等格式
    pattern = re.findall(r'\d+', exp_text)  # 提取所有数字
    if not pattern:
        return (0, 0)  # 无数字（如“无需经验”）
    years = list(map(int, pattern))
    if len(years) == 1:
        # 处理“X年”“X年及以上”（最小为X，最大设为100表示“及以上”）
        return (years[0], 100)
    elif len(years) >= 2:
        # 处理“X-Y年”（取最小和最大）
        return (min(years), max(years))
    return (0, 0)
def get_exp_score(exp_text):
    min_year, max_year = extract_years(exp_text)
    # 分数规则：按经验要求从低到高，对应1-6分
    if min_year == 0 and max_year == 0:
        return 1  # 无需经验/不限经验
    elif (min_year <= 1 and max_year <= 3) or (min_year == 1 and max_year == 100) or (min_year == 2 and max_year == 100):
        return 2  # 1年
    elif (min_year <= 3 and max_year <= 5) or (min_year >= 3 and max_year <= 5) or (min_year == 3 and max_year == 100) or (min_year == 4 and max_year == 100):
        return 3
    elif (min_year <= 5 and max_year <= 8) or (min_year >= 5 and max_year <= 8)or (min_year ==5  and max_year == 100) or (min_year == 6 and max_year == 100)or (min_year == 7 and max_year == 100):
        return 4
    elif (min_year <= 8 and max_year <= 10) or (min_year >= 8 and max_year <= 10) or (min_year == 8 and max_year == 100) or (min_year == 9 and max_year == 100):
        return 5
    elif min_year >= 10 and max_year == 100:
        return 6
    else:
        return 1
def exp_edu_score(row):
 # exp_map = {"无需经验": 1, "1-3年": 2, "3-5年": 3, "5-8年": 4, "8-10年": 5, "10年及以上": 6}
    exp_text = row["工作经验要求"]
    exp_score = get_exp_score(exp_text)
    edu_map = {"大专":1,"本科": 2, "硕士": 3,"博士": 4}
    edu = row["学历要求"]
    edu_score = edu_map.get(edu, 1)
    return round(exp_score * 0.5+ edu_score * 0.5, 1)
df["经验-学历得分"] = df.apply(exp_edu_score, axis=1)


# 企业属性组合
def comp_attr(row):
    comp_type = row["公司类型"]
    comp_scale = row["公司规模"]
    return f"{comp_type}+{comp_scale}"
df["企业属性组合"] = df.apply(comp_attr, axis=1)

# 计算岗位-薪资匹配度
tech_mean_sal = df.groupby("岗位类型")["综合薪资_千/月"].mean().to_dict()
df["岗位类型均值薪资"] = df["岗位类型"].map(tech_mean_sal)
df["岗位-薪资匹配度"] = df["综合薪资_千/月"] / df["岗位类型均值薪资"]

# 关键可视化结果
# 1-各企业类型的匹配度箱线图
plt.figure(figsize=(16, 8))
sns.boxplot(
    x="公司类型",
    y="岗位-薪资匹配度",
    data=df,
    palette="Set3",
    hue="公司类型",
    legend=False
)
plt.title("各企业类型的岗位-薪资匹配度", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("企业类型", fontsize=14, fontweight='bold')
plt.ylabel("岗位-薪资匹配度（实际薪资/同岗位均值）", fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(fontsize=12)
plt.ylim(0, 3)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("企业类型匹配度.png", dpi=300, facecolor='white')
plt.show()

# 2-经验-学历得分与薪资的关系
plt.figure(figsize=(12, 8))
scatter = sns.regplot(x="经验-学历得分", y="综合薪资_千/月", data=df,
                      scatter_kws={"alpha": 0.6, "s": 50, "color": "steelblue"},
                      line_kws={"color": "crimson", "linewidth": 2.5})
plt.title("经验-学历得分与薪资的相关性", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("经验-学历得分（1-5分）", fontsize=14, fontweight='bold')
plt.ylabel("综合薪资（千/月）", fontsize=14, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("经验学历薪资相关性.png", dpi=300, bbox_inches="tight", facecolor='white')
plt.show()

# 特征工程
# 1. 地域-岗位交互特征
df["地域-岗位交互"] = df["地域集群"] + "-" + df["岗位类型"]
# 2. 经验-学历-岗位适配度
def fit_score(row):
    """经验学历-岗位适配度：1=高适配，0=非高适配"""
    edu = row["学历要求"]  # 本科/硕士/博士
    exp_edu_score = row["经验-学历得分"]  # 1-6分（经验60%+学历40%）
    job_type = row["岗位类型"]  # 技术类/金融风控类/管理类等
    has_tech_skill = row["技术类技能集"]  # 1=掌握至少1项技术技能
    has_func_skill = row["职能类技能集"]  # 1=掌握至少1项职能技能
    has_sales_skill = row["运营销售类技能集"]  # 1=掌握至少1项销售运营技能
    high_fit_combinations = [
        # 1. 高学历 + 高要求岗位（技术/金融/教育医疗）
        (edu == "博士") & (job_type in ["技术类", "金融风控类", "教育医疗类"]),
        (edu == "硕士") & (job_type in ["技术类", "金融风控类", "设计类"]),
        # 2. 高经验学历得分（≥4分，中等偏上资质） + 对应岗位
        (exp_edu_score >= 4.0) & (job_type == "技术类") & (has_tech_skill == 1),  # 高资质+技术岗+技术技能
        (exp_edu_score >= 4.5) & (job_type == "管理类"),  # 高资质+管理岗（管理岗看重综合资质）
        (exp_edu_score >= 4.0) & (job_type == "金融风控类") & (has_tech_skill == 1),  # 金融岗+数据分析技能
        # 3. 特定技能 + 对应岗位（技能匹配）
        (job_type == "技术类") & (has_tech_skill == 1) & (edu in ["本科", "硕士"]),  # 技术岗+技术技能+本科及以上
        (job_type == "职能类") & (has_func_skill == 1) & (exp_edu_score >= 3.0),  # 职能岗+职能技能+中等资质
        (job_type == "销售市场类") & (has_sales_skill == 1) & (exp_edu_score >= 3.0),  # 销售岗+销售技能+中等资质
        # 4. 稀缺组合（高经验+高学历+核心岗位）
        (edu == "硕士") & (exp_edu_score >= 5.0) & (job_type in ["技术类", "管理类"]),
        (edu == "本科") & (exp_edu_score >= 5.5) & (job_type == "技术类") & (has_tech_skill == 1),  # 本科但高经验+技术技能
        # 5. 专业岗位特殊适配（教育医疗/设计类）
        (job_type == "教育医疗类") & (edu in ["硕士", "博士"]) & (exp_edu_score >= 3.5),
        (job_type == "设计类") & (has_tech_skill == 1) & (edu in ["本科", "硕士"])
    ]
    return 1 if any(high_fit_combinations) else 0
df["经验学历-岗位适配度"] = df.apply(fit_score, axis=1)

# 选择候选特征
categorical_features = ["地域集群", "岗位类型", "企业属性组合"]
numerical_features = ["岗位技能要求复杂度", "经验-学历得分", "技术类技能集", "运营销售类技能集", "职能类技能集",
                      "经验学历-岗位适配度"]
# 构建预处理流水线
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_features),
        ("num", StandardScaler(), numerical_features)
    ])
# 生成特征矩阵X和目标变量y
X = df[categorical_features + numerical_features]
y = df["薪资分层"]
# 预处理特征
X_processed = preprocessor.fit_transform(X)
cat_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features)
feature_names = list(cat_feature_names) + numerical_features
X_df = pd.DataFrame(X_processed, columns=feature_names)
# 处理缺失值和异常值
X_df = X_df.fillna(X_df.mean())
for col in X_df.columns:
    if (X_df[col] == np.inf).any():
        max_val = X_df[col].replace(np.inf, np.nan).max()
        X_df[col] = X_df[col].replace(np.inf, max_val)
    if (X_df[col] == -np.inf).any():
        min_val = X_df[col].replace(-np.inf, np.nan).min()
        X_df[col] = X_df[col].replace(-np.inf, min_val)

# VIF检验
def calculate_vif(X_df):
    vif_data = pd.DataFrame()
    vif_data["特征"] = X_df.columns
    vif_values = []
    for i in range(X_df.shape[1]):
        try:
            vif = variance_inflation_factor(X_df.values, i)
            vif_values.append(vif)
        except:
            vif_values.append(np.nan)
            print(f"特征 {X_df.columns[i]} 存在完全共线性，VIF无法计算")
    vif_data["VIF"] = vif_values
    return vif_data
vif_df = calculate_vif(X_df)
print("\nVIF检验结果")
print(vif_df.sort_values("VIF", ascending=False).head(10))
# 剔除VIF>10的特征
high_vif_features = vif_df[vif_df["VIF"] > 10]["特征"].tolist()
X_selected = X_df.drop(columns=high_vif_features)
print(f"\n剔除高VIF特征后，剩余特征数：{X_selected.shape[1]}")

# 随机森林特征重要性
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_selected, y)
feature_importance = pd.DataFrame({
    "特征": X_selected.columns,
    "重要性": rf.feature_importances_
}).sort_values("重要性", ascending=False)
# 筛选重要性>0.03的特征
important_features = feature_importance[feature_importance["重要性"] > 0.03]["特征"].tolist()
X_final = X_selected[important_features]
print("\n最终核心特征（重要性>0.03）")
print(important_features)

#算法选择
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.3, random_state=42, stratify=y
)
# 评估不同算法和损失函数的性能
def evaluate_models_with_different_losses(X_train, X_test, y_train, y_test):
    results = {}
    # SVM模型评估
    print("SVM模型评估")
    svm_models = {
        'SVM_linear': SVC(kernel='linear', random_state=42),
        'SVM_rbf': SVC(kernel='rbf', random_state=42),
        'SVM_poly': SVC(kernel='poly', degree=3, random_state=42)
    }
    for name, model in svm_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        results[name] = {'accuracy': accuracy, 'macro_f1': macro_f1}
        print(f"{name}: 准确率={accuracy:.3f}, 宏F1={macro_f1:.3f}")
    # 逻辑回归模型评估
    print("逻辑回归模型评估")
    lr_models = {
        'LR_l2': LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, random_state=42),
        'LR_l1': LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000, random_state=42),
        'LR_elasticnet': LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000,
                                            random_state=42)
    }
    for name, model in lr_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        results[name] = {'accuracy': accuracy, 'macro_f1': macro_f1}
        print(f"{name}: 准确率={accuracy:.3f}, 宏F1={macro_f1:.3f}")
    # 随机森林模型评估
    print("随机森林模型评估")
    rf_models = {
        'RF_gini': RandomForestClassifier(criterion='gini', n_estimators=100, random_state=42),
        'RF_entropy': RandomForestClassifier(criterion='entropy', n_estimators=100, random_state=42),
        'RF_log_loss': RandomForestClassifier(criterion='log_loss', n_estimators=100, random_state=42)
    }
    for name, model in rf_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        results[name] = {'accuracy': accuracy, 'macro_f1': macro_f1}
        print(f"{name}: 准确率={accuracy:.3f}, 宏F1={macro_f1:.3f}")
    # XGBoost模型评估
    print("XGBoost模型评估")
    xgb_models = {
        'XGB_default': XGBClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42),
        'XGB_logloss': XGBClassifier(n_estimators=150, learning_rate=0.1, max_depth=5,
                                     objective='multi:softprob', random_state=42)
    }
    for name, model in xgb_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        results[name] = {'accuracy': accuracy, 'macro_f1': macro_f1}
        print(f"{name}: 准确率={accuracy:.3f}, 宏F1={macro_f1:.3f}")
    return results
# 运行模型评估
print("\n模型评估")
print("-" * 60)
results = evaluate_models_with_different_losses(X_train, X_test, y_train, y_test)

# 找出最优模型
best_model_name = max(results, key=lambda x: results[x]['macro_f1'])
best_accuracy = results[best_model_name]['accuracy']
best_f1 = results[best_model_name]['macro_f1']
print("\n最优模型结果")
print("-" * 60)
print(f"最优模型: {best_model_name}")
print(f"最佳准确率: {best_accuracy:.3f}")
print(f"最佳宏F1: {best_f1:.3f}")
# 使用最优模型进行详细评估
def train_best_model(model_name, X_train, y_train):
    if 'SVM' in model_name:
        if 'linear' in model_name:
            return SVC(kernel='linear', random_state=42).fit(X_train, y_train)
        elif 'rbf' in model_name:
            return SVC(kernel='rbf', random_state=42).fit(X_train, y_train)
        else:
            return SVC(kernel='poly', degree=3, random_state=42).fit(X_train, y_train)
    elif 'LR' in model_name:
        if 'l1' in model_name:
            return LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000, random_state=42).fit(X_train,
                                                                                                            y_train)
        elif 'elasticnet' in model_name:
            return LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000,
                                      random_state=42).fit(X_train, y_train)
        else:
            return LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, random_state=42).fit(X_train,
                                                                                                        y_train)
    elif 'RF' in model_name:
        if 'entropy' in model_name:
            return RandomForestClassifier(criterion='entropy', n_estimators=100, random_state=42).fit(X_train, y_train)
        elif 'log_loss' in model_name:
            return RandomForestClassifier(criterion='log_loss', n_estimators=100, random_state=42).fit(X_train, y_train)
        else:
            return RandomForestClassifier(criterion='gini', n_estimators=100, random_state=42).fit(X_train, y_train)
    else:
        return XGBClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42).fit(X_train, y_train)
# 训练最优模型
best_model = train_best_model(best_model_name, X_train, y_train)
y_best_pred = best_model.predict(X_test)
# 详细评估函数
def evaluate_model_detailed(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    all_recalls = recall_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    high_recall = all_recalls[2]
    print(f"{model_name} 详细评估结果：")
    print(f"准确率：{accuracy:.3f}")
    print(f"宏F1值：{macro_f1:.3f}")
    print(f"高薪层（标签2）召回率：{high_recall:.3f}")
    print("分类报告：")
    print(classification_report(y_true, y_pred, target_names=["低薪层", "中薪层", "高薪层"],
                                labels=[0, 1, 2], zero_division=0))
evaluate_model_detailed(y_test, y_best_pred, f"最优模型 ({best_model_name})")
# 绘制最优模型混淆矩阵
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_best_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["低薪层", "中薪层", "高薪层"],
            yticklabels=["低薪层", "中薪层", "高薪层"],
            annot_kws={"size": 14, "weight": "bold"})
plt.title(f"最优模型 ({best_model_name}) 薪资分层预测混淆矩阵", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("预测标签", fontsize=14, fontweight='bold')
plt.ylabel("真实标签", fontsize=14, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig(f"{best_model_name}混淆矩阵.png", dpi=300, bbox_inches="tight", facecolor='white')
plt.show()

# 输出特征重要性（如果最优模型是树模型）
if hasattr(best_model, 'feature_importances_'):
    feature_importance_df = pd.DataFrame({
        "特征": X_final.columns,
        "重要性": best_model.feature_importances_
    }).sort_values("重要性", ascending=False).head(10)
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_importance_df)))
    bars = plt.barh(range(len(feature_importance_df)), feature_importance_df["重要性"], color=colors)
    plt.yticks(range(len(feature_importance_df)), feature_importance_df["特征"], fontsize=12)
    plt.xlabel("特征重要性", fontsize=14, fontweight='bold')
    plt.title(f"最优模型 ({best_model_name}) Top10特征重要性", fontsize=16, fontweight='bold', pad=20)
    # 添加数值标签
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.001, bar.get_y() + bar.get_height() / 2,
                 f'{width:.3f}', ha='left', va='center', fontsize=11, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig("最优模型特征重要性.png", dpi=300, facecolor='white')
    plt.show()

# 模型性能比较可视化
results_df = pd.DataFrame(results).T.reset_index()
results_df.columns = ['Model', 'Accuracy', 'Macro_F1']
plt.figure(figsize=(16, 10))
x = np.arange(len(results_df))
width = 0.35
fig, ax = plt.subplots(figsize=(16, 10))
colors_acc = plt.cm.Blues(np.linspace(0.5, 0.9, len(results_df)))
colors_f1 = plt.cm.Oranges(np.linspace(0.5, 0.9, len(results_df)))

rects1 = ax.bar(x - width / 2, results_df['Accuracy'], width, label='准确率',
                color=colors_acc, alpha=0.8, edgecolor='black', linewidth=0.5)
rects2 = ax.bar(x + width / 2, results_df['Macro_F1'], width, label='宏F1',
                color=colors_f1, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.set_xlabel('模型类型', fontsize=14, fontweight='bold')
ax.set_ylabel('分数', fontsize=14, fontweight='bold')
ax.set_title('不同算法和损失函数的性能比较', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=11)
ax.legend(fontsize=12)
ax.grid(axis='y', alpha=0.3)


# 在柱状图上添加数值标签
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
autolabel(rects1)
autolabel(rects2)
fig.tight_layout()
plt.savefig("模型性能比较.png", dpi=300, bbox_inches="tight", facecolor='white')
plt.show()

# 1. 模型性能结果
results_df.to_excel("模型性能分析结果.xlsx", index=False)
# 2. 特征重要性结果
if hasattr(best_model, 'feature_importances_'):
    feature_importance_df.to_excel("特征重要性结果.xlsx", index=False)
# 3. 完整数据
df.to_excel("大数据管理就业岗位分析全数据.xlsx", index=False)
print("所有结果已导出")

# 输出算法选择建议
print("\n算法选择建议")
print(f"实际最优模型: {best_model_name}")