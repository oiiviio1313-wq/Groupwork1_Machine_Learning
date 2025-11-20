import pandas as pd
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import jieba
import numpy as np
import sys
import re

plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "Microsoft JhengHei", "KaiTi", "FangSong"]
plt.rcParams["axes.unicode_minus"] = False

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def get_stopwords():
    stopwords = {
        '的', '和', '是', '在', '有', '我', '你', '他', '我们', '你们', '他们',
        '工作职责', '岗位职责', '任职要求', '岗位要求', '工作内容', '任职资格', 
        '岗位条件', '工作要求', '职位职责', '职位要求', '岗位职责要求','熟练','执行','推动','跟踪'
        '公司', '工作', '要求', '负责', '进行', '开展', '完成', '具备', '熟悉','建设','构建','建立',
        '经验', '能力', '以上', '相关', '岗位', '职位', '任职', '资格', '优先','公司',
        '良好', '团队', '合作', '精神', '发展', '平台', '提供', '机会', '学习',
        '能够', '独立', '完成', '参与', '协助', '以及', '或者', '主要', '包括',
        '根据', '通过', '基于', '使用', '掌握', '了解', '一定', '较强', '积极','精通',
        '主动', '认真', '负责', '严谨', '高效', '优秀', '良好', '较强', '具有',
        '从事', '以上学历', '学历', '专业',  '分析', '设计', '实现', '维护', '支持', '管理', '项目', '用户',
         '服务', '业务', '应用', '研究', '学习', '处理','考虑','任务'
    }
    return stopwords

def clean_text(text):
    if pd.isna(text):
        return ""
    
    patterns = [
        r'工作职责[:：]?',
        r'岗位职责[:：]?', 
        r'任职要求[:：]?',
        r'岗位要求[:：]?',
        r'工作内容[:：]?',
        r'任职资格[:：]?',
        r'岗位条件[:：]?',
        r'工作要求[:：]?',
        r'职位职责[:：]?',
        r'职位要求[:：]?'
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, '', text)
    
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def generate_wordcloud(file_path, output_dir):
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        
        if '职位描述' not in df.columns:
            print(f"文件 {os.path.basename(file_path)} 中没有'职位描述'列")
            return None
        
        df['职位描述_清理'] = df['职位描述'].apply(clean_text)
        all_text = ' '.join(df['职位描述_清理'].astype(str).fillna(''))
        
        if not all_text.strip():
            print(f"文件 {os.path.basename(file_path)} 的职位描述内容为空")
            return None
        
        seg_list = jieba.cut(all_text)
        seg_text = ' '.join([word for word in seg_list if len(word) > 1])
        
        wc = WordCloud(
            font_path='C:/Windows/Fonts/msyh.ttc',
            width=1200,
            height=800,
            background_color='white',
            max_words=200,
            stopwords=get_stopwords(),
            colormap='tab20',
            max_font_size=200,
            random_state=42,
            margin=5
        ).generate(seg_text)
        
        job_name = os.path.basename(file_path).replace('_cleaned.csv', '').replace('.csv', '')
        output_path = os.path.join(output_dir, f'{job_name}_职位描述词云.png')
        
        plt.figure(figsize=(15, 10))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        
        plt.title(f'{job_name} 职位描述关键词词云', fontsize=20, fontweight='bold', pad=20, 
                 fontfamily='Microsoft YaHei')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[成功] {job_name} 职位描述词云已保存：{output_path}")
        return output_path
        
    except Exception as e:
        print(f"[错误] 处理文件 {os.path.basename(file_path)} 时出错：{str(e)}")
        return None

def main():
    FILE_PATH = 'D:/机器学习2/作业一/51job_spider/职业总数据/merge4_cleaned.csv'
    OUTPUT_DIR = os.path.dirname(FILE_PATH)
    
    if not os.path.exists(FILE_PATH):
        print(f"错误：文件不存在 - {FILE_PATH}")
        return
    
    print(f"开始处理文件：{os.path.basename(FILE_PATH)}\n")
    
    generate_wordcloud(FILE_PATH, OUTPUT_DIR)
    
    print(f"\n[完成] 词云生成完成！")

if __name__ == "__main__":
    plt.switch_backend('Agg')
    
    jieba.initialize()
    
    main()