
"""
合并指定路径下的所有csv文件，结果输出为 merge_all.csv
"""
import csv
import os

target_dir = r'D:\机器学习2\作业一\51job_spider\分职业数据\清洗前'
print('目标文件目录:', target_dir)

csv_files = []
for filename in os.listdir(target_dir):

    if filename.endswith('.csv') and not filename.startswith('merge'):
        file_path = os.path.join(target_dir, filename)
        csv_files.append(file_path)
        print(f'找到文件: {file_path}')

if not csv_files:
    raise SystemExit('错误：没有找到任何需要合并的CSV文件！')
aim_dir = r'D:\机器学习2\作业一\51job_spider\职业总数据'

output_path = os.path.join(aim_dir, 'merge4.csv')

with open(output_path, 'w', newline='', encoding='utf-8') as fout:
    writer = None
    total_rows = 0
    
    for idx, file in enumerate(csv_files):
        print(f'\n正在处理: {os.path.basename(file)}  ({idx+1}/{len(csv_files)})')
        try:
            with open(file, 'r', newline='', encoding='utf-8') as fin:
                reader = csv.reader(fin)
 
                if idx == 0:
                    writer = csv.writer(fout)
                    header = next(reader)
                    writer.writerow(header)
                    print(f'写入表头: {header}')
                    row_count = 0
                    for row in reader:
                        writer.writerow(row)
                        row_count += 1
                    total_rows += row_count
                    print(f'写入数据行: {row_count}')

                else:

                    try:
                        header = next(reader)
                    except StopIteration:
                        print(f'警告：文件 {os.path.basename(file)} 为空，跳过')
                        continue
                    
                    row_count = 0
                    for row in reader:
                        writer.writerow(row)
                        row_count += 1
                    total_rows += row_count
                    print(f'写入数据行: {row_count}')
        except Exception as e:
            print(f'错误：处理文件 {os.path.basename(file)} 时出错 - {e}')
            continue

print(f'\n合并完成！')
print(f'输出文件: {output_path}')
print(f'共处理 {len(csv_files)} 个文件')
print(f'总共写入数据行: {total_rows}')