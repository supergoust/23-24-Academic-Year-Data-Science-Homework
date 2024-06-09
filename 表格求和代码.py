import pandas as pd
import os
import glob

# 源文件夹和目标文件夹的路径（编辑为你本地的位置）
source_folder = r'C:\Users\12864\Desktop\数据科学大作业\ready to sum'
target_folder = r'C:\Users\12864\Desktop\数据科学大作业\have sum'

# 确保目标文件夹存在，如果不存在则创建
os.makedirs(target_folder, exist_ok=True)

# 获取源文件夹中所有的Excel文件
file_paths = glob.glob(os.path.join(source_folder, '*.xlsx'))

# 创建一个空的数据框架用于汇总所有数据
all_data = pd.DataFrame()

# 循环处理每一个文件
for file_path in file_paths:
    # 读取数据
    data = pd.read_excel(file_path)
    
    # 转换日期列为日期格式
    data['date'] = pd.to_datetime(data['date'])
    
    # 添加年份和月份列
    data['year_month'] = data['date'].dt.to_period('M')
    
    # 按年份和月份汇总人数
    monthly_summary = data.groupby('year_month')['population'].sum().reset_index()
    
    # 设置year_month为索引
    monthly_summary.set_index('year_month', inplace=True)
    
    # 获取文件名作为列名
    base_name = os.path.basename(file_path).replace('.xlsx', '')
    
    # 将数据添加到总的数据框架中
    if all_data.empty:
        all_data = monthly_summary.rename(columns={'population': base_name})
    else:
        all_data = all_data.join(monthly_summary.rename(columns={'population': base_name}), how='outer')
    
    print(f"文件 {base_name} 已处理")

# 将索引转换为年月格式的字符串
all_data.index = all_data.index.astype(str)

# 构建最终汇总文件的路径
summary_file_path = os.path.join(target_folder, 'all_data_summary.xlsx')

# 将所有汇总的数据保存到一个Excel文件中
all_data.to_excel(summary_file_path)

print(f"所有文件已处理并汇总保存为 {summary_file_path}")
