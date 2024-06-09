import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 加载数据(这里注意改成自己文件所放置的位置)
data = pd.read_excel(r'C:\Users\12864\Desktop\数据科学大作业\行为各项求和数据.xlsx', usecols=['indicator_name', 'population', 'date'])

# 转换日期列为日期格式（如果它还不是）
data['date'] = pd.to_datetime(data['date'])

# 汇总数据，计算每个日期和指标的人口总和
summary = data.groupby(['indicator_name', 'date'])['population'].sum().reset_index()

# 将人口单位转换为百万人
summary['population'] = summary['population'] / 1e6

# 绘图
plt.figure(figsize=(32, 24))
for label, df in summary.groupby('indicator_name'):
    plt.plot(df['date'], df['population'], label=label, linewidth=4)  # 加粗折线
    # 标记最高点
    max_population = df['population'].max()
    max_date = df[df['population'] == max_population]['date']
plt.title('Total number of behaviors in COVID-19', fontsize=48)
plt.xlabel('Date', fontsize=48)
plt.ylabel('Person times (million)', fontsize=48)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)

# 将图例放置在图表的底部
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=1, frameon=False, fontsize=38)
plt.grid(False)
plt.tight_layout()
plt.show()
