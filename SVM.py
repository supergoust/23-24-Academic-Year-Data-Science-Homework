import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
city_data = pd.read_excel(r'C:\Users\12864\Desktop\数据科学大作业/筛选后城市.xlsx')
rural_data = pd.read_excel(r'C:\Users\12864\Desktop\数据科学大作业/筛选后农村.xlsx')

# 添加标签
city_data['label'] = 'city'
rural_data['label'] = 'rural'

# 合并数据
combined_data = pd.concat([city_data, rural_data])

# 提取特征和标签
X = combined_data.drop(columns=['year_month', 'label'])
y = combined_data['label']

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练集和测试集数量
train_size = len(X_train)
test_size = len(X_test)

# 训练SVM模型
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# 进行预测
y_pred = svm_model.predict(X_test)

# 评估模型
classification_report_result = classification_report(y_test, y_pred)
classification_report_dict = classification_report(y_test, y_pred, output_dict=True)

# 打印模型参数
model_params = svm_model.get_params()
print("Model Parameters:", model_params)

# 生成混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)

# 绘制混淆矩阵热力图
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['city', 'rural'], yticklabels=['city', 'rural'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix Heatmap')
plt.show()

# 打印评估结果
print("Classification Report:", classification_report_result)
print("Training set size:", train_size)
print("Testing set size:", test_size)
