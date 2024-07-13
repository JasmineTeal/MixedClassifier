import pandas as pd
from sklearn.model_selection import train_test_split

# 加载数据
file_path = 'result_data.csv'
data = pd.read_csv(file_path)

# 选择20%数据作为测试集，80%作为训练集
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# 保存分割后的数据到新的文件中
train_file_path = 'result_data_train.csv'
test_file_path = 'result_data_test.csv'

train_data.to_csv(train_file_path, index=False)
test_data.to_csv(test_file_path, index=False)

print(f"Training data saved to {train_file_path}")
print(f"Testing data saved to {test_file_path}")
