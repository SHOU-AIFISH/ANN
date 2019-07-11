from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

data = pd.read_excel('数据集有氨氮的数据3.12.1.xlsx')
y = data['溶解氧'].values
x = data.drop(['溶解氧'], axis=1)
print(x)
# print(y)
scaler = StandardScaler().fit(x)
x = pd.DataFrame(scaler.transform(x)).values
# 设定训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# keras的序贯模型
model = Sequential()
# 搭建3层的BP神经网络的结构,units表示隐含层神经元数，input_dim表示输入层神经元数，
# activation表示激活函数
model.add(Dense(units=14, input_dim=x_train.shape[1], activation='sigmoid'))
model.add(Dense(units=1, input_dim=14, activation='sigmoid'))

# loss表示损失函数，这里损失函数为mse，优化算法采用Adam，metrics表示训练集的拟合误差
model.compile(loss='mse', optimizer=Adam(lr=0.01), metrics=['mape'])
model.summary()
# 将训练集的x和y输入到BP神经网络进行训练，epoch表示训练次数，
# batch_size表示每次训练的训练集大小，此处为24）使用sklearn对输入数据进行极大极小归一化。
# scaler = MinMaxScaler().fit(x)
history = model.fit(x_train, y_train, epochs=200, batch_size=8)

loss = history.history['loss']
epochs = range(len(loss))
plt.plot(epochs, loss, '-b', label='Train_loss')
plt.legend()

# 进行测试集的预测
result = model.predict(x_test, batch_size=1)
print(y_test)
print('测试集的预测结果为：', result)
#对预测结果和实际值进行可视化
plt.figure()
plt.plot(y_test, label='true data')
plt.plot(result, 'r:',label='predict')
plt.legend()