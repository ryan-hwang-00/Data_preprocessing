import seaborn as sns
import matplotlib.pylab as plt

# 데이터 전체에 대한 히스토그램
data.hist(figsize=(14,14), bins=20)
plt.show()

# 성별을 비교값으로 설정하여 barplot
sns.barplot(x="Pclass", y="Survived", hue ='Sex', data=data)

# 데이터에 대한 히트맵(상관관계) 표시
plt.subplots(figsize=(10, 10))
sns.heatmap(data.corr(), annot=True, linewidths=.2)