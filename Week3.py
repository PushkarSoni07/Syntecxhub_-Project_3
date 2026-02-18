#Titanic dataset EDA
#Q1 Load Titanic dataset, inspect missingness and dtypes.
import pandas as pd
df = pd.read_csv("titanic.csv")

print("First 5 Rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset Info:")
print(df.info())

#Q2 Analyze survival rates by sex, class, age buckets.

import pandas as pd
df = pd.read_csv("titanic.csv")

print("Survival Rate by Sex:")
print(df.groupby("Sex")["Survived"].mean())

print("\nSurvival Rate by Class:")
print(df.groupby("Pclass")["Survived"].mean())

df["AgeGroup"] = pd.cut(df["Age"],
                        bins=[0,12,20,40,60,100],
                        labels=["Child","Teen","Adult","MiddleAge","Senior"])

print("\nSurvival Rate by Age Group:")
print(df.groupby("AgeGroup")["Survived"].mean())

#Q3 Visualize findings (bar charts,violin/boxplots).

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Titanic.csv")

survival_sex = df.groupby("Sex")["Survived"].mean()

plt.figure()
survival_sex.plot(kind="bar")
plt.title("Survival Rate by Sex")
plt.xlabel("Sex")
plt.ylabel("Survival Rate")
plt.show()


survival_class = df.groupby("Pclass")["Survived"].mean()

plt.figure()
survival_class.plot(kind="bar")
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.show()


plt.figure()
df.boxplot(column="Age", by="Survived")
plt.title("Age Distribution by Survival")
plt.suptitle("")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Age")
plt.show()


survived_age = df[df["Survived"] == 1]["Age"].dropna()
not_survived_age = df[df["Survived"] == 0]["Age"].dropna()

plt.figure()
plt.violinplot([not_survived_age, survived_age])
plt.xticks([1, 2], ["Not Survived", "Survived"])
plt.xlabel("Survival Status")
plt.ylabel("Age")
plt.title("Violin Plot of Age by Survival")
plt.show()

#Q4 Write a short insight report (3–5 bullets).

# Titanic EDA Insights:

# Female passengers had a much higher survival rate compared to males.
# First-class passengers had better survival chances than second and third class.
# Children had relatively higher survival rates than older passengers.
# Many missing values were found in the Age and Cabin columns.
# Higher socio-economic status (class) positively impacted survival probability.