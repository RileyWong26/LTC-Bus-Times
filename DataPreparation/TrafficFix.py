import pandas as pd

# ------------- IMPORT --------------

# df = pd.read_csv("test.csv")
# print(df)

# for row in df.itertuples():
#     if "\n" in row.Traffic:
#         split = row.Traffic.split("\n")
#         df.loc[row.Index, "Traffic"] = split[0]

# df = df.drop(columns=['Unnamed: 0'], axis=1)
# df.to_csv("test2.csv",index=False)

df = pd.read_csv("./Data/OldFiles/102_trafficbackup.csv")

df['Traffic'] = df['Traffic'].replace("Series([], )", "0.0").astype(float)
print(df.dtypes)

df = df.astype(float)
print(df.dtypes)

df.to_csv('./Data/CondensedDataFiles/102_traffic.csv',index=False)