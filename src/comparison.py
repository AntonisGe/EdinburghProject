import pandas as pd

path1 = r'D:\Project\code\Weights\SSA\long_10_back_to_bigger_switch\results.csv'
path2 = r'D:\Project\code\Weights\SSA\long_11_long_agent\results.csv'

df1 = pd.read_csv(path1)
df2 = pd.read_csv(path2)

df = pd.DataFrame()

df['Mine'] = df1.Validations
df['Long'] = df2.Validations