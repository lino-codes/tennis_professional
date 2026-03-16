import pandas as pd

from test_df_helper import pandas_show_all

pandas_show_all()

atp_files = [pd.read_excel('./data/atp_tennis-data_20251231.xlsx'),
             pd.read_excel('./data/atp_tennis-data_20241231.xlsx')]

wta_files = [pd.read_excel('./data/wta_tennis-data_20251231.xlsx'),
             pd.read_excel('./data/wta_tennis-data_20241231.xlsx')]


atp_df = pd.concat(atp_files, ignore_index=True).sort_values(by=['Date']).reset_index(drop=True)
wta_df = pd.concat(wta_files, ignore_index=True).sort_values(by=['Date']).reset_index(drop=True)


# print(wta_df.head())
# print(wta_df.tail())
atp_df.to_excel('./data/atp_tennis-data_2024-2025.xlsx')
wta_df.to_excel('./data/wta_tennis-data_2024-2025.xlsx')
