import pandas as pd
import datetime

from test_df_helper import pandas_show_all

pandas_show_all()

today = datetime.date.today().strftime('%Y%m%d')
df = pd.read_excel(f'./data/wta_tennis-data_{today}.xlsx')

print(df)