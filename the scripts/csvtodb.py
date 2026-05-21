import pandas as pd
from sqlalchemy import create_engine

# read csv
df = pd.read_csv('scripts/data_biceps_clean.csv')

# koneksi mysql
engine = create_engine("mysql+pymysql://metabase_user:admin@localhost/db_capstone")

# kirim
df.to_sql("dataBiceps", engine, if_exists="replace", index=False)

print("Success upload")