from datetime import datetime
import pandas as pd


def df_processor(df: pd.DataFrame) -> pd.DataFrame:
    df['driven_per_year'] = df.km_driven/(datetime.now().year - df.year)
    df['power_per_liter'] = df.max_power / df.engine
    df['year_squared'] = df.year**2
    df['rpm_per_torque'] = df.max_torque_rpm / df.torque
    df['brand'] = df.name.apply(lambda x: x.split(' ')[0])
    df.drop(['name'], axis=1, inplace=True)

    return df
