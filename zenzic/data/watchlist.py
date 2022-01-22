import pymysql as mysql
import pandas as pd

class SP500():
    db_conn = None

    def __init__(self):
        if SP500.db_conn is None:
            SP500.db_conn = mysql.connect(read_default_file="~/my.cnf")

    def get_symbols(self):
        query = """
                    SELECT symbol AS symbol,
                           finviz_info->>'$.\"Company Name\"' as company_name,
                           finviz_info->>'$.\"Avg Volume\"' as avg_volume
                    FROM symbols JOIN stock_info USING(sym_id)
                    WHERE finviz_info->>'$.Index' LIKE '%S&P%'
                          AND u_date = (SELECT MAX(u_date) FROM stock_info)
                    ORDER BY company_name
                """
        data = pd.read_sql(query, SP500.db_conn, index_col='company_name')
        data['avg_volume'] = data['avg_volume'].apply(lambda x: self.__covert_vol(x))
        symbols = []
        for c in data.index.unique():
            max_vol = data.loc[c]['avg_volume'].max()
            symbols.append(data[data['avg_volume'] == max_vol].loc[c]['symbol'])
        return sorted(symbols)

    def __covert_vol(self, vol):
        if vol == '-':
            return 0
        elif vol[-1] == 'M':
            return float(vol[:-1]) * 1000000
        elif vol[-1] == 'K':
            return float(vol[:-1]) * 1000
        else:
            return float(vol[:-1])
