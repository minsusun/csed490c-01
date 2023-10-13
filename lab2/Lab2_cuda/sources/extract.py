import re
import pandas as pd

tile_width = [2, 4, 8, 12, 16, 24, 32]
prefixes = ['70-80\] Elapsed time: ', '85-91] Elapsed time: ', '93-98] Elapsed time: ', '105-110] Elapsed time: ', '112-116] Elapsed time: ', '118-124] Elapsed time: ']
postfix = ' ms'

for i in range(len(prefixes)):
    result = []
    for width in tile_width:
        with open(f"./result_{width}") as f:
            s = f.read()
            elements = re.findall(f'{prefixes[i]}(.*?){postfix}', s)
            result.append(elements)
    pd.DataFrame(result, columns = [f"testcase_{idx}" for idx in range(9)], index = tile_width).to_csv(f'./exp{i+1}.csv')