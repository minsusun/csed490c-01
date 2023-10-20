import re
import pandas as pd

n_stream = [i+1 for i in range(32)]
prefixes = ['26-32] Elapsed time: ', '54-78] Elapsed time: ', '81-90] Elapsed time: ']
postfix = ' ms'

for i in range(len(prefixes)):
    result = []
    for n in n_stream:
        with open(f"./result_{n}") as f:
            s = f.read()
            elements = re.findall(f'{prefixes[i]}(.*?){postfix}', s)
            result.append(elements)
    pd.DataFrame(result, columns = [f"testcase_{idx}" for idx in range(10)], index = n_stream).to_csv(f'./exp{i+1}.csv')