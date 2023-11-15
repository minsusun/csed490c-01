import re
import pandas as pd

prefixes = ['58-62', '67-72', '74-78', '83-88', '90-94', '96-100']
postfix = ' ms'

for i in range(len(prefixes)):
    result = []
    for size in [256, 512, 1024]:
        with open(f"./result_{size}") as f:
            s = f.read()
            elements = re.findall(f'{prefixes[i]+"] Elapsed time: "}(.*?){postfix}', s)
            result.append(elements)
    pd.DataFrame(result, columns = [f"testcase_{idx}" for idx in range(8)], index = ["256", "512", "1024"]).to_csv(f'./exp_{i}.csv')