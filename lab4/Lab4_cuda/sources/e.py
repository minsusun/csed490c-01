import re
import pandas as pd

prefixes = ['118-123', '125-129', '131-139', '141-146', '116-148']
postfix = ' ms'

for i in range(len(prefixes)):
    result = []
    with open(f"./result") as f:
        s = f.read()
        elements = re.findall(f'{prefixes[i]+"] Elapsed time: "}(.*?){postfix}', s)
        result.append(elements)
    pd.DataFrame(result, columns = [f"testcase_{idx}" for idx in range(7)], index = ["-"]).to_csv(f'./exp_{i}.csv')