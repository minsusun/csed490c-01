import re
import pandas as pd

prefixes = ['89-92', '97-107', '109-111', '113-116', '127-142', '144-147', '149-158']
postfix = ' ms'

for i in range(len(prefixes)):
    result = []
    for size in [512, 1024]:
        with open(f"./result_{size}") as f:
            s = f.read()
            elements = re.findall(f'{prefixes[i]+"] Elapsed time: "}(.*?){postfix}', s)
            result.append(elements)
    pd.DataFrame(result, columns = [f"testcase_{idx}" for idx in range(10)], index = ["512", "1024"]).to_csv(f'./exp_{i}.csv')