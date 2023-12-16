import re
import pandas as pd

prefixes = ['71-81', '154-164', '166-175', '181-186', '188-192', '195-205']
postfix = ' ms'

for i in range(len(prefixes)):
    result = []
    with open(f"./../result_wo_staging") as f:
        s = f.read()
        elements = re.findall(f'{prefixes[i]+"] Elapsed time: "}(.*?){postfix}', s)
        result.append(elements)
    pd.DataFrame(result, columns = [f"testcase_{idx}" for idx in range(9)], index = [""]).to_csv(f'./exp_{i}.csv')