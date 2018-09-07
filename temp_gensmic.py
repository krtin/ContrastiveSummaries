import config
import pandas as pd
import numpy as np

smic_data = pd.read_csv(config.smiccorpus_filtered, usecols=['sourceid', 'smic', 'smic_id'], dtype={"sourceid": object, "smic": object, "smic_id": object})
smc_data = pd.read_csv(config.preprocessed_with_lm, usecols=['sourceid', 'source'], dtype={"sourceid": object, "source": object})
smc_data.drop_duplicates(subset='sourceid', inplace=True)


data = pd.merge(smic_data, smc_data, left_on='sourceid', right_on='sourceid', how='left')

#print(data)

print(len(smic_data), len(data), data['source'].count())


with open(config.smicfile_filtered_gtp, 'w') as f:
    f.write('\n'.join(list(data["smic"])))

with open(config.sourcefile_filtered_gtp, 'w') as f:
    f.write('\n'.join(list(data["source"])))

with open(config.smicidfile_filtered_gtp, 'w') as f:
    f.write('\n'.join(list(data["smic_id"])))
