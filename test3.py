import config

with open(config.smicidfile_filtered_gtp, 'r') as f:
    print(len(f.read().split('\n')))
