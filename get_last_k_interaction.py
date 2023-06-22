import pandas as pd

if __name__ == '__main__':
    dir_name = './dataset'
    dataset_name = 'ml-100k'
    target_dataset_name = 'ml-100k'

    filename = f'{dataset_name}.inter'
    sep = '\t'
    header = 0
    columns = 'user_id:token item_id:token rating:float timestamp:float'.split()

    k = 50

    df = pd.read_csv(f'{dir_name}/{dataset_name}/{filename}', header=header, sep=sep)
    df.to_csv(f'{dir_name}/{target_dataset_name}/{dataset_name}.bak.inter', sep=sep, index=False)
    last_k = df.groupby('user_id:token').apply(lambda x: x.nlargest(k, 'timestamp:float'))
    last_k.index = last_k.index.droplevel(0)
    rest = df[~df.index.isin(last_k.index)]

    rest.to_csv(f'{dir_name}/{target_dataset_name}/{filename}', sep=sep, index=False)
    last_k.to_csv(f'{dir_name}/{target_dataset_name}/{dataset_name}.last{k}.inter', sep=sep, index=False)
