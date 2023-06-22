import pandas as pd

if __name__ == '__main__':
    dir_name = './dataset'
    dataset_name = 'yelp'
    target_dataset_name = dataset_name

    filename = f'{dataset_name}.bak.inter'
    sep = ','
    header = 0
    columns = 'user_id:token item_id:token rating:float timestamp:float'.split()

    k = 10

    df = pd.read_csv(f'{dir_name}/{dataset_name}/{filename}', header=header, sep=sep)
    print(df.head())
    # df.to_csv(f'{dir_name}/{target_dataset_name}/{dataset_name}.bak.inter', sep=sep, index=False)
    last_k = df.groupby('user_id:token').apply(lambda x: x.nlargest(k, 'timestamp:float'))
    print('Get last k interaction')
    last_k.index = last_k.index.droplevel(0)
    rest = df[~df.index.isin(last_k.index)]
    print('Get rest interaction')
    rest.to_csv(f'{dir_name}/{target_dataset_name}/{dataset_name}.inter', sep=sep, index=False)
    print('Save rest interaction')
    last_k.to_csv(f'{dir_name}/{target_dataset_name}/{dataset_name}.last{k}.inter', sep=sep, index=False)
    print('Save last k interaction')
    print('Done')
