import pandas as pd

SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 3
RATIO_TO_PREDICT = 'LTC-USD'


def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0

# df = pd.read_csv('LTC-USD.csv', names=['time', 'low', 'hight', 'open', 'close', 'volume'])
main_df = pd.DataFrame()

ratios = ['BTC-USD', 'LTC-USD', 'ETH-USD', 'BCH-USD']
for ratio in ratios:
    dataset = f'{ratio}.csv'

    df = pd.read_csv(dataset, names=['time', 'low', 'hight', 'open', 'close', 'volume'])
    # print(df.head())
    df.rename(columns={'close': f'{ratio}_close', 'volume': f'{ratio}_volume'}, inplace=True)

    df.set_index('time', inplace=True)
    df = df[[f'{ratio}_close', f'{ratio}_volume']]

    # print(df.head())
    if len(main_df) == 0:
        main_df = df
    else:
        main_df = main_df.join(df)

# print(main_df.head())
#
# for c in main_df.columns:
#     print(c)


main_df['future'] = main_df[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_PERIOD_PREDICT)
print(main_df[[f'{RATIO_TO_PREDICT}_close', 'future']].head())

main_df['target'] = list(map(classify, main_df[f'{RATIO_TO_PREDICT}_close'], main_df['future']))
print(main_df[[f'{RATIO_TO_PREDICT}_close', 'future', 'target']].head())
