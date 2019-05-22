import pandas as pd
import numpy as np

def main():
    np.random.seed(1)
    df = pd.read_csv('cleveland_heart_disease.csv')
    train_indices = []
    test_indices = []
    for i in range(5):
        t = np.array(df.index[df['label'] == i])
        permutation = np.random.permutation(len(t))
        t = t[permutation]
        cutoff = int(len(t) // 5)
        train_indices.append(t[:-cutoff])
        test_indices.append(t[-cutoff: ])
    train_indices = np.concatenate(train_indices)
    test_indices = np.concatenate(test_indices)

    train_df = df.iloc[train_indices]
    test_df = df.iloc[test_indices]
    test_df.to_csv('heart_disease_test_GT.csv', index=False)
    del test_df['label']

    train_df.to_csv('heart_disease_train.csv', index=False)
    test_df.to_csv('heart_disease_test.csv', index=False)
    return


if __name__ == '__main__':
    main()