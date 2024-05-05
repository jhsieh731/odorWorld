# Demonstrates that keeping data as-is (fixed distribution) is too simple of a task: solved below by dp

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

def optimized_search(df, target_vector, n):
    target_vector = np.array(target_vector)
    df_vectors = df.drop(columns=['Name']).to_numpy()
    names = df['Name'].to_numpy()
    results = []

    def search(combination, index, current_or):
        # Pruning condition: if current combination OR result is already not matching the target
        if np.any(np.bitwise_and(current_or, target_vector) != current_or):
            return
        
        # If combination size is n, check if it matches the target_vector exactly
        if len(combination) == n:
            if np.array_equal(current_or, target_vector):
                results.append([names[i] for i in combination])
            return

        # Avoid going beyond the dataframe's rows
        if index >= len(df_vectors):
            return
        
        # Explore further with or without the current index
        search(combination + [index], index + 1, np.bitwise_or(current_or, df_vectors[index]))
        search(combination, index + 1, current_or)

    # Initialize the search
    search([], 0, np.zeros_like(target_vector))

    return results

def optimized_search_sum(df, target_vector, n):
    target_vector = np.array(target_vector)
    df_vectors = df.drop(columns=['Name']).to_numpy()
    names = df['Name'].to_numpy()
    results = []

    def search(combination, index, current_sum):
        # Pruning condition: if current sum exceeds target_vector in any dimension, return
        if np.any(current_sum > target_vector):
            return
        
        # If combination size is n, check if it matches the target_vector exactly
        if len(combination) == n:
            if np.array_equal(current_sum, target_vector):
                # results.append([names[i] for i in combination])
                results.append([i for i in combination])

            return

        # Avoid going beyond the dataframe's rows
        if index >= len(df_vectors):
            return
        
        # Explore further with or without the current index
        search(combination + [index], index + 1, current_sum + df_vectors[index])
        search(combination, index + 1, current_sum)

    # Initialize the search with an empty sum vector
    search([], 0, np.zeros_like(target_vector))

    return results


# Load data
df = pd.read_csv('data_binary.csv').drop(['Unnamed: 0', 'index', 'chems', 'type'], axis=1)

chems = list(df.columns)
chems.remove('Name')
df[chems] = df[chems].astype(int)


def test(m):
    num_tests = 10
    running_accuracy = 0.0
    for _ in range(num_tests):
        corrects = [random.randrange(0, 500) for _ in range(m)]
        target_vector = np.zeros(1781)
        for c in corrects:
            target_vector += df[chems].iloc[c].values
        result = optimized_search_sum(df, target_vector, m)

        if not result: continue
        # fix index issue where set lengths diff
        overlap = len(set(result[0]) & set(corrects))

        running_accuracy += overlap / m
    print('finished testing for', m, 'odors')
    return running_accuracy/num_tests

m = [i for i in range(2, 5)]
accuracies = [test(i) for i in m]
std = [np.sqrt(p*(1-p)/10) for p in accuracies]

plt.errorbar(m, accuracies, yerr=std, linewidth=2)
plt.plot(m, accuracies)
plt.xlabel('Number of odors')
plt.xticks([2, 3, 4])
plt.xlim(2, 4)
plt.ylim(0, 1.2)
plt.ylabel('Accuracy')
plt.title('Accuracy of optimized search algorithm')

# save to file
plt.savefig('optimized_search_accuracy_234.png')
        # target_vector_sum = sum(df[chems].iloc[idx1].values, df[chems].iloc[idx2].values)
    # Define target vectors with arbitrary indices
    # idx1, idx2 = 102, 298
    # print("Correct result (as names):", f"{df['Name'][idx1]}, {df['Name'][idx2]}")
    # target_vector = np.bitwise_or(df[chems].iloc[idx1].values, df[chems].iloc[idx2].values)
    # target_vector_sum = df[chems].iloc[idx1].values + df[chems].iloc[idx2].values

    # # Number of vectors to OR/SUM
    # n = 3

    # # Find combinations
    # result = optimized_search(df, target_vector, n)
    # print("Matching combinations (as names):", result)

    # result = optimized_search_sum(df, target_vector_sum, n)
    # print("Matching combinations (as names):", result)