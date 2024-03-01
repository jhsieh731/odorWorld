import pandas as pd
import itertools
import numpy as np
import pickle as pkl


def find_combinations(df, target_vector, n):
    # Convert target_vector to a numpy array for easy comparison
    target_vector = np.array(target_vector)
    
    # Get all row indices in the DataFrame
    indices = range(len(df))
    
    # Store combinations that match the target_vector
    matching_combinations = []
    
    # Generate all combinations of n indices
    for combination in itertools.combinations(indices, n):
        # OR the vectors in the combination
        or_result = df.iloc[combination[0]].values
        for idx in combination[1:]:
            or_result = np.bitwise_or(or_result, df.iloc[idx].values)
        
        # Check if the result matches the target_vector
        if np.array_equal(or_result, target_vector):
            # If it matches, add the combination (as row numbers) to the results
            matching_combinations.append(combination)
    
    return matching_combinations

# Example usage
# Define your DataFrame here. For demonstration, I'll create a sample DataFrame.
# data = {
#     'A': [0, 1, 0, 1],
#     'B': [1, 1, 0, 0],
#     'C': [0, 0, 1, 0],
#     'D': [0, 0, 0, 1]
# }
# df = pd.read_csv('data_binary.csv').drop('Unnamed: 0', axis=1)
# df_trunc = df.drop(['index', 'Name', 'type', 'chems'], axis=1).dtypes('int64')

with open('binary_opens.pkl','rb') as readfile:
  df = pd.DataFrame(pkl.load(readfile))

# Target vector
target_vector = (df.iloc[5] + df.iloc[9]).values

# Number of vectors to OR
n = 2

# Find combinations
result = find_combinations(df, target_vector, n)
print("Matching combinations (as row indices):", result)