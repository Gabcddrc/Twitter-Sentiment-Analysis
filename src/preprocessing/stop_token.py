# Custom function to load pre-split train, validation and test data as pandas
# DataFrames.
import pandas as pd


def main():

  # Load token counts
  token_count = pd.read_csv('token_count.csv', index_col=0)

  # Calculate token summaries
  token_summary = token_count.copy()

  num_tokens = token_count['pos_count'].sum() + token_count['neg_count'].sum()

  token_summary['total_count'] = token_summary.apply(
      lambda row: row['pos_count'] + row['neg_count'],
      axis=1
  )

  token_summary['frequency'] = token_summary.apply(
      lambda row: (row['pos_count'] + row['neg_count']) / num_tokens,
      axis=1
  )

  token_summary['pos_ratio'] = token_summary.apply(
      lambda row: row['pos_count'] / (row['pos_count'] + row['neg_count']),
      axis=1
  )

  token_summary['neg_ratio'] = token_summary.apply(
      lambda row: row['neg_count'] / (row['pos_count'] + row['neg_count']),
      axis=1
  )

  token_summary['ratio_diff'] = token_summary.apply(
      lambda row: abs(row['pos_ratio'] - row['neg_ratio']),
      axis=1
  )


  # Non-discriminatory stop token candidates are:
  # 1) More than half a standard deviation above average frequency in the data set; and
  # 2) positive v. negative occurence ratio difference below a small threshold (e.g. 0.01)
  avg_frequency = token_summary['frequency'].mean()
  std_frequency = token_summary['frequency'].std() / 2
  ratio_diff_threshold = 0.01

  stop_token_candidates = token_summary[token_summary['frequency'] > avg_frequency + std_frequency]
  stop_token_candidates = stop_token_candidates[stop_token_candidates['ratio_diff'] < ratio_diff_threshold]
  stop_token_candidates = stop_token_candidates.sort_values('total_count', ascending=False)

  stop_token_candidates.to_csv('stop_token_candidates.csv', index=False)

if "__main__" == __name__:
  main()
