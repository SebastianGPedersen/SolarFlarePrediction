from _0_DataCreation.Read_Data import load_dataframe

fold1_df = load_dataframe(filename = 'fold1_NA_features.dat')
fold2_df = load_dataframe(filename = 'fold2_NA_features.dat')
fold3_df = load_dataframe(filename = 'fold3_NA_features.dat')
testset_df = load_dataframe(filename = 'testSet_NA_features.dat')

## Percentages of ones
perc_ones1 = sum(fold1_df['label']) / len(fold1_df) #16.35%
perc_ones2 = sum(fold2_df['label']) / len(fold2_df) #15.10%
perc_ones3 = sum(fold3_df['label']) / len(fold3_df) #17.66%
perc_total = (sum(fold1_df['label']) + sum(fold2_df['label']) + sum(fold3_df['label'])) / (len(fold1_df) + len(fold2_df) + len(fold3_df))

#The total percentage is 15.94% however the testset only has 12.64% in the public part



