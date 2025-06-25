Sps I'm training a model, where the dataset is noisy (basically residual log return of a universe of 3000 stocks (changes everyday)

I use a day as a batch, and lookback is 121 days. My input is therefore [3000,121,6], 6 means 6 features per stock (log residual log return, volume, sell side volume, buy side volume, sell short volume). My target is next 10 day return (a scaler). My loss function is market cap weighted mse.

I want to use seq2seq model, specifically patchTST. However, I would like to merge the 6 variables into a "mixed" variable before I feed into patchTST (ie, we treat it like a univariable task by first mixing the variables), ie, [3000,121,6] into [3000,121,1]. Not: this somehow like an exogenous task

Should i:

1. Mix all 6 together into 1, then patch embedding on the mixed variable to get [3000, 6, number of patches, patch dim]  then transformer


2. Patch embedding, on each variable [3000, 6, number of patches, patch dim]  , mix on the patched result to get [3000, 1, number of patches, patch dim], then transformer



tell me what I should do and justify
