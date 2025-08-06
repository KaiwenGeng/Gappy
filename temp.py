So basically, for every day t, we have an universe of stocks (uni_t) with corresponding next 10 day residual return, which is a scaler. This is our target, and its shape will be [size of uni_t, 1]. As for the input, we get the historical feature data for the universe uni_t. Since our universe changes everyday, we unavoidable will have some 0 placeholders. In this case, our input will be shape [size of uni_t, 126, num_of_features] where 126 is our lookback window length.

Then we conduct the regression task using custom seq2seq backbone. The loss function can also be customly weighted, which is market cap is this case

Our data starts from 2012 and rolling window validation is conducted since 2017
