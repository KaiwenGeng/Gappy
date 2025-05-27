from arch import arch_model
from pykalman import KalmanFilter
import numpy as np
class Univarite_Volatility:
    """
    takes in a dataframe with daily excess return and/or log excess return
    output volatility estimates
    """
    def __init__(self, df, use_log_return):
        self.df = df
        self.use_log_return = use_log_return
    def estimate_volatility_GARCH(self, p=1, o=0, q=1):
        if self.use_log_return:
            r = (self.df['log excess return'].dropna()*100).astype(float)
        else:
            r = (self.df['daily excess return'].dropna()*100).astype(float)
        am   = arch_model(r, mean="Constant", vol="GARCH",
                        p=p, o=o, q=q)
        res  = am.fit(update_freq=0, disp="off")
        sigma = res.conditional_volatility       
        return sigma / 100.0
    
    def estimate_volatility_EWMA(self, lam=0.94):
        alpha = 1 - lam
        if self.use_log_return:
            r = self.df['log excess return'].dropna() * 100.0
        else:
            r = self.df['daily excess return'].dropna() * 100.0
        # compute EWMA of squared returns
        var = (r**2).ewm(alpha=alpha, adjust=False).mean()
        # take square root to get volatility
        return np.sqrt(var) / 100.0
    

    def estimate_volatility_Kalman(self,
                                    initial_vol=None,
                                    trans_cov=1e-6,
                                    obs_cov=1e-2):
        """
        Estimate volatility using a Kalman Filter on squared log returns.

        Parameters:
        - df: DataFrame with 'log return' column
        - initial_vol: initial estimate of variance (defaults to mean of squared returns)
        - trans_cov: process (transition) covariance
        - obs_cov: observation covariance

        Returns:
        - Series of estimated volatility (standard deviation)
        """
        # Prepare data
        if self.use_log_return:
            r = self.df['log excess return'].dropna() * 100.0
        else:
            r = self.df['daily excess return'].dropna() * 100.0
        y = (r ** 2).values  # squared returns as observations

        # Set initial state
        if initial_vol is None:
            initial_state_mean = y.mean()
        else:
            initial_state_mean = initial_vol

        # Build Kalman filter: state = variance_t, observation = variance_t + noise
        kf = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            initial_state_mean=initial_state_mean,
            initial_state_covariance=1.0,
            transition_covariance=trans_cov,
            observation_covariance=obs_cov,
        )

        # Run filtering
        state_means, state_covs = kf.filter(y)

        # Convert variance estimates to volatility
        sigma = np.sqrt(state_means.flatten())
        return sigma / 100.0