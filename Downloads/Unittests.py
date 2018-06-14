#!/usr/bin/env python


# Used for creating our helper path
app_name = 'inferstat_prototype'

nan_policies = ['omit', 'raise', 'propagate']

# Basic stuff needed for all
try:
    import numpy as np
except:
    raise ImportError('NumPy not installed. Use "pip install numpy" to do so.')

try:
    import pandas as pd
except:
    raise ImportError('Pandas not installed. Use "pip install pandas" to do so.')

try:
    import unittest, sys
except:
    raise ImportError('Problem importing unittest or sys.')

if sys.version_info < (2, 7):
    print('need Python version 3 or greater to run this')
    raise AssertionError('Need python 3x')

_crosscorr = lambda x, y, lag=0: x.corr(y.shift(lag))

import warnings

warnings.filterwarnings("ignore")


def reducer_ratios(data, **kwargs):
    '''
    Takes a dataset list of dicts (date, series1, series2) and reduces the
    dimensionality to mean, stdev, skew, kurtosis & median
    Returns a dict of vectors
    {
       'nobs'      : 3,
       'minmax'    : (array([1,3]), array([3,5])),
       'mean'      : array([2. , 3.6666667]),
       'variance'  : array([1. , 1.3333333]),
       'skewness'  : array([0. , 0.70710678]),
       'kurtosis'  : array([-1.5, -1.5]),
    }
    Note: we are responsible for flattening to a dict of values.
    i.e. minmax is min_prx: x, min_rsch: y etc.
    '''
    fn = "reducer_ratios"
    from scipy import stats

    # This is a python3 module!
    if sys.version_info < (2, 7):
        print('need Python version 3 or greater to run this')
        sys.exit(0)

    reducer_suite = __name__.split('.')[-1]
    dsc = stats.describe(data[['research', 'price']])
    # is nobs / minmax([x,y],[x,y]) / mean(x,y) / variance(x,y) / skewness(x,y)
    # / kurtosis(x,y)
    output = {}
    price = stats.describe(data[['price']])
    research = stats.describe(data[['research']])

    # Ugly but it works
    output['prx_nobs'] = int(price.nobs)
    output['prx_min'] = float(price.minmax[0])
    output['prx_max'] = float(price.minmax[1])
    output['prx_mean'] = float(price.mean[0])
    output['prx_variance'] = float(price.variance[0])
    output['prx_skewness'] = float(price.skewness[0])
    output['prx_kurtosis'] = float(price.kurtosis[0])

    output['rsch_nobs'] = int(research.nobs)
    output['rsch_min'] = float(research.minmax[0])
    output['rsch_max'] = float(research.minmax[1])
    output['rsch_mean'] = float(research.mean[0])
    output['rsch_variance'] = float(research.variance[0])
    output['rsch_skewness'] = float(research.skewness[0])
    output['rsch_kurtosis'] = float(research.kurtosis[0])

    return output


def _breakout_dates(pt):
    '''
    Assumes that the datetime object has been converted. It will return output in the form of year, month and dayname.
    '''
    fn = "_breakout_dates"
    # index the date
    pt.index = pd.to_datetime(pt.index)
    pt['year'] = pt.index.year
    pt['month'] = pt.index.month
    pt['dayname'] = pt.index.weekday_name
    return pt


def _enrich_data(pt, lag=1):
    '''
    Will enrich the price/research series dataset by adding "LAG" changes &
    "LAG" % changes.
    Adds the following colums to the dataset where n is the lag:
    prx_dlt_n   : price delta
    rsch_dlt_n  : research delta
    prx_pc_n    : price delta percentage chage
    rsch_pc_n   : research delta percentage change
    '''
    pt['prx_dlt_{}'.format(lag)] = pt['price'] - pt['price'].shift(lag)
    pt['rsch_dlt_{}'.format(lag)] = pt['research'] - pt['research'].shift(lag)

    pt['prx_pc_{}'.format(lag)] = pt['prx_dlt_{}'.format(lag)] / pt['price']
    pt['rsch_pc_{}'.format(lag)] = pt['rsch_dlt_{}'.format(lag)] / pt['research']

    # Curious to know if we use n > 1 here what the calculations are
    return pt


def pd_reducer_ratios(pt, nan_policy='raise', **kwargs):
    '''
    Takes a pandas dataframe with s_date, price & research columns
    Top tip: use the _rs_to_ptbl(recordset) to convert your query (with the
    fields s_date, s_type and s_val) into a valid pandas pivot table :)
    input = [
        {'date':'2017-01-01', 'x':12.1, 'y':22  },
        {'date':'2017-01-02', 'x':13.7, 'y':32.2},
        {'date':'2017-01-03', 'x':11.7, 'y':12.8},
        ]
    Returns a dict.
    '''
    # axis==1 is column and axis==0 is row for all pandas operations requiring
    fn = "pandas_reducer_ratios"
    reducer_suite = __name__.split('.')[-1]

    try:
        from scipy import stats
        import pandas as pd
    except:
        raise ImportError('{} needs pandas and scipy')

    # Enforce our default in case it gets out of hand.
    if nan_policy not in nan_policies:
        raise AttributeError( \
            'nan_policy {} not accepted - try omit, raise or propagate'.format(
                nan_policy)
        )

    output = {}

    output['prx_mean'] = pt.price.mean()
    output['prx_kurtosis_st'] = stats.kurtosistest(pt.price, nan_policy=nan_policy)[0]
    output['prx_kurtosis_pv'] = stats.kurtosistest(pt.price, nan_policy=nan_policy)[1]
    output['prx_skewtest_st'] = stats.skewtest(pt.price, nan_policy=nan_policy)[0]
    output['prx_skewtest_pv'] = stats.skewtest(pt.price, nan_policy=nan_policy)[1]
    output['prx_corr'] = pt.price.corr(pt.research)
    output['rsch_mean'] = pt.research.mean()
    output['rsch_kurtosis_st'] = stats.kurtosistest(pt.research, nan_policy=nan_policy)[0]
    output['rsch_kurtosis_pv'] = stats.kurtosistest(pt.research, nan_policy=nan_policy)[1]
    output['rsch_skewtest_st'] = stats.skewtest(pt.research, nan_policy=nan_policy)[0]
    output['rsch_skewtest_pv'] = stats.skewtest(pt.research, nan_policy=nan_policy)[1]
    output['rsch_corr'] = pt.research.corr(pt.price)

    kur_ratio = float(output['prx_kurtosis_st']) / output['rsch_kurtosis_st']

    output['kurtosis_ratios'] = kur_ratio
    return output

    # documentation here for future reference
    # We can resample our index (convert to datetime if not already  with
    # pandas) and then do it.
    # e.g. sample = pd.to_datetime(pt.index)
    # From there we can do things like:
    # sample.resample('5D').mean() to get the mean of 5-day periods
    # sample.resample('1W').sum() to get the sum of 1-week periods
    # sample.resample('10B').ohlc() to get OHLC


def pd_reducer_correlations(pt, nan_policy='raise', **kwargs):
    '''
    Take a pandas dataframe containing 'date', 'price', and 'research' series
    Returns a dict of autocorrelation values
    # challenge was:
    Daily Correlations (5-Dimensional):
        Autocorrelation of 1-step % changes in Research series
        Autocorrelation of 1-step % changes in Price series
        Standard Deviation of 1-step % Changes in Research Series
        Standard Deviation of 1-step % Changes in Price Series
        Cross-correlation of 1-step changes in Price series to 1-step changes in Research Series
    '''
    fn = "reducer_correlations"
    reducer_suite = __name__.split('.')[-1]

    # This is a python3 module!
    if sys.version_info < (2, 7):
        print('need Python version 3 or greater to run this')
        sys.exit(0)

    # Getting real sciency now
    from scipy.stats import pearsonr

    # Should check our NaN policy and raise hell if there are NaN's
    if nan_policy == 'raise' and pt.isnull().values.any():
        raise AttributeError('NaN values in dataset and you asked for errors')

    lag = 1
    if lag in kwargs:
        lag = kwargs['lag']

    # Remove NaN  values - we could deal with these differently then
    # c_pt   = pt[['price','research']].dropna()

    output = {}

    # Auto-correlation
    output['rsch_corr'] = pt['research'].autocorr(lag=lag)
    output['prx_corr'] = pt['price'].autocorr(lag=lag)

    # Some cross correlation bits n bobs
    output['prx_rsch_xcorr'] = _crosscorr(pt['price'], pt['research'], lag=lag)
    output['prx_rsch_pearsoncoeff'] = pearsonr(pt['price'], pt['research'])[1]

    pt['prx_dlt'] = pt['price'] - pt['price'].shift(1)
    pt['rsch_dlt'] = pt['research'] - pt['research'].shift(1)
    pt['prx_pc'] = pt['prx_dlt'] / pt['price']
    pt['rsch_pc'] = pt['rsch_dlt'] / pt['research']

    # Autocorrelation of % changes in data
    output['rsch_pc_corr'] = pt['prx_pc'].autocorr(lag=lag)
    output['rsch_sigma_pc_corr'] = pt['rsch_pc'].std()
    output['prx_pc_corr'] = pt['rsch_pc'].autocorr(lag=lag)
    output['prx_sigma_pc_corr'] = pt['prx_pc'].std()

    return output


def getMean(df_concat):
    ks = pd_reducer_ratios(test_dataset)
    print(ks)
    for k, v in ks.items():
        print("k is {}, v is {}".format(k, v))
    return ks


class MainTest(unittest.TestCase):
    def setUp(self):
        # set up the environment needed for the test, in this case we do not need to use it
        pass

    def main_test(self):
        self.test_reducer_ratios()
        self.test_breakout_dates()
        self.test_enrich_data()
        self.test_pd_reducer_ratios_propagate()
        self.test_pd_reducer_ratios_omit()
        self.test_pd_reducer_correlations_without_lag()
        self.test_pd_reducer_correlations_with_lag()

    def test_reducer_ratios(self):
        df1 = [{'date': '2017-01-09', 'price': 12, 'research': 23}]
        df2 = [{'date': '2017-01-09', 'price': 13, 'research': 40}]
        df3 = pd.DataFrame(df1)
        df4 = pd.DataFrame(df2)
        df_concat = pd.concat((df3, df4))

        # Define expected parameters to be verified in the results
        params = ['prx_min', 'rsch_skewness', 'rsch_nobs', 'rsch_mean', 'prx_nobs', 'prx_max',
                  'rsch_variance', 'prx_variance', 'prx_kurtosis', 'rsch_max',
                  'rsch_min', 'rsch_kurtosis', 'prx_skewness', 'prx_mean']

        # Here set the values of the expected 'Reduced Ratios'
        expected_results = [
            {'prx_min': 12.0, 'rsch_skewness': 0.0, 'rsch_nobs': 2, 'rsch_mean': 31.5, 'prx_nobs': 2, 'prx_max': 13.0,
             'rsch_variance': 144.5, 'prx_variance': 0.5, 'prx_kurtosis': -2.0, 'rsch_max': 40.0, 'rsch_min': 23.0,
             'rsch_kurtosis': -2.0, 'prx_skewness': 0.0, 'prx_mean': 12.5}]
        expected_results_df = pd.DataFrame(expected_results)

        # Call the method that will return 'Reduced Ratios' of the series
        actual_results = reducer_ratios(df_concat)

        for i in range(len(params)):
            self.assertEqual(actual_results[params[i]], expected_results_df[params[i]].values)

    def test_breakout_dates(self):
        df1 = [{'date': '2017-01-09'}]
        df2 = pd.DataFrame(df1)
        final_df = _breakout_dates(df2)

        date = [{"date": "2017-01-09", "year": 1970, "month": 1, "dayname": 'Thursday'}]
        final_date = pd.DataFrame(date)

        self.assertEqual(final_df['date'].values, final_date['date'].values)
        self.assertEqual(final_df['year'].values, final_date['year'].values)
        self.assertEqual(final_df['month'].values, final_date['month'].values)
        self.assertEqual(final_df['dayname'].values, final_date['dayname'].values)

    def test_enrich_data(self):
        df1 = df1 = [{'date': '2017-01-09', 'price': 56, 'research': 76}]
        df2 = pd.DataFrame(df1)
        final_df = _enrich_data(df2, 0)

        # Create an expected data frame
        expected_date = [
            {'date': '2017-01-09', 'price': 56, 'research': 76, 'prx_dlt_0': 0, 'rsch_dlt_0': 0, 'prx_pc_0': 0.0,
             'rsch_pc_0': 0.0}]
        expected_date_df = pd.DataFrame(expected_date)

        # Compare expected data frame with the one returned from _enrich_data function
        self.assertEqual(final_df['rsch_dlt_0'].values, expected_date_df['rsch_dlt_0'].values)
        self.assertEqual(final_df['prx_pc_0'].values, expected_date_df['prx_pc_0'].values)
        self.assertEqual(final_df['prx_dlt_0'].values, expected_date_df['prx_dlt_0'].values)
        self.assertEqual(final_df['rsch_pc_0'].values, expected_date_df['rsch_pc_0'].values)
        self.assertEqual(final_df['price'].values, expected_date_df['price'].values)
        self.assertEqual(final_df['research'].values, expected_date_df['research'].values)

    def test_pd_reducer_ratios_propagate(self):  # n>20
        # Define expected parameters to be verified in the results
        params = ['prx_corr', 'rsch_skewtest_st', 'rsch_kurtosis_pv', 'prx_kurtosis_st', 'rsch_kurtosis_st',
                  'rsch_corr', 'rsch_mean', 'prx_skewtest_st', 'prx_kurtosis_pv', 'kurtosis_ratios', 'prx_mean']

        df1 = [{'date': '2017-01-09', 'price': 56, 'research': 76}, {'date': '2017-01-09', 'price': 42, 'research': 7},
               {'date': '2017-01-09', 'price': 66, 'research': 23}, {'date': '2017-01-09', 'price': 53, 'research': 6},
               {'date': '2017-01-09', 'price': 16, 'research': 76}, {'date': '2017-01-09', 'price': 23, 'research': 2},
               {'date': '2017-01-09', 'price': 86, 'research': 76}, {'date': '2017-01-09', 'price': 98, 'research': 16},
               {'date': '2017-01-09', 'price': 56, 'research': 76}, {'date': '2017-01-09', 'price': 42, 'research': 10},
               {'date': '2017-01-09', 'price': 66, 'research': 46}, {'date': '2017-01-09', 'price': 53, 'research': 76},
               {'date': '2017-01-09', 'price': 16, 'research': 76}, {'date': '2017-01-09', 'price': 23, 'research': 76},
               {'date': '2017-01-09', 'price': 86, 'research': 76}, {'date': '2017-01-09', 'price': 98, 'research': 76},
               {'date': '2017-01-09', 'price': 56, 'research': 74}, {'date': '2017-01-09', 'price': 42, 'research': 89},
               {'date': '2017-01-09', 'price': 66, 'research': 70}, {'date': '2017-01-09', 'price': 53, 'research': 76},
               {'date': '2017-01-09', 'price': 16, 'research': 89}, {'date': '2017-01-09', 'price': 23, 'research': 76},
               {'date': '2017-01-09', 'price': 86, 'research': 76}, {'date': '2017-01-09', 'price': 98, 'research': 76}]

        df2 = pd.DataFrame(df1)
        # Call pd_reducer_ratios with param 'propagate'
        final_result_with_propagate = pd_reducer_ratios(df2, 'propagate')

        # Create expected results here
        expected_df = [{'prx_corr': 0.0013989220326844523, 'rsch_skewtest_st': -2.2329682992553517,
                        'rsch_kurtosis_pv': 0.5667471992545061, 'rsch_skewtest_pv': 0.025551034945657296,
                        'prx_skewtest_pv': 0.7938954344224488, 'prx_kurtosis_st': -1.4789722682295174,
                        'rsch_kurtosis_st': -0.572848614924687, 'rsch_corr': 0.0013989220326844523,
                        'rsch_mean': 59.166666666666664, 'prx_skewtest_st': 0.2612555603501589,
                        'prx_kurtosis_pv': 0.1391477267157739, 'kurtosis_ratios': 2.5817855358242587, 'prx_mean': 55.0}]
        expected_final_df = pd.DataFrame(expected_df)

        # Compare actual and expected results
        for i in range(len(params)):
            if ('nan' in expected_final_df[params[i]].values):
                self.assertEqual(repr(final_result_with_propagate[params[i]]), expected_final_df[params[i]].values)
            else:
                # print type(np.asscalar(final_result_with_propagate[params[i]]))
                print (round(np.asscalar(final_result_with_propagate[params[i]]),3))
                print (round(np.asscalar(expected_final_df[params[i]].values),3))
                self.assertEqual(round(np.asscalar(final_result_with_propagate[params[i]]),3), round(np.asscalar(expected_final_df[params[i]].values),3))

    def test_pd_reducer_ratios_omit(self):  # n=8
        # Define expected parameters to be verified in the results
        params = ['prx_corr', 'rsch_skewtest_st', 'rsch_kurtosis_pv', 'prx_kurtosis_st', 'rsch_kurtosis_st',
                  'rsch_corr', 'rsch_mean', 'prx_skewtest_st', 'prx_kurtosis_pv', 'kurtosis_ratios', 'prx_mean']

        df1 = [{'date': '2017-01-09', 'price': 56, 'research': 76}, {'date': '2017-01-09', 'price': 42, 'research': 76},
               {'date': '2017-01-09', 'price': 66, 'research': 76}, {'date': '2017-01-09', 'price': 53, 'research': 76},
               {'date': '2017-01-09', 'price': 16, 'research': 76}, {'date': '2017-01-09', 'price': 23, 'research': 76},
               {'date': '2017-01-09', 'price': 86, 'research': 76}, {'date': '2017-01-09', 'price': 98, 'research': 76}]
        df2 = pd.DataFrame(df1)
        # Call pd_reducer_ratios with param 'omit'
        final_result_with_omit = pd_reducer_ratios(df2, 'omit')

        # Create expected results here
        expected_df = [{'prx_corr': 'nan', 'rsch_skewtest_st': 1.0108048609177787,
                        'rsch_kurtosis_pv': 2.7473248254639826e-35,
                        'rsch_skewtest_pv': 0.3121098361421897, 'prx_skewtest_pv': 0.8545126767991549,
                        'prx_kurtosis_st': -0.47416376260462284, 'rsch_kurtosis_st': -12.395989756431076,
                        'rsch_corr': 'nan',
                        'rsch_mean': 76.0, 'prx_skewtest_st': 0.1833636736772389, 'prx_kurtosis_pv': 0.6353831321467589,
                        'kurtosis_ratios': 0.03825138386861164, 'prx_mean': 55.0}]
        final_expected_df = pd.DataFrame(expected_df)

        # Compare actual and expected results
        for i in range(len(params)):
            if ('corr' in params[i]):
                self.assertEqual(repr(final_result_with_omit[params[i]]), final_expected_df[params[i]].values)
            else:
                self.assertEqual(final_result_with_omit[params[i]], final_expected_df[params[i]].values)

    def test_pd_reducer_correlations_without_lag(self):
        # Define expected parameters to be verified in the results
        params = ['prx_pc_corr', 'prx_sigma_pc_corr', 'prx_corr', 'prx_rsch_xcorr', 'prx_rsch_pearsoncoeff',
                  'rsch_corr', 'rsch_pc_corr', 'rsch_sigma_pc_corr']

        df1 = [{'date': '2017-01-09', 'price': 56, 'research': 76}, {'date': '2017-01-09', 'price': 42, 'research': 76},
               {'date': '2017-01-09', 'price': 66, 'research': 76}, {'date': '2017-01-09', 'price': 53, 'research': 76},
               {'date': '2017-01-09', 'price': 16, 'research': 76}, {'date': '2017-01-09', 'price': 23, 'research': 76},
               {'date': '2017-01-09', 'price': 86, 'research': 76}, {'date': '2017-01-09', 'price': 98, 'research': 76}]
        df2 = pd.DataFrame(df1)
        final_result_reducer_correlations = pd_reducer_correlations(df2, 'raise')
        expected_reducer_correlations = [
            {'prx_pc_corr': 'nan', 'prx_sigma_pc_corr': 1.0023331235194493, 'prx_corr': 0.33174948574689733,
             'prx_rsch_xcorr': 'nan', 'prx_rsch_pearsoncoeff': 1.0, 'rsch_corr': 'nan',
             'rsch_pc_corr': -0.04891273014488685, 'rsch_sigma_pc_corr': 0.0}]
        expected_df_reducer_correlations = pd.DataFrame(expected_reducer_correlations)

        for i in range(len(params)):
            if (params[i] in ("prx_pc_corr", "prx_rsch_xcorr", "rsch_corr")):
                self.assertEqual(repr(final_result_reducer_correlations[params[i]]),
                                 expected_df_reducer_correlations[params[i]].values)
            else:
                self.assertEqual(final_result_reducer_correlations[params[i]],
                                 expected_df_reducer_correlations[params[i]].values)

    def test_pd_reducer_correlations_with_lag(self):
        # Define expected parameters to be verified in the results
        params = ['prx_pc_corr', 'prx_sigma_pc_corr', 'prx_corr', 'prx_rsch_xcorr', 'prx_rsch_pearsoncoeff',
                  'rsch_corr', 'rsch_pc_corr', 'rsch_sigma_pc_corr']

        df1 = [{'date': '2017-01-09', 'price': 56, 'research': 76}, {'date': '2017-01-09', 'price': 42, 'research': 76},
               {'date': '2017-01-09', 'price': 66, 'research': 76}, {'date': '2017-01-09', 'price': 53, 'research': 76},
               {'date': '2017-01-09', 'price': 16, 'research': 76}, {'date': '2017-01-09', 'price': 23, 'research': 76},
               {'date': '2017-01-09', 'price': 86, 'research': 76}, {'date': '2017-01-09', 'price': 98, 'research': 76}]
        df2 = pd.DataFrame(df1)
        final_result_reducer_correlations_with_lag = pd_reducer_correlations(df2, 'raise', lag=20)
        expected_reducer_correlations = [
            {'prx_pc_corr': 'nan', 'prx_sigma_pc_corr': 1.0023331235194493, 'prx_corr': 0.33174948574689733,
             'prx_rsch_xcorr': 'nan', 'prx_rsch_pearsoncoeff': 1.0, 'rsch_corr': 'nan',
             'rsch_pc_corr': -0.04891273014488685, 'rsch_sigma_pc_corr': 0.0}]
        expected_df_reducer_correlations = pd.DataFrame(expected_reducer_correlations)

        for i in range(len(params)):
            if (params[i] in ("prx_pc_corr", "prx_rsch_xcorr", "rsch_corr")):
                self.assertEqual(repr(final_result_reducer_correlations_with_lag[params[i]]),
                                 expected_df_reducer_correlations[params[i]].values)
            else:
                self.assertEqual(final_result_reducer_correlations_with_lag[params[i]],
                                 expected_df_reducer_correlations[params[i]].values)

    def tearDown(self):
        # clean up after the test, in this case we do not need to use it
        pass


if __name__ == '__main__':
    test_dataset = [
        {'date': '2017-01-09', 'price': 12, 'research': 23},
        {'date': '2017-01-09', 'price': 13, 'research': 23},
        {'date': '2017-01-09', 'price': 18, 'research': 23},
        {'date': '2017-01-09', 'price': 19, 'research': 23},
        {'date': '2017-01-09', 'price': 11, 'research': 23},
        {'date': '2017-01-09', 'price': 22, 'research': 23},
        {'date': '2017-01-09', 'price': 32, 'research': 23},
        {'date': '2017-01-09', 'price': 82, 'research': 23},
        {'date': '2017-01-09', 'price': 22, 'research': 23},
    ]
    test_panda = pd.DataFrame(data=test_dataset)

    # See if it can process the above Panda.
    # TODO3 - Currently not doing an in file test, other than running the MainTest object.
    # getMean(test_panda)'''

    # See if the unit test also works
    # testObject = MainTest()
    # testObject.main_test()

    unittest.main()
    print('Tests completed successfully.')