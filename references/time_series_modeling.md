Use time series models to predict order number based on other features and chronicle trend

Models used:

1. Decomposition

Aggregated all data points into monthly basis, and utilized the seasonal_decomposition package by statsmodels to plot decomposed sections of the order trends by different attribute combinations.

Tried decomposition on:

- Daily level data of warehouse A
	
Trend data spiked around the end of 2015, and seasonal data are quite static.
	
- Daily level data of category Category_028
	
Trend data spiked around the end of 2015, and extremely repetitive pattern on seasonal data observed.
	
- Daily level data of product Product_0606(which is a mostly-ordered product in its category)
	
A generally ascending trend observed, indicating its growing product popularity. But seasonals are not that predictable.
	
- Monthly level data of product Product_0606(mostly-ordered product in its category)
	
An ascending trend, and seasonals are quite interesting: 3 spikes are observed respectively on every November, every February and every May.
	
- Monthly level data of product Product_1101(mostly-ordered product in its category)
	
A bell shape observed on trend, with kurtosis approximately during Dec 2014. For the seasonals, exremely regular pattern displayed: 4 spikes per year respectively on February, April, July and December, and 3 valleys on every March, June and September.
	
- Monthly level data of product Product_1361(mostly-ordered product in its category)
	
A 2-journey trend observed: continuously descending until Dec 2014, and then gradually ascending till now. For the seasonals, every May sees a yearly spike, following a yearly minimum in every July, quite fluctuating. Maybe the product is extremely dependent on seasonality.

2. ARIMA

Used the ARIMA model implemented in statsmodels package, to apply on aggregated monthly data of different attribute value combinations.

- used Augmented Dickey Fuller test, and found that the p-value is larger than 0.05 so differencing was needed
- tried an ARIMA model with p, d, q starting as 2, 1, 1 on product Product_0606 monthly data. Used the trained model to predict training data, and found it's not sensitive on spikes and valleys.
- tuned several sets of parameters and set p, d, q to be 3, 2, 1. Retrain model and use it to predict the test set. Predictions generally guesses the trend right, but not so good on extreme values: spikes and valleys.
- used the same set of parameter to fit monthly data of Product_0458 in warehouse Whse_S. Since test data does not contain abrupt spikes, the predictions are quite good, going not far alongside actual values.
- did the same training & prediction on the entire monthly data on warehouse Whse_S. The predictions are quite accurate about the descending trend, but lacking detailed zigzags as actual values display.
- finally tried the model on predicting monthly trend for category: Category_017. The model successfully predicts the ascending spike at May 2016(though with a different spike size)...but overestimates the order numbers in the following months.

3. Polynomial Regression

Tried linear regression(Polynomial) on the dataset, just to serve as a comparison. Used PolynomialFeatures by sklearn to wrap a linear regression layer outside a polynomial mapping.

I tried tuning the polynomial degree, and it's obvious that it's difficult for any polynomial mapping to fit the frequent zigzag patterns in time series dataset. Also, predictions by polynomial models are also smooth curves, meaning that it underfits the characteristic patterns presented by time series. Anyway linear regression here is only for reference.

Just come out with several thoughts here on improving linear regression:

- In order to fit better and learn more about the time series trend, the linear regression should take more into its independent variable, such as lagged difference. And this improvement would be the rudimentary thought of ARIMA or ARMA models.
- Another way of improvement is to integrate linear regression with decomposition, since the trend part in decomposition result is more often suitable for a polynomial fit. So the final model would be an ensemble of linear regression(polynomial) on trend, the seasonal and a residual noise. Then on the other hand, it's required to build another 2 models for seasonals and residuals respectively.

