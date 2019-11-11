A data science project on the manufacturing product demand dataset to explore the seasonal trend in features and build time series model to predict the product demands.

# Feature explorative ideas
- Daily Trend(no aggregation, each data on daily level)
1. Explore the order trend of single feature value
	* number of order on a certain category(all 32 product categories)
	* number of order on a certain warehouse(all 4 warehouses)
2. Explore the order trend of single product entry
	* number of order on some single product entries(choose some random products to look at)
	* number of order on the mostly-ordered product entry(choose 3 mostly-ordered product in its category)
	* plot autocorrelation & partial autocorrelation for the mostly-ordered products


- Monthly Trend(aggregating raw data into monthly basis)
1. Explore the monthly order trend of 3 mostly-ordered product
2. Compare order trends between attribute values
	* compare order trends among different categories(among all 32 product categories)
	* compare order trends among different warehouses(among all 4 warehouses)
3. Compare order trends between single product entries
	* select a few product entries, and compare the order trends of the product under different warehouses
 
# Use time series models to predict order number based on other features and chronicle trend
1. Decomposition

Aggregated all data points into monthly basis, and utilized the seasonal_decomposition package by statsmodels to plot decomposed sections of the order trends by different attribute combinations.

Tried decomposition on:

	* Daily level data of warehouse A
	
	* Daily level data of category Category_028
	
	* Daily level data of product Product_0606(which is a mostly-ordered product in its category)
	
	* Monthly level data of product Product_0606(mostly-ordered product in its category)
	
	* Monthly level data of product Product_1101(mostly-ordered product in its category)
	
	* Monthly level data of product Product_1361(mostly-ordered product in its category)
	
	
2. ARIMA

Used the ARIMA model implemented in statsmodels package, to apply on aggregated monthly data of different attribute value combinations.

	* used Augmented Dickey Fuller test, and found that the p-value is larger than 0.05 so differencing was needed
	
	* tried an ARIMA model with p, d, q starting as 2, 1, 1 on product Product_0606 monthly data. Used the trained model to predict training data, and found it's not sensitive on spikes and valleys.
	
	* tuned several sets of parameters and set p, d, q to be 3, 2, 1. Retrain model and use it to predict the test set. Predictions generally guesses the trend right, but not so good on extreme values: spikes and valleys.
	
	* used the same set of parameter to fit monthly data of Product_0458 in warehouse Whse_S. Since test data does not contain abrupt spikes, the predictions are quite good, going not far alongside actual values.
	
	* did the same training & prediction on the entire monthly data on warehouse Whse_S. The predictions are quite accurate about the descending trend, but lacking detailed zigzags as actual values display.
	
	* finally tried the model on predicting monthly trend for category: Category_017. The model successfully predicts the ascending spike at May 2016(though with a different spike size)...but overestimates the order numbers in the following months.
	

3. Polynomial Regression

Tried linear regression(Polynomial) on the dataset, just to serve as a comparison. Used PolynomialFeatures by sklearn to wrap a linear regression layer outside a polynomial mapping.

I tried tuning the polynomial degree, and it's obvious that it's difficult for any polynomial mapping to fit the frequent zigzag patterns in time series dataset. Also, predictions by polynomial models are also smooth curves, meaning that it underfits the characteristic patterns presented by time series. Anyway linear regression here is only for reference.

Just come out with several thoughts here on improving linear regression:

- In order to fit better and learn more about the time series trend, the linear regression should take more into its independent variable, such as lagged difference. And this improvement would be the rudimentary thought of ARIMA or ARMA models.
- Another way of improvement is to integrate linear regression with decomposition, since the trend part in decomposition result is more often suitable for a polynomial fit. So the final model would be an ensemble of linear regression(polynomial) on trend, the seasonal and a residual noise. Then on the other hand, it's required to build another 2 models for seasonals and residuals respectively.

