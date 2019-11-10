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
