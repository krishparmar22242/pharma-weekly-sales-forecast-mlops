# Pharmaceutical Sales Forecasting — Technical Interview Guide

This guide explains the technical reasoning behind the project. It is for a student who implemented the project and may need to defend each choice in a technical interview.

No implementation code is included.

## 1. Project problem

The project predicts the total pharmaceutical sales for the next week.

The original data contains eight weekly drug-category columns:

- M01AB
- M01AE
- N02BA
- N02BE
- N05B
- N05C
- R03
- R06

These values were summed to create one target column called `Total_Sales`.

The goal was to determine whether recent sales history could predict the next week's total sales and to compare simple statistical forecasting with a machine learning model.

## 2. Dataset used

The main dataset is `salesweekly.csv`.

Important details:

- It has 302 weekly records.
- The first column contains dates.
- The remaining columns represent eight pharmaceutical categories.
- The date column was converted to a date type.
- Records were sorted chronologically.
- The eight category values were summed to calculate total weekly sales.
- Rows with invalid or missing target values could be removed or handled during preparation.

The project did not forecast all eight categories separately. It forecasted the aggregate weekly total.

An interview answer should clearly distinguish between:

- Features
- Target
- Historical time-series order
- Actual and forecast values

In this project:

- Target: next or current week's total sales
- Features: lag sales and rolling averages
- Data order: date ascending

## 3. Why time-series forecasting was used

The target is measured repeatedly over time.

Forecasting is appropriate because:

- Past observations may predict future observations.
- Sales are likely to contain trend or seasonal patterns.
- The objective is a future value, not a normal classification label.
- The test period must occur after the training period.

The main forecasting challenge is that observations are not independent. One week's sales are related to earlier weeks.

## 4. Why forecasting models were selected

The project used both classical and machine learning forecasting approaches.

### Holt-Winters

Holt-Winters was selected as a simple seasonal baseline.

It is useful because it can model:

- Level
- Trend
- Seasonality

Level represents the current sales baseline.

Trend represents whether sales are generally increasing or decreasing.

Seasonality represents regular patterns that repeat over time, such as yearly patterns.

It was a reasonable starting model because weekly sales often show repeating patterns.

Its limitation is that it may not capture complex relationships or sudden changes in sales.

### ARIMA

ARIMA was selected because it is a standard statistical model for stationary time series.

The name means:

- AR: relationship with past values
- I: differencing to reduce trend and increase stationarity
- MA: relationship with previous forecast errors

The project used ARIMA(1,1,1).

This means:

- One previous observation is used in the autoregressive term.
- The series is differenced once.
- One previous forecast error is used in the moving-average term.

ARIMA is useful when:

- The series is related to previous values.
- Differencing removes trend.
- The pattern is not too complex.

It may perform poorly when relationships are nonlinear or when external factors influence sales.

### Prophet

Prophet was selected because it is designed for business time-series forecasting.

It can model:

- Overall trend
- Yearly seasonality
- Weekly seasonality
- User-defined known events

The project considered weekly and yearly seasonality.

Prophet is generally easy to interpret and robust to missing or irregular data. It may perform poorly if the available history is short or if lag relationships are stronger than trend and seasonality.

### XGBoost

XGBoost was selected because it can model nonlinear relationships in engineered lag features.

It is a supervised machine learning algorithm based on gradient boosting.

The model builds many decision trees. Each new tree tries to correct the errors made by previous trees.

XGBoost was suitable because the project represented forecasting as a supervised regression problem:

- Input: previous sales and rolling sales statistics
- Output: current weekly total sales

XGBoost does not automatically understand time order. The lag and rolling features provide the time dependency.

## 5. Why all four model types were tested

A project should not select XGBoost only because it is popular.

The models represent three levels of complexity:

1. Holt-Winters for level, trend, and seasonality
2. ARIMA for autocorrelation and differencing
3. Prophet for trend and calendar seasonality
4. XGBoost for nonlinear relationships in engineered features

Comparing them answers:

- Is a simple model sufficient?
- Is seasonality the main pattern?
- Is autocorrelation important?
- Do nonlinear lag relationships improve prediction?

This makes the final choice evidence-based.

## 6. Why lag features were created

A lag feature uses an older value as a predictor for a later value.

The project used:

- Lag 1: sales from one week earlier
- Lag 2: sales from two weeks earlier
- Lag 3: sales from three weeks earlier
- Lag 4: sales from four weeks earlier

Example:

The target for week 5 can use sales from weeks 4, 3, 2, and 1.

Four lags were selected because they represent one recent month of weekly observations. They capture short-term dependence without creating too many redundant inputs.

Lag features should be based only on information already available at prediction time.

Using the target week's sales or future sales would create data leakage.

## 7. Why a four-week rolling average was used

The rolling mean summarizes the average sales from the previous four weeks.

For one weekly record, it represents short-term general sales level.

Why it is useful:

- It reduces the effect of one unusual week.
- It identifies a short-term trend.
- It combines the most recent month of information.
- It provides a stable feature for a model that may react too strongly to one lag.

Example:

If lag values fluctuate heavily, the four-week rolling mean provides the recent average as a smoother signal.

The rolling window should end before the target week. It must not include the value being predicted.

The project used a shift through lag features and a historical rolling calculation. These methods prevent the model from seeing the target week while computing features.

## 8. Why four lags and a four-week rolling mean may be related

The features overlap.

The lags describe exact recent values:

- Most recent value
- Second most recent value
- Third most recent value
- Fourth most recent value

The rolling mean describes their average.

Together, they give XGBoost both detailed and smoothed recent information.

This is useful because a spike in one week may not represent the whole trend.

## 9. Why a chronological split was used

The project used 85% of observations for training and 15% for testing.

The split was chronological:

- Earlier observations for training
- Later observations for testing

Random splitting is unsuitable for time-series data because it can:

- Place future observations in training
- Place past observations in testing
- Create unrealistically high performance
- Leak future patterns into the model

A chronological split makes the evaluation resemble a real forecast.

## 10. What “85% training and 15% testing” means

The dataset was ordered by date before splitting.

The first 85% represented the historical period used to learn patterns.

The final 15% represented unseen future observations used for evaluation.

This was a practical project split. For more reliable evaluation, time-series cross-validation would repeatedly move the test window forward.

## 11. Why data leakage was important

Data leakage occurs when training uses information that would not be available at prediction time.

In this project, leakage risks include:

- Including the target week's sales in features
- Using a rolling mean that includes the target week
- Randomly placing future observations in training
- Calculating dataset-wide statistics and using the same values for training and testing

The lag and rolling features must be created carefully from earlier observations.

An interviewer may ask whether the target is available. The answer should be:

"The target was not used as an input for the corresponding prediction."

## 12. Metrics used in the project

The project evaluated models using:

- Mean Absolute Error
- Root Mean Squared Error
- R²

These are regression metrics, not classification metrics.

## 13. Mean Absolute Error

MAE is the average absolute difference between actual and predicted values.

It answers:

"How far is the predicted sales value from the actual value, on average?"

For example, a MAE of 22.48 means that predictions differed from actual weekly sales by approximately 22.48 units on average.

MAE advantages:

- Easy to explain
- Uses the original sales unit
- Treats all errors equally
- Less affected by a small number of extreme values than RMSE

MAE limitation:

- It does not heavily emphasize unusually large forecasting mistakes.

## 14. Root Mean Squared Error

RMSE is the square root of the average of squared errors.

It answers:

"How large are the prediction errors, while giving additional weight to large errors?"

RMSE advantages:

- In the same units as the target
- Penalizes large errors more than MAE
- Useful when severe large mistakes are costly

RMSE limitation:

- It is sensitive to outliers.
- It can make one unusual sales period dominate the result.

For the recorded XGBoost result:

- MAE: 22.48
- RMSE: 37.46

RMSE is larger than MAE because squared errors increase the effect of larger mistakes.

## 15. R² score

R² measures how much variation in actual sales is explained by the model.

A value of 1 means perfect prediction.

A value of 0 means the model explains no more variation than predicting the mean target.

A negative value means the model performs worse than predicting the mean.

The recorded XGBoost R² was 0.907.

This can be explained as:

"The model explained approximately 90.7% of the variation in weekly sales in the recorded test result."

R² limitations:

- It does not show the average error in sales units.
- It can remain high when individual predictions are still operationally poor.
- It is not sufficient when there are outliers or a changing target range.
- It should be reviewed together with MAE and RMSE.

## 16. Why all three metrics were needed

One metric is not enough.

- MAE explains average absolute mistake.
- RMSE emphasizes large mistakes.
- R² explains the proportion of variation captured.

The project should not say:

"The model is excellent because R² is 0.907."

A stronger answer is:

"The model had an R² of 0.907 and low MAE and RMSE relative to the alternatives. MAE and RMSE show the actual forecasting error, while R² shows how much variation was explained."

## 17. Recorded model comparison

The notebook recorded the following results:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| XGBoost Lag-Enhanced | 22.48 | 37.46 | 0.907 |
| Prophet Tuned | 88.19 | 115.82 | 0.042 |
| ARIMA(1,1,1) | 97.65 | 123.57 | -0.090 |
| Holt-Winters | 107.19 | 131.62 | -0.236 |

The XGBoost model was selected because it had:

- The lowest MAE
- The lowest RMSE
- The highest R²

No model achieved the highest value for R² while having poor MAE and RMSE in this comparison. This made XGBoost a clear winner based on all three metrics.

## 18. How the final model should be selected

The selection process should follow an ordered process:

1. Establish simple baselines.
2. Define the correct chronological test period.
3. Create features that use only historical information.
4. Train all candidate models.
5. Calculate MAE, RMSE, and R².
6. Compare the same test period for every model.
7. Prefer the model with the strongest error metrics.
8. Check stability, interpretability, and operational cost.
9. Record the selected model and parameters.

A good model should not be chosen only by R². MAE and RMSE should be considered together.

## 19. Why XGBoost performed better

Based on the project results, the XGBoost lag model found stronger nonlinear relationships than the classical models.

This suggests that:

- Recent sales were strong predictors.
- Short-term rolling patterns were useful.
- Lag relationships were more useful than simple trend and seasonality assumptions.
- XGBoost learned interactions among lag features.

This is a conclusion for the available dataset and test period. It does not prove that XGBoost will always be the best model for all future data.

## 20. XGBoost parameters in this project

The main training parameters were:

- 400 estimators
- Learning rate: 0.05
- Maximum depth: 6
- Subsample: 0.9
- Column subsample: 0.9
- Random state: 42

### Number of estimators

An estimator is a boosting tree.

More trees can learn more detail but may increase training time and overfitting.

### Learning rate

The learning rate controls how much each tree affects the final model.

A smaller learning rate makes training gradual. It often needs more trees.

### Maximum depth

Maximum depth limits tree complexity.

The project used 6, allowing the model to capture nonlinear relationships without creating extremely deep trees.

### Subsample

Subsample determines the fraction of training rows used for building each tree.

A value of 0.9 means each tree uses 90% of the training observations.

### Column subsample

Column subsample determines the fraction of features considered by a tree.

A value of 0.9 means 90% of features are available to the tree-building process.

### Random state

Random state makes results reproducible when a process uses randomness.

## 21. Why tune or tune less

The project used a limited set of XGBoost parameters.

Tuning should be performed using a validation or cross-validation process, not the final test set.

Using test data to choose parameters can overfit the test period.

A stronger workflow would use:

- Rolling-origin validation
- Multiple historical validation periods
- Parameter search
- Final evaluation only after choosing the model

## 22. Why XGBoost needs scaling?

XGBoost is based on decision trees.

It generally does not require feature standardization because tree splits depend on feature values and thresholds, not distances.

This differs from models such as k-nearest neighbors or logistic regression.

## 23. Why the rolling mean is not enough as an input

A rolling mean provides a smooth average, but it loses detail.

For example, it cannot tell whether:

- Sales rose or fell over the previous four weeks
- The latest week was unusual
- The pattern was stable or volatile

The four lag features retain the individual values and can provide this detail.

This is why the project combines lag features and a rolling average.

## 24. Difference between time-series models and lag-based models

Holt-Winters, ARIMA, and Prophet directly model time-series patterns.

XGBoost was trained on engineered rows where lag features converted historical observations into supervised inputs.

The key difference is:

- Classical model: understands time through the sequence or calendar
- Lag-based model: converts time history into separate columns

XGBoost is therefore dependent on the quality of feature engineering.

## 25. Stationarity and why it matters

Stationarity means statistical properties such as mean and variance remain reasonably stable over time.

ARIMA uses differencing to handle trend and make a series more stable.

Signs of non-stationarity include:

- A strong upward or downward trend
- Increasing variance
- Changing seasonal pattern

In this project, ARIMA used one difference step. A complete implementation should check stationarity before the final model choice.

## 26. Seasonality in this project

The project considered:

- Weekly seasonality
- Yearly seasonality

Weekly seasonality means a pattern repeats every week.

Yearly seasonality means a pattern repeats during the same period across years.

However, the dataset has only 302 weekly records, which covers approximately six years of data. It may not provide enough observations for every complex seasonal pattern.

Holt-Winters and Prophet explicitly consider seasonality. ARIMA and XGBoost require features or transformations to represent it effectively.

## 27. Difference between MAE and business cost

MAE is a technical metric.

A low MAE does not automatically represent the lowest business cost.

For example:

- Underforecasting may cause lost sales or shortages.
- Overforecasting may create excess inventory.
- Large forecast errors may be more costly than many small errors.

A pharmacy or distributor should map MAE and RMSE to inventory cost before choosing a model for real use.

## 28. Why the test result may differ in another run

The project notebook reported R² 0.907, MAE 22.48, and RMSE 37.46.

A local script execution produced different metrics:

- MAE: 45.65
- RMSE: 63.02
- R²: 0.721

This may be due to:

- Different code between notebook and script
- Different data filtering
- Different train-test boundaries
- Different library versions
- Different model settings
- Reproducibility issues
- Experiment names or data-processing inconsistencies

An interview answer should not hide this difference.

The project should use one reproducible pipeline and report only results from the same final model and test period.

## 29. Why actual versus predicted metrics are not enough

Metrics summarize performance across many weeks.

They do not show:

- When the model fails
- Whether errors increase near the end
- Whether certain seasons are difficult
- Whether underforecasting is more frequent
- Whether the model follows the correct trend

A technical review should also include:

- Actual versus predicted plots
- Residual plots
- Error distribution
- Worst forecasting periods
- Weekly or monthly performance

## 30. What residuals are

A residual is the difference between actual and predicted sales:

Actual sales minus predicted sales.

Residual analysis helps answer:

- Are errors centered around zero?
- Are errors increasing over time?
- Is there an obvious pattern?
- Are there extreme mistakes?

If residuals show a pattern, the model may be missing an important trend or seasonal component.

## 31. Important time-series cross-validation

For a stronger evaluation, use rolling-origin or expanding-window cross-validation.

Rolling-origin validation:

- Trains on earlier data
- Tests on a later small window
- Moves forward in time
- Repeats across multiple periods

Expanding-window validation uses a training period that grows with every iteration.

This produces several MAE, RMSE, and R² values instead of only one test score.

## 32. How the project could be improved technically

### Data improvements

- Add holidays
- Add promotions
- Add weather
- Add stock availability
- Add drug-category forecasts
- Handle missing and outlier values consistently

### Feature improvements

- Add month and season indicators
- Add year
- Add trend features
- Add longer rolling averages
- Add lag values for seasonal periods
- Test different lag lengths

### Modeling improvements

- Tune XGBoost using validation data
- Test direct multi-step forecasting
- Use rolling-window cross-validation
- Compare ensemble models
- Add a statistical benchmark with retraining

### Evaluation improvements

- Add residual plots
- Report mean and standard deviation across folds
- Add business-cost metrics
- Separate performance by season
- Evaluate underforecasting and overforecasting separately

### MLOps improvements

- Save each exact dataset version
- Pin dependency versions
- Remove hardcoded credentials
- Register a stable production model
- Add API input validation
- Implement real drift detection
- Retrain when performance falls
- Add automated tests

## 33. Batch versus real-time forecasting

This project demonstrates real-time API forecasting.

A weekly pharmaceutical forecast can also be generated as a batch:

- Run once per week
- Load the latest sales
- Create lag features
- Predict upcoming weeks
- Save or publish forecast values

Real-time API forecasting is useful for demonstrations and manual requests.

Batch forecasting may be more practical for a large inventory system.

## 34. Direct versus rolling multi-step forecasts

Direct forecasting predicts multiple future periods using separate models.

Rolling forecasting:

- Predicts the next week
- Uses the predicted value as an input
- Recreates lag features
- Predicts the next period
- Repeats

The FastAPI project uses one-week lag inputs rather than providing a complete, reliable multi-week forecasting workflow.

This difference should be understood in an interview.

## 35. What to say if asked why XGBoost was used

A strong answer is:

"I used XGBoost because the project converts time-series history into lag and rolling features. XGBoost can learn nonlinear relationships and interactions among those features. I did not choose it only by popularity; I compared it with Holt-Winters, ARIMA, and Prophet on the same chronological test period. In the recorded results, XGBoost had the lowest MAE and RMSE and the highest R²."

## 36. What to say if asked why rolling averages were used

A strong answer is:

"I used a four-week rolling average to provide a smoothed measure of recent sales. The lag values preserve exact recent observations, while the rolling average captures the recent overall level and reduces the effect of a one-week spike or drop."

## 37. What to say if asked why these metrics were chosen

A strong answer is:

"MAE shows the average absolute forecasting error in sales units. RMSE gives more importance to large mistakes. R² shows how much variation in sales is explained. I considered all three rather than relying on only R²."

## 38. What to say if asked why the 85/15 split was used

A strong answer is:

"The data was ordered by date and the first 85% was used for training while the latest 15% was used for testing. This avoids random mixing of future and past observations and gives a realistic out-of-time evaluation."

## 39. What to say if asked why classical models did not win

A strong answer is:

"The classical models performed poorly on the same test period. Their assumptions about trend, seasonality, or linear autocorrelation were insufficient for this dataset. The recent lag and rolling patterns appear to provide stronger predictive information."

Avoid claiming the classical models are generally worse. They only performed worse in this specific dataset and test period.

## 40. Final 60-second technical answer

"The project forecasts total weekly pharmaceutical sales from a 302-week dataset containing eight drug categories. I aggregated the categories and engineered lag features for the previous four weeks plus a four-week rolling mean.

I used these variables because recent sales are likely to predict near-future demand, and the rolling mean adds a smoothed short-term sales level.

I split the data chronologically with 85% for training and 15% for testing. I tested Holt-Winters, ARIMA, Prophet, and XGBoost.

Holt-Winters models level, trend, and seasonality. ARIMA uses past values and forecast errors through autoregressive and moving-average terms. Prophet models trend and calendar seasonality. XGBoost learns nonlinear relationships in the engineered lag features.

I compared MAE, RMSE, and R². The recorded XGBoost model achieved the best result, with MAE 22.48, RMSE 37.46, and R² 0.907.

The result is specific to this dataset and test period. A stronger real implementation would use time-series cross-validation, business-cost metrics, residual analysis, and real drift monitoring."
