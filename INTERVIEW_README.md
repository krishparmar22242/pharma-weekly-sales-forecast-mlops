# Pharmaceutical Weekly Sales Forecasting — MLOps Interview Guide

This project should help you explain how you built an end-to-end machine learning system that predicts future weekly pharmaceutical sales. The answers below are written in simple language and contain no code.

## 1. Short project introduction

This project predicts weekly pharmaceutical sales using historical sales data.

The dataset contains eight drug-category columns. I aggregated those categories into one total weekly sales value. I then created lag features and a rolling-average feature. Lag features represent sales in previous weeks, so the model can learn from recent trends.

I trained and compared several forecasting models, including Holt-Winters, ARIMA, Prophet, and XGBoost. The lag-enhanced XGBoost model performed best. The project also uses MLflow and DagsHub for experiment tracking, FastAPI for predictions, Streamlit for visualization, Docker for packaging, and GitHub Actions for CI/CD.

In the project notebook, XGBoost achieved:

- R²: 0.907
- MAE: 22.48
- RMSE: 37.46

The system is complete from data processing to model tracking and deployment.

## 2. Project objective

The objective was to forecast the next week's total pharmaceutical sales.

The project helps with:

- Planning inventory
- Forecasting demand
- Reducing overstock and shortages
- Comparing forecasting methods
- Demonstrating an automated MLOps workflow

## 3. Two-minute explanation

Start with this simple explanation:

"I built an end-to-end MLOps project to forecast weekly pharmaceutical sales. The source data contains eight drug-category sales for each week. I combined those categories into a total-sales value.

I prepared lag features for the previous four weeks and a four-week rolling mean. These features allowed XGBoost to learn recent sales patterns. I split the data chronologically, with 85% for training and 15% for testing.

I compared XGBoost with Holt-Winters, ARIMA, and Prophet. In the recorded results, XGBoost performed best with an R² of 0.907, MAE of 22.48, and RMSE of 37.46.

I logged parameters, metrics, and artifacts using MLflow and DagsHub. FastAPI exposes a prediction endpoint, and Streamlit provides a dashboard for metrics and manual predictions. GitHub Actions trains the model, creates an artifact, simulates deployment, and records monitoring output. Docker packages the application.

The goal was not only to train a model, but also to show how a machine learning project can be tracked, served, deployed, and monitored."

## 4. Common interview questions and answers

### What problem does this project solve?

It predicts total pharmaceutical sales for the next week.

Accurate forecasts can help a pharmacy or distribution organization plan inventory and anticipate demand changes.

### Why did you use weekly sales instead of daily sales?

Weekly aggregation reduces noise and matches the available dataset. It is useful for short-term demand planning and inventory decisions.

### What were the input features?

The model used five engineered features:

- Sales one week earlier
- Sales two weeks earlier
- Sales three weeks earlier
- Sales four weeks earlier
- Average sales across the previous four weeks

The target was the current week's total sales.

### How were lag features created?

A lag feature shifts sales backward by a fixed number of weeks.

For example:

- Lag 1 is the previous week's sales.
- Lag 2 is sales from two weeks earlier.
- Lag 4 is sales from four weeks earlier.

The rolling mean summarizes the average of the previous four weeks.

### Why did you use feature engineering instead of only a time-series model?

Traditional time-series models can model seasonality and trends. However, recent sales values and short-term averages were strong predictors in this dataset.

Converting the forecasting problem into lag features allowed XGBoost to learn relationships among recent observations.

### Why did you train more than one model?

Multiple models provide a fair comparison.

The project compared:

- Holt-Winters
- ARIMA
- Prophet
- Lag-enhanced XGBoost

This showed whether a more complex machine learning model was better than simpler statistical approaches.

### What is Holt-Winters?

Holt-Winters is a statistical forecasting method that models level, trend, and seasonality.

It is useful when a time series has predictable patterns. It is generally easier to explain than a machine learning model.

### What is ARIMA?

ARIMA means AutoRegressive Integrated Moving Average.

It forecasts by modeling relationships between:

- Past values
- Past forecast errors
- Differencing used to make the series stable

The project used ARIMA(1,1,1).

### What is Prophet?

Prophet is a forecasting library designed to handle trends and seasonality.

It can model:

- Weekly seasonality
- Yearly seasonality
- Long-term trends

In this project, Prophet did not perform as well as XGBoost.

### What is XGBoost?

XGBoost is an implementation of gradient boosting based on decision trees.

It trains many trees sequentially. Each new tree attempts to correct errors made by previous trees.

It was the strongest model in this project because recent sales history and rolling averages were useful signals.

### Why did you use XGBoost for time-series forecasting?

XGBoost is a supervised machine learning model. The project converted time-series data into a table with lag features.

Advantages include:

- Ability to learn non-linear relationships
- Strong performance
- Support for regularization
- Ability to handle interactions between features

Its main limitation is that it does not automatically understand time-series seasonality unless those patterns are represented as features.

### How did you split the data?

The data was split chronologically.

- 85% was used for training.
- 15% was used for testing.

Random splitting was avoided because future data must not be used to predict the past. A chronological split gives a more realistic evaluation.

### Why is chronological splitting important?

Sales data is time-dependent. A random split could place future observations in training and previous observations in testing.

That would create data leakage and give an unrealistically high score.

### What metrics did you use?

The project used MAE, RMSE, and R².

### What is MAE?

MAE is Mean Absolute Error.

It measures the average absolute difference between actual and predicted sales. A MAE of 22.48 means the predictions were off by approximately 22.48 sales units on average.

### What is RMSE?

RMSE is Root Mean Squared Error.

It measures prediction error and gives more weight to large mistakes.

RMSE is useful because an unusually wrong forecast should be considered more serious.

### What is R²?

R² shows how much variation in the actual sales is explained by the model.

An R² of 0.907 means the model explained approximately 90.7% of the variation in the test data.

R² does not prove that a model is perfect. It should be interpreted with MAE and RMSE.

### How did XGBoost compare with the baseline models?

The recorded comparison was:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| XGBoost with lag features | 22.48 | 37.46 | 0.907 |
| Prophet Tuned | 88.19 | 115.82 | 0.042 |
| ARIMA(1,1,1) | 97.65 | 123.57 | -0.090 |
| Holt-Winters | 107.19 | 131.62 | -0.236 |

XGBoost had the lowest MAE and RMSE and the highest R².

Negative R² for a model means it performed worse than simply predicting the average target value.

### Which hyperparameters were used for XGBoost?

The main settings were:

- 400 trees
- Learning rate of 0.05
- Maximum tree depth of 6
- Subsample ratio of 0.9
- Column-sample ratio of 0.9
- Random state of 42

### What is the learning rate?

The learning rate controls how strongly each new tree influences the final prediction.

A lower learning rate usually needs more trees but can provide more stable learning.

### What is max depth?

Max depth controls how complex an individual decision tree can be.

Deeper trees can learn complex patterns but may overfit. This project used a depth of 6 as a compromise.

### What is subsampling?

Subsampling means training each tree using only a fraction of the training rows.

The project used 90% of rows per tree. This increases randomness and can reduce overfitting.

### What is column subsampling?

Column subsampling means that each tree considers only a fraction of input features.

It can reduce dependency on one feature and make the model more robust.

### What is MLflow?

MLflow is an experiment-tracking and model-management platform.

It was used to record:

- Experiment names
- Run names
- Model parameters
- Performance metrics
- Model artifacts

This makes experiments easier to compare and reproduce.

### What is DagsHub in this project?

DagsHub is used as the remote MLflow tracking and model repository.

The project can log runs and retrieve model information through the DagsHub MLflow endpoint.

### What is the model registry?

A model registry is a centralized location for managing model versions.

The intended workflow in this project is to select the best model and register or export it for deployment. It helps distinguish between development models and production models.

### What does the FastAPI service do?

FastAPI provides a REST API for predictions.

It accepts the five engineered feature values and returns a forecast value.

The API also has a health endpoint that reports whether the service is running and whether it loaded a DagsHub or local model.

### What is the role of Streamlit?

Streamlit provides an interactive dashboard.

Its roles are to:

- Display recent MLflow metrics
- Accept manual lag-feature inputs
- Send prediction requests to FastAPI
- Visualize uploaded actual-versus-predicted data

### What is Docker used for?

Docker packages the application, dependencies, and runtime configuration into a container.

This makes the application easier to run consistently on a local machine or server.

### What is CI/CD?

CI/CD means Continuous Integration and Continuous Deployment or Delivery.

In this project:

- Pushes and pull requests trigger the workflow.
- Dependencies are installed.
- The model is trained.
- Metrics are tracked.
- A model artifact is created.
- FastAPI deployment is simulated.
- Monitoring output is generated.

CI/CD increases automation and reduces manual deployment work.

### What happens in the GitHub Actions workflow?

The workflow has three stages:

1. Build, train, and track
2. Deploy a model artifact
3. Monitor the API

The build stage trains the model and uploads the model artifact.

The deployment stage downloads the artifact and starts a temporary FastAPI process for testing.

The monitoring stage checks the health endpoint and uploads monitoring logs.

### Is this real production deployment?

No, the deployment is a demonstration rather than a live external deployment.

The workflow starts a temporary server inside a GitHub Actions runner. It does not automatically deploy to AWS, Azure, Google Cloud, Kubernetes, or another managed platform.

### What is model monitoring?

Model monitoring checks whether a deployed model remains accurate and reliable.

It may monitor:

- API availability
- Prediction latency
- Prediction volume
- Accuracy over time
- Feature distribution
- Data drift
- Concept drift

### What monitoring is implemented in this project?

The project checks API health and logs a monitoring record.

The current monitoring data for accuracy, latency, and drift is simulated. It does not calculate them from real production predictions.

### What is data drift?

Data drift occurs when the distribution of incoming feature values changes from the data on which the model was trained.

For example, if recent sales become much higher or lower than historical values, the input distribution may have changed.

### What is concept drift?

Concept drift occurs when the relationship between input features and the target changes.

For example, a past lag pattern may stop predicting future sales correctly after a policy, promotion, or market change.

### What is the difference between data drift and concept drift?

Data drift concerns changes in input distributions. Concept drift concerns changes in the relationship between inputs and target values.

Both can reduce model performance over time.

### What is data leakage in this project?

Data leakage means future information accidentally influences training.

This project uses lagged sales and a chronological split, which helps prevent leakage. Randomly splitting time-series observations or using future rolling statistics would create leakage.

### What is overfitting?

Overfitting occurs when a model learns training patterns too closely and performs poorly on unseen data.

For XGBoost, a high maximum depth or too many trees can increase overfitting.

The holdout test provides an estimate of generalization.

### How could you prevent overfitting in this project?

Possible methods include:

- Using a validation period
- Limiting tree depth
- Reducing the number of trees
- Lowering the learning rate
- Using subsampling
- Applying regularization
- Testing with rolling-window backtesting
- Keeping more future observations for testing

### Why should time-series cross-validation be used?

A single chronological split is useful but may not cover enough time periods.

Rolling or expanding-window cross-validation repeatedly trains on past data and tests on later periods. This gives a more reliable estimate of real forecasting performance.

### Why is 85/15 a reasonable split?

It gives the model enough historical data for training while reserving the newest 15% for testing.

It would be better to split based on a meaningful period such as the final year rather than only a percentage.

### What was the result of the local execution?

When the training script was executed in the local environment with a local SQLite MLflow backend, the script completed successfully.

The local run produced:

- MAE: 45.65
- RMSE: 63.02
- R²: 0.721

These metrics differed from the notebook's recorded XGBoost metrics. That difference should be explained honestly in an interview.

Possible reasons include environment differences, dependency versions, and notebook-specific processing or model setup.

### Why might local metrics differ from notebook metrics?

Possible reasons include:

- A mismatch between notebook and script settings
- Different dependency or library versions
- Different data filtering
- A different random state
- A mismatch in model training or evaluation code
- A different train-test split

This is one reason experiment tracking and reproducibility are important.

### What would you improve in the project?

I would improve:

- Removal of hardcoded credentials
- Environment-variable-based configuration
- Version pinning
- Consistent metric logging and naming
- Proper model registration
- A reliable model export process
- Real drift monitoring
- Real production deployment
- Time-series cross-validation
- A stronger API input validation system
- Automated tests
- One consistent project structure instead of duplicate Flask and FastAPI implementations

### How would you improve security?

I would:

- Revoke existing exposed tokens
- Use environment variables or a secret manager
- Store credentials in GitHub Actions secrets
- Never commit passwords or API tokens
- Use read-only credentials where appropriate
- Add authentication to prediction APIs in production

### How would you deploy this project in a real company?

A practical process would be:

1. Store training code and requirements in version control.
2. Trigger training through CI/CD.
3. Validate data and model performance.
4. Register the model in a model registry.
5. Package the model in a Docker container.
6. Deploy it behind an authenticated API.
7. Use a staging environment before production.
8. Monitor health, latency, drift, and prediction quality.
9. Retrain automatically or manually when performance declines.

### How would you automate retraining?

Retraining could run:

- On a fixed schedule
- When new data arrives
- When data drift is detected
- When model accuracy falls below a threshold
- When a manual approval is given

The pipeline should retrain, test, compare, register, and deploy only when the new model passes required thresholds.

### What is a baseline model?

A baseline is a simple model used as a comparison point.

In this project, Holt-Winters, ARIMA, and Prophet acted as forecasting baselines. XGBoost was compared against them.

### What is a holdout set?

A holdout set is data reserved for final evaluation.

The model does not use these observations during training. The 15% test portion is the holdout set.

### What is reproducibility?

Reproducibility means another person or system can run the same process and obtain approximately the same result.

It requires:

- A fixed dataset
- Dependency versions
- A random state
- Documented parameters
- Saved training code
- Experiment tracking
- A consistent environment

### What is an ML artifact?

An ML artifact is an output or file associated with a model run.

Examples include:

- Trained model file
- Metrics
- Parameters
- Plots
- Training data snapshot
- Logs

The project saves a trained XGBoost model as a pickle file.

### What is model serialization?

Serialization means converting a trained model into a file that can be stored or loaded later.

In this project, joblib is used to save and load the XGBoost model.

### What is the difference between batch forecasting and real-time prediction?

Batch forecasting calculates predictions for many periods or stores at once, usually on a schedule.

Real-time prediction provides a forecast immediately through an API after receiving input.

This project provides a real-time prediction endpoint, although weekly forecasting could also be performed in batches.

### What are the main challenges in this project?

The main challenges are:

- Maintaining chronological order
- Preventing future-data leakage
- Creating useful lag features
- Selecting and tuning models
- Comparing multiple metrics
- Making experiment tracking consistent
- Serving the correct model version
- Handling missing or invalid API inputs
- Monitoring real-world performance

### What are the limitations of this project?

- It does not use external factors such as promotions or holidays.
- It forecasts only total weekly sales, not sales for individual drug categories.
- The holdout split is limited and does not represent all possible future periods.
- The dashboard fetches remote metrics.
- Monitoring is partly simulated.
- Deployment is not an external production deployment.
- Credential management needs improvement.
- The API and export scripts have mismatched experiment and model assumptions.
- The project contains duplicate implementations with inconsistent APIs.

### What business decisions could this forecast support?

It can support:

- Weekly inventory planning
- Purchase scheduling
- Stock allocation
- Reduction of stockouts
- Reduction of excess inventory
- Warehouse planning
- Demand trend monitoring

### What risks could lead to incorrect forecasts?

Possible risks include:

- Data-quality problems
- Missing historical weeks
- Sudden promotions
- Supply-chain disruptions
- Changes in prescription behavior
- New drug products
- Outdated model parameters
- Incorrect lag-feature inputs
- Data drift

### What does an API request contain?

The intended FastAPI prediction request contains:

- lag_1
- lag_2
- lag_3
- lag_4
- rolling_mean_4

The API returns a forecast number.

### What is feature validation?

Feature validation checks that prediction inputs are:

- Present
- Numeric
- Within reasonable ranges
- In the correct format
- Consistent with training features

The current project has limited validation and should add stronger checks for production use.

### What happens if the registry model is unavailable?

The FastAPI service attempts to load the registered production model. If that fails, it falls back to a local backup model file.

This improves availability but may serve an outdated model silently unless the fallback is clearly monitored.

## 5. Simple architecture interview answer

"The data starts in a CSV file. I aggregate the eight drug categories and sort the records by date.

I create lag features and a rolling average, then split the data chronologically. I train XGBoost and compare it with statistical models.

I log the experiment with MLflow and DagsHub. The best model can be exported or registered.

FastAPI loads the model and provides a prediction endpoint. Streamlit provides a dashboard. Docker makes the service portable. GitHub Actions automates training, artifact creation, deployment simulation, and monitoring."

## 6. Terminology for quick revision

### XGBoost

A gradient-boosted decision-tree model that combines weak learners to create strong predictions.

### Lag feature

A previous observation transformed into an input variable.

### Rolling mean

The average of observations over a moving time window.

### MAE

Average absolute prediction error.

### RMSE

Error metric that emphasizes larger errors.

### R²

The percentage of target variation explained by the model.

### MLflow

Platform for experiment tracking and model management.

### DagsHub

Remote platform used for MLflow tracking and data science collaboration.

### FastAPI

Python framework for building REST APIs.

### Streamlit

Python framework for interactive dashboards.

### Docker

Tool for packaging applications and dependencies into containers.

### GitHub Actions

CI/CD automation platform integrated with GitHub.

### CI/CD

Automated testing, building, tracking, and deployment workflow.

### Data drift

Change in the distribution of input data.

### Concept drift

Change in the relationship between inputs and prediction target.

### Holdout set

Data kept aside to test the final model.

### Data leakage

Accidental use of future or target information during training.

### Model registry

Centralized storage for managing model versions and lifecycle stages.

### Artifact

A generated file or result from a model run.

### Serialization

Saving a trained model so it can be loaded later.

### Reproducibility

The ability to repeat an experiment with consistent data, code, parameters, and environments.

## 7. Important interview points to emphasize

- You used a chronological split rather than random splitting.
- You prevented future information leakage by using lag features and not training on test data.
- You compared XGBoost with three alternative forecasting methods.
- XGBoost was the best model in the recorded notebook results.
- MLflow and DagsHub provide experiment traceability.
- FastAPI and Streamlit separate prediction service and visualization.
- GitHub Actions automates the pipeline.
- Docker provides repeatable packaging.
- The project is a working MLOps demonstration, but deployment and monitoring can be strengthened.

## 8. Strong 30-second project summary

"This project predicts weekly pharmaceutical sales using lag features and XGBoost. I compared it with Holt-Winters, ARIMA, and Prophet using MAE, RMSE, and R². XGBoost performed best with R² 0.907. I used MLflow and DagsHub for experiment tracking, FastAPI for prediction, Streamlit for visualization, Docker for packaging, and GitHub Actions for CI/CD. I also identified that monitoring and production deployment are simulated and would need to be improved before real-world use."

## 9. Weak points you should be able to explain honestly

Do not claim the project is fully production-ready.

Be honest about these areas:

- DagsHub credentials must not remain hardcoded.
- The deployment workflow is simulated.
- Monitoring metrics are random in the current workflow.
- Notebook metrics and local script metrics differ.
- Cross-validation for time series is still needed.
- Real production drift detection is not implemented.
- API input validation and authentication need improvement.
- The repository contains duplicate Flask and FastAPI implementations.
- Some experiment and model names are inconsistent.

An interviewer is more likely to appreciate a clear explanation of limitations than an unsupported claim that everything is complete.

## 10. Final sample interview answer

"My project is an end-to-end MLOps solution for forecasting weekly pharmaceutical sales. I started with a dataset containing eight drug-category columns and aggregated them into total weekly sales. I engineered lag features for the previous four weeks and a four-week rolling average.

I used a chronological train-test split to avoid future leakage and compared Holt-Winters, ARIMA, Prophet, and XGBoost. XGBoost achieved the best recorded R² of 0.907, with an MAE of 22.48 and RMSE of 37.46.

I tracked the model parameters and metrics using MLflow and DagsHub. I exposed the model with FastAPI and built a Streamlit dashboard. Docker is used for packaging, and GitHub Actions automates training, artifact creation, deployment simulation, and monitoring.

The main areas I would improve are credential security, stronger test coverage, time-series cross-validation, real drift detection, and deployment to an external production environment."
