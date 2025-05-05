# Datathon Kit ⚡

A set of opinionated tools, code, and resources for quick and dirty Machine Learning projects, datathons, or ad-hoc competitions.

## 🚀 Workflow

When starting to address a new problem, follow these steps for a solid foundation.

1. Learn more about the problem.
   - Search for similar [Kaggle competitions](https://www.kaggle.com/competitions).
   - Check the task (classification, regression, ...) in [Papers with Code](https://paperswithcode.com/).
   - Check [Machine Learning subreddit](https://www.reddit.com/r/MachineLearning) for similar problems.
2. Do a basic data exploration.
   - Understand the problem and gather a sense of what can be important.
   - Check outliers and missing values.
3. Get baseline model (e.g: return average of target) working end to end.
   - Keep track of the experiments and results.
4. Design an evaluation method as close as the final evaluation.
   - Plot local evaluation metrics against the public ones (correlation) to validate how well your validation strategy works.
5. Try different approaches for preprocessing (encodings, Deep Feature Synthesis, lags, aggregations, imputers, target/count encoding, sin/cos encoding, ...).
   - If you're working as a group, split preprocessing feature generation between files.
6. Apply any postprocessing that might fix small things.
   - Plot real and predicted target distribution to see how well your model understand the underlying distribution.
   - Heuristics (clipping, mapping to another distribution, ...) can be applied to improve or correct predictions.
7. Tune hyper-parameters once you've settled on an specific approach ([optuna](https://optuna.readthedocs.io/)).
   - Plot learning curves (with [sklearn](https://scikit-learn.org/stable/modules/learning_curve.html) or [external tools](https://github.com/reiinakano/scikit-plot)) to avoid overfitting.
8. Plot and visualize the predictions (target vs predicted errors, histograms, random prediction, ...) to make sure they're doing as expected. Explain the predictions with [SHAP](https://github.com/slundberg/shap).
9. If time allows, [stack](https://scikit-learn.org/stable/auto_examples/ensemble/plot_stack_predictors.html) models ([example](https://www.kaggle.com/couyang/featuretools-sklearn-pipeline#ML-Pipeline)) to improve performance.
10. Iterate. Try to keep performance in mind so you can make the best use of your time.

## 🛠️ Toolkit

This repository is built on top of the following tools.

- [Polars](https://pola.rs/)
- [Altair](https://altair-viz.github.io/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [AutoGluon](https://auto.gluon.ai/stable/install.html)
- [XGBoost](https://xgboost.readthedocs.io/en/stable/)
- [LightGBM](https://lightgbm.readthedocs.io/en/latest/)
- [CatBoost](https://catboost.ai/en/docs/v3/about/)
- [Optuna](https://optuna.org/)

## 📚 Resources

### ⚙️ Preprocessing Resources

- [Feature Engineering Library](https://feature-engine.trainindata.com/).
- [Feature Engineering Ideas](https://github.com/aikho/awesome-feature-engineering).
- [Deep Feature Synthesis](https://featuretools.alteryx.com/en/stable/getting_started/afe.html). [Simple tutorial](https://www.kaggle.com/willkoehrsen/automated-feature-engineering-basics).
- [Modern Feature Engineering Ideas](https://www.kaggle.com/c/playground-series-s4e12/discussion/554328) ([code](https://www.kaggle.com/code/cdeotte/first-place-single-model-cv-1-016-lb-1-016)).
  - [Target Encoding](https://www.kaggle.com/competitions/playground-series-s4e12/discussion/554328) (with cross-validation to avoid leakage). [Data leakage is a common problem in Target Encoding](https://www.geeksforgeeks.org/target-encoding-using-nested-cv-in-sklearn-pipeline/#the-challenge-of-data-leakage-nested-crossvalidation-cv)!
  - Forward Feature Selection.
- [Hillclimbing](https://www.kaggle.com/competitions/playground-series-s3e14/discussion/410639).

### 🛞 Scikit Learn Compatible Transformers

- [LEGO](https://github.com/koaning/scikit-lego)
- [Skrub](https://github.com/skrub-data/skrub)
- [Skoot](https://github.com/tgsmith61591/skoot)
- [Sktools](https://github.com/david26694/sktools)
- [Scikit-Learn Related Projects](https://scikit-learn.org/stable/related_projects.html).
- [Contributions repository](https://github.com/scikit-learn-contrib)
- [Awesome Scikit-Learn](https://github.com/fkromer/awesome-scikit-learn)
- [Category Encoders](https://contrib.scikit-learn.org/category_encoders)

### 🐻‍❄️ Polars

- [Modern Polars](https://kevinheavey.github.io/modern-polars/)
- [Polars The Definitive Guide](https://github.com/jeroenjanssens/python-polars-the-definitive-guide)

### 📈 Time Series Resources

- [Quick Tutorials](https://www.kaggle.com/c/jane-street-market-prediction/discussion/198951)
- [Tsfresh](https://tsfresh.readthedocs.io/en/latest/)
- [Fold](https://github.com/dream-faster/fold)
- [Neural Prophet](https://neuralprophet.com/) or [TimesFM](https://github.com/google-research/timesfm)
- [Darts](https://github.com/unit8co/darts)
- [Functime](https://docs.functime.ai/)
- [Pytimetk](https://github.com/business-science/pytimetk)
- [Sktime](https://github.com/alan-turing-institute/sktime) / [Aeon](https://github.com/aeon-toolkit/aeon)
- [skforecast](https://skforecast.org/)
- [Awesome Collection](https://github.com/MaxBenChrist/awesome_time_series_in_python)
- [Video with great ideas](https://www.youtube.com/watch?v=9QtL7m3YS9I)
- [Tutorial Kaggle Notebook](https://www.kaggle.com/code/tumpanjawat/s3e19-course-eda-fe-lightgbm)
- Think about adding external datasets like [related Google Trends search](https://trends.google.com/trends/), PiPy Packages downloads, [Statista](https://www.statista.com/), weather, ...
- [TabPFN for time series](https://github.com/liam-sbhoo/tabpfn-time-series)
- [Gradient Boosting for Survival Analysis](https://soda-inria.github.io/hazardous/)

### 🤖 AutoML

- Tabular: [FLAML](https://github.com/microsoft/FLAML), [AutoGluon](https://auto.gluon.ai/), [Perpetual](https://github.com/perpetual-labs/perpetual), [LightAutoML](https://github.com/sb-ai-lab/LightAutoML), [AutoSklearn](https://github.com/automl/auto-sklearn), Google AI Platform, [PyCaret](https://github.com/pycaret/pycaret), [Fast.ai](https://docs.fast.ai/), [TableVectorizer](https://skrub-data.org/stable/reference/generated/skrub.TableVectorizer.html#tablevectorizer), [TabICL](https://github.com/soda-inria/tabicl), [TabPFN](https://github.com/PriorLabs/tabpfn) ([community](https://github.com/PriorLabs/tabpfn-community)).
- Time Series: [AtsPy](https://github.com/firmai/atspy), [DeepAR](https://docs.aws.amazon.com/forecast/latest/dg/aws-forecast-recipe-deeparplus.html), [Nixtla's NBEATS](https://nixtlaverse.nixtla.io/neuralforecast/models.nbeats.html), [AutoTS](https://github.com/winedarksea/AutoTS).

### 📊 Datathon Platforms

- [Kaggle](https://www.kaggle.com/competitions)
- [MLContest](https://mlcontests.com/). They also share a "State of Competitive Machine Learning" report every year ([2023](https://mlcontests.com/state-of-competitive-machine-learning-2023), [2024](https://mlcontests.com/state-of-machine-learning-competitions-2024/)) and summaries on the state of the art for ["Tabular Data"](https://mlcontests.com/tabular-data/).
- [Humyn](https://app.humyn.ai/)
- [DrivenData](https://www.drivendata.org/competitions/)
- [Xeek](https://xeek.ai/challenges)
- [Cryptopond](https://cryptopond.xyz/)
