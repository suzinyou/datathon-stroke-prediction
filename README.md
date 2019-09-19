# datathon-stroke-prediction

Korea Clinical Datathon 2019
Team 3's repository

## Note to developers

개발하며 노트북에서 `src` 에 있는 모듈을 사용하려면, 프로젝트 root 디렉토리에서 다음과 같이 입력합니다.
```
pip install --editable .
```


## Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed. For example, `*.csv` files exported from `*.vital` files.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump. `*.vital` files only.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   ├── vitalutils <- Cloned from https://github.com/vitaldb/vitalutils
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


## Development of real-time prediction model for perioperative stroke
(subject to change)

### Background
Perioperative stroke is associated with significant morbidity and mortality. Despite
advances in surgical techniques and improvements in perioperative care, the
incidence of perioperative strokes has not decreased. In addition to the potentially
under-appreciated incidence and significance of perioperative stroke, recent data
have demonstrated that mortality from perioperative stroke may be particularly high,
with an approximate incidence ranging from 20-60% depending on the stroke,
operation, and patient. As such, interest has grown in the identification of those at
risk for perioperative stoke.

### Objective
The purpose of this study is to develop the real-time prediction model for
perioperative stroke.

### Methods
Each patient in the Vital DB will be classified according to whether or not the subject
has a perioperative stroke. Age, gender, BMI, and other risk factors will be
investigated to create a model for predicting perioperative stroke. Risk factors such
as anesthetic and monitoring techniques, pharmacologic strategies, and physiologic
strategies will be used for the deep learning model. We will measure the accuracy
and timeliness of the deep learning model’s forecasts. Univariate and multivariate
analysis will be used to calculate the impact of each variable to perioperative stroke.

### Expected result
The observed improvements in prediction for perioperative stroke are noteworthy in
that these variables routinely collected during operation without the need for any
additional effort.



--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
