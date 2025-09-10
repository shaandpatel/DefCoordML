# DefCoordML
This project simulates the role of an **AI-powered NFL Defensive Coordinator**, designed to anticipate whether the offense will **pass** or **rush** before the snap. By analyzing historical NFL play-by-play data and using advanced machine learning models with hyperparameter tuning, the model learns situational tendencies and informs smarter, faster defensive play calls. Includes feature engineering for game context and offensive strategy, comparison of logistic regression, tree-based, and boosting algorithms, with thorough evaluation using ROC AUC, precision, recall, accuracy, and F1 metrics.


## Project Overview
This project predicts whether an NFL offense will **pass** or **rush** using play-by-play data from the `nfl_data_py` package.  

The dataset spans **2014–2023** and includes contextual features such as:
- Game situation (quarter time remaining, score differential)
- Offensive lineup (formation and personnel)
- Previous offensive strategy (previous play type, pass rate in drive)

I compared multiple machine learning models to assess predictive performance and identified which approaches are most effective for this task.

---

## Data
- **Package**: [`nfl_data_py`](https://pypi.org/project/nfl-data-py/)
- **Data Range**: 2014–2023 NFL seasons
- **Plays Included**: All Passing and Rushing plays

---

## Feature Engineering
Key features used:
- **Goal to go**, **Down**, **Yards to go:** Field position, down value, and distance to first down can influence play selection.
- **Quarter seconds remaining**, **Score differential:** Game conditions that may impact play type.
- **Offense formation:** Offensive alignment (e.g., shotgun, singleback) impacts play design.
- **Offense personnel:** Player grouping (e.g., 11 personnel = 1 RB, 1 TE) reflects strategic intent.

Advanced features created:
- **Previous play yards gained/lost:** Preceding play yardage outcome helps assess momentum.
- **Previous play type:** Records whether the last play was a pass or rush, providing recent playcalling context.
- **Offensive coach:** Identifies the coach as coaching style can influence decisions.
- **Play number in drive:** Counts how many plays have occurred in the current drive, indicating game flow and drive progression.
- **Total pass and rush plays in drive:** Tracks the cumulative number of pass and rush plays called so far in the drive, giving insight into the offensive strategy.
- **Pass rate in drive:** Calculates the proportion of pass plays within the current drive to capture passing tendencies.

---

## Models Used

- **Logistic Regression**
- **Random Forest**
- **Gradient Boosting**
---

## Evaluation Metrics
We evaluated models using:
- **Accuracy**: Overall correctness
- **F1 Score**: A balance between precision and recall, higher when both are high and penalizes extreme imbalances
- **ROC AUC**: Measures ranking quality (probability estimates)
- **Precision**: Percentage of predicted makes that were actually makes
- **Recall**: Percentage of actual makes that were predicted as makes

---

### Results Summary
- Due to the slightly skewed nature of the dataset towards more passing plays, we use ROC AUC as an additional evaluation metric.
- ROC AUC and accuracy values show the models overcome the class imbalance.
- We find that Gradient Boosting leads the pack, but the ROC AUC values are fairly similar at the top end of performance at ~0.80-0.82.

---

## Expected vs. Actual Performance

**Expectation**:  
- Gradient Boosting and Random Forest models would outperform Logistic Regression due to capturing the non-linear factors such as down and distance or formation and score differential.
- Logistic Regression would be a strong performer but lack the complexity to capture subtle patterns as those found in boosting models.
- `goal_to_go` would matter but not dominate the prediction.

**Actual**:  
- Logistic Regression models did perform the worst but performance was fairly consistent across all models.
- Gradient boosting models had the highest recall rates.
- Overall metrics were good but enhanced features and sequential models would further improve the results.

---

### Model Performance

| Model             | Config                                                             | Config (Short)                        | Accuracy | F1 Score | ROC AUC | Precision | Recall | Train Time (s) |
|------------------|--------------------------------------------------------------------|--------------------------------------|----------|----------|---------|-----------|--------|----------------|
| LogisticRegression | LogisticRegression_C=1.0                                           | LogisticRegression_C=1.0             | 0.7377   | 0.7786   | 0.8031  | 0.8167    | 0.7445 | 2.82           |
| LogisticRegression | LogisticRegression_C=0.1                                           | LogisticRegression_C=0.1             | 0.7386   | 0.7800   | 0.8037  | 0.8159    | 0.7475 | 1.52           |
| LogisticRegression | LogisticRegression_C=10                                            | LogisticRegression_C=10              | 0.7358   | 0.7760   | 0.8021  | 0.8177    | 0.7395 | 2.51           |
| RandomForest       | RandomForest_n_estimators=50_max_depth=None                       | RandomForest_n_estimators=50         | 0.7407   | 0.7929   | 0.8097  | 0.7861    | 0.7999 | 110.55         |
| RandomForest       | RandomForest_n_estimators=100_max_depth=10                        | RandomForest_n_estimators=100        | 0.7434   | 0.7890   | 0.8063  | 0.8057    | 0.7731 | 3.92           |
| RandomForest       | RandomForest_n_estimators=200_max_depth=20                        | RandomForest_n_estimators=200        | 0.7462   | 0.7883   | 0.8229  | 0.8173    | 0.7614 | 41.50          |
| GradientBoosting   | GradientBoosting_n_estimators=50_max_depth=3_learning_rate=0.1    | GradientBoosting_n_estimators=50     | 0.7492   | 0.7990   | 0.8208  | 0.7954    | 0.8031 | 8.11           |
| GradientBoosting   | GradientBoosting_n_estimators=100_max_depth=3_learning_rate=0.05  | GradientBoosting_n_estimators=100    | 0.7488   | 0.7987   | 0.8209  | 0.7950    | 0.8029 | 14.86          |
| GradientBoosting   | GradientBoosting_n_estimators=200_max_depth=5_learning_rate=0.01  | GradientBoosting_n_estimators=200    | 0.7496   | 0.8001   | 0.8230  | 0.7933    | 0.8074 | 57.79          |

## Project Structure

`DefCoordML.ipynb`: Main project notebook
`requirements.txt`: Python dependencies

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. Create and activate a virtual environment
macOS / Linux
```bash
python3 -m venv env
source env/bin/activate
```
Windows
```bash
python -m venv env
env\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook
```bash
jupyter notebook DefCoordML.ipynb
```
