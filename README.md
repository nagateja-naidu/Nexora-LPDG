
## 1. Problem Statement

The challenge is to identify the **15 gateways that should be visited by an engineer each week**.

We are given historical telemetry data from a set of gateways. The telemetry contains information such as:

- `gateway_id` - unique identifier of each gateway
- `ts_utc` - timestamp of the observation
- `offline_duration_sec` - how long the gateway was offline
- `disconnection_cnt` - number of disconnections
- `reboot_cnt` - number of reboots

The goal is not simply to find gateways with the highest number of failures. The goal is to **prioritize the gateways that are most likely to need an engineering visit**, using only information that would have been available before making the prediction.

For every week, the system must produce exactly **15 gateways**, ordered from rank 1 to rank 15.

The required output contains:

```text
week_start
rank
gateway_id
score
reason# AdaptiveRAG
```

## 2. Given Baseline

The challenge already provides a **3-Sigma Baseline**, which I used as the starting point for my approach.

The baseline looks at the past behavior of each gateway and checks whether its recent telemetry is unusually high.

It mainly uses these three signals:

- `offline_duration_sec`
- `disconnection_cnt`
- `reboot_cnt`

It uses the gateway's recent history to identify abnormal behavior and then ranks the gateways based on those abnormal events.

Finally, it selects the **top 15 gateways for each week**, since only 15 gateways can be visited by the engineer.

The baseline also provides a reason for selecting a gateway, such as high disconnections, high offline duration, or frequent reboots.

I used this baseline as my reference point. For Part 2, I built my own machine learning approach and compared it against the baseline.

## 3. My Approach

After understanding the baseline, I wanted to build a model that could use multiple telemetry signals instead of depending mainly on abnormal threshold detection.

For Part 2, I used a **Random Forest Classifier**.

The main idea is to combine the recent behavior of each gateway with its changes over time and use these features to predict which gateways are more likely to need attention.

### Features I used

I created weekly features from the telemetry data, including:

- `offline_duration_sec`
- `disconnection_cnt`
- `reboot_cnt`
- Mean values
- Standard deviation
- Maximum values
- Total values
- Number of observations
- Number of active days
- Days with disconnections
- Days with reboots
- Previous week's values
- Recent trends
- Four-week average
- Change from the recent average

I also used the available **engineer review data** as historical labels for training the model.

### Why I used Random Forest

I selected Random Forest because it can work well with a combination of different numerical features and can learn relationships between multiple signals.

For example, instead of looking only at whether the number of disconnections is unusually high, the model can consider the disconnections together with offline duration, reboots, recent trends, and other gateway behavior.

### How I make the final ranking

I did not use the Random Forest prediction alone.
I combined the machine learning score with a telemetry-based risk score:
- I aggregate the raw telemetry data week by week.
- I create useful features from the weekly telemetry.
- I use these features as input to the Random Forest model.
- The Random Forest produces an ML risk score for each gateway.
- I calculate a separate telemetry-based risk score.
- I combine both scores to get the final gateway score.
- I rank all gateways using the final score.
- I select the top 15 gateways for each prediction week.

## 4. Detailed Workflow

### Step 1: Load the Telemetry Data

I read the telemetry files from the `data/telemetry` folder.

The main information I use is:

- `gateway_id`
- `ts_utc`
- `offline_duration_sec`
- `disconnection_cnt`
- `reboot_cnt`

### Step 2: Create Weekly Features

I group the telemetry data by gateway and week.

For each gateway, I calculate features such as:

- Mean
- Standard deviation
- Maximum
- Total
- Number of observations
- Number of active days
- Days with disconnections
- Days with reboots

This converts the raw telemetry into a weekly format that can be used by the model.

### Step 3: Add Time-Based Features

I also look at how the gateway's behavior changes over time.

I create features such as:

- Previous week's value
- Recent trend
- Trend ratio
- Four-week average
- Difference from the recent average

This helps the model understand whether a gateway is becoming better or worse over time.

### Step 4: Create a Telemetry Risk Score

Along with the machine learning model, I calculate a separate risk score using the main telemetry signals.

The score considers:

- Offline duration
- Disconnections
- Reboots
- Days with disconnections
- Days with reboots

This gives a direct measure of the recent risk of each gateway.

### Step 5: Use Engineer Review Data

I use the historical engineer review file to identify gateways that were previously marked as requiring attention.

These historical labels are used to train the Random Forest model.

I also make sure that only reviews available before the prediction week are used.

### Step 6: Train the Random Forest

The engineered features are given to the Random Forest model.

The model learns patterns from the historical data and produces a risk score for each gateway.

I use class balancing in the model because the review labels are not perfectly balanced.

### Step 7: Combine the Scores

I combine the Random Forest score with the telemetry risk score.

The machine learning score has the main weight, while the telemetry score is also kept as part of the final ranking.

This gives the final score for each gateway.

### Step 8: Rank the Gateways

I sort all gateways based on their final score.

The gateways with higher scores are placed higher in the ranking.

### Step 9: Select 15 Gateways

I select exactly **15 gateways for each prediction week**.

Each selected gateway receives:

- Rank
- Gateway ID
- Score
- Reason for selection

### Step 10: Generate the Final Submission

The final predictions are written to:

`predictions.csv`

The file contains:

- `week_start`
- `rank`
- `gateway_id`
- `score`
- `reason`

Finally, I run the validation scripts to check the submission format and the required evaluation checks.

## 5. Project Files

### `src/generate_predictions.py`
Main file for my ML solution.

It loads the data, uses the engineered features, trains the Random Forest model, calculates the risk scores, ranks the gateways, and generates the final `predictions.csv`.

### `src/feature_engineering.py`
Contains the feature engineering part of my ML pipeline.

It converts raw telemetry into weekly gateway-level features and creates features based on recent behavior, previous weeks, trends, and four-week averages.

### `src/baseline_3sigma.py`
Contains the provided 3-Sigma baseline implementation.

It generates the baseline gateway rankings for comparison.

### `run.py`
Runs my complete pipeline in one command.

It generates the predictions and runs the required validation scripts.

### `run_baseline.py`
Runs the 3-Sigma baseline separately.

### `src/validate_submission.py`
Checks whether `predictions.csv` follows the required submission format.

### `src/validate_temporal.py`
Checks the temporal setup of the prediction process.

### `src/validate_unseen_gateways.py`
Checks the model's behavior on gateways that were not used for training.

### `src/validate_network_shift.py`
Checks the model when network conditions change.

### `predictions.csv`
Contains the final predictions generated by my model.

It contains the top 15 gateways for each of the 8 prediction weeks.

## 6. Comparision with the Baseline

I compare my ML approach with the given 3-Sigma baseline using the **total maintenance cost**.

The challenge uses:

- €380 for each engineer visit
- €600 per week for each broken gateway that is left unattended
- Maximum 15 visits per week

For my offline evaluation, I assumed the maintenance cost formula as:

```text
Total Cost = (Number of Visits × €380) + (Unattended Broken Gateway-Weeks × €600)
```
I applied the same calculation to both the baseline and my ML approach.

The estimated cost of my ML approach is lower than the baseline by:

### €330,280 - €280,300 = €49,980


## 7. Conclusion

In this project, I started with the given 3-Sigma baseline and then built a machine learning approach using Random Forest.

My approach:

- Uses multiple telemetry signals instead of only checking individual abnormal values.
- Creates weekly features from the raw telemetry.
- Includes recent behavior and changes over time.
- Uses historical engineer review data for training.
- Combines the ML score with a telemetry-based risk score.
- Ranks all gateways based on the final score.
- Selects exactly 15 gateways for each week.
- Generates a simple reason for every selected gateway.
- Validates the final predictions using the provided validation scripts.

I then compare my approach with the 3-Sigma baseline using the maintenance cost formula.

The main goal is to identify the gateways that should receive the limited maintenance visits while keeping the estimated maintenance cost low.


##LINK FOR THE SCREEN RECORDING VIDEO 
https://drive.google.com/file/d/1klU74vnkwsIBC31JdHAJaqE8Vb5szENl/view?usp=drive_link

