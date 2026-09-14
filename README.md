# Churn Risk Scoring

Which telecom customers are about to leave, and what does it cost to go after them?

This repo scores 7,043 customers on churn risk, then works out where to draw the line between "send a retention offer" and "leave alone". The model is a logistic regression. The interesting part is what you do with its output.

---

## What the data says

**Contract type is the whole story.** Customers on month-to-month contracts churn at 43%. Customers on two-year contracts churn at 2.8%. That is a 15x gap, and no other variable comes close. Mutual information ranks `contract` at 0.098, roughly 1.5x the next strongest feature.

**Paying by electronic check is the second flag.** 46% of those customers churn, against a 27% base rate. It stays significant in the model even after contract type is accounted for, so it is not just a proxy for month-to-month billing.

**Fiber optic customers churn more than DSL customers.** 43% against 19%. Fiber is the premium product, so this is worth someone's attention.

**Customers without tech support or online security churn at ~42%.** Both roughly 1.55x the base rate. These are cheap add-ons.

**Customers with no internet service at all churn at 7.8%.** They pay the least and stay the longest.

## Who to target

Ranking every customer by predicted risk and taking the top slice:

| Take the top | Churn rate in that slice | Lift | Share of all churners caught |
|---|---|---|---|
| 10% | 76% | 3.1x | 31% |
| 20% | 65% | 2.6x | 53% |
| 30% | 57% | 2.3x | 69% |

Your first 10% of spend reaches customers who churn at three times the average rate. After the top 30% it flattens out fast.

The threshold you pick decides whether you tolerate false alarms or missed churners:

| Threshold | Customers flagged | Precision | Recall |
|---|---|---|---|
| 0.3 | 38% | 0.51 | 0.80 |
| 0.4 | 31% | 0.56 | 0.71 |
| 0.5 | 23% | 0.63 | 0.59 |
| 0.6 | 15% | 0.70 | 0.42 |
| 0.7 | 7% | 0.81 | 0.23 |

## What it's worth

Precision and recall are abstract until you attach money to them. Assume a €50 retention offer, a 30% acceptance rate among customers who would otherwise have left, and 12 months of retained revenue at their current monthly charge. All three are assumptions, not facts from the data, and the whole table moves if you change them.

Run against the 1,409-customer test set:

| Threshold | Offers sent | Campaign cost | Revenue saved | Net | ROI |
|---|---|---|---|---|---|
| 0.2 | 679 | €33,950 | €83,542 | **€49,592** | 2.5x |
| 0.3 | 541 | €27,050 | €76,490 | **€49,440** | 2.8x |
| 0.4 | 440 | €22,000 | €69,422 | €47,422 | 3.2x |
| 0.5 | 326 | €16,300 | €59,381 | €43,081 | 3.6x |
| 0.7 | 98 | €4,900 | €22,837 | €17,937 | 4.7x |
| Everyone | 1,409 | €70,450 | €92,823 | €22,373 | 1.3x |

Two things fall out of this.

Total net value peaks around a threshold of 0.2 to 0.3, but ROI keeps climbing as you get stricter. Those are different questions. If retention budget is fixed and small, go to 0.6 or higher and spend it on the customers most likely to leave. If the budget is flexible and the goal is maximum revenue retained, drop to 0.3 and accept that half your offers go to people who were staying anyway.

Targeting beats blanketing by a wide margin. Sending the offer to everyone retains slightly more revenue in absolute terms, but costs €70k to net €22k. Scoring first nets more than twice that for a third of the spend.

At a €50 offer the campaign breaks even once about 9% of targeted churners accept. Raise the offer to €200 and you need 33%. Nobody has measured the real acceptance rate, so that is the number worth chasing before anything gets sent.

## What this doesn't tell you

The contract finding is correlational and probably confounded. Customers who sign two-year contracts are likely the ones who already intended to stay. Moving a wavering customer onto a two-year contract will not automatically drop their churn risk to 2.8%. Answering that properly needs an experiment, not this dataset.

Same caution on fiber optic. The higher churn could be price, could be service quality, could be that fiber customers are in competitive urban markets with more alternatives. The data has no field that separates these.

The 30% acceptance rate is invented. Nothing in this dataset records whether anyone was ever offered anything. It is a placeholder until someone runs a real campaign and measures it.

There is also no time dimension here. Each row is one snapshot, so the model says "this customer looks like someone who churns", not "this customer will churn in the next 90 days". For a retention campaign you usually want the second one.

## Model performance

Logistic regression with one-hot encoded categoricals, `C=1.0`, `liblinear` solver.

| Metric | Value |
|---|---|
| 5-fold CV AUC | 0.841 ± 0.007 |
| Held-out test AUC | 0.858 |
| Accuracy at 0.5 threshold | 0.813 |
| Accuracy of always guessing "stays" | 0.753 |

That last row matters. The model beats the do-nothing baseline by 6 percentage points of accuracy, which is not much. AUC is the honest metric here, because the model's value is in *ranking* customers by risk, not in labelling them.

Strongest coefficients:

| Pushes toward churn | | Pushes away from churn | |
|---|---|---|---|
| month-to-month contract | +0.54 | two-year contract | −0.53 |
| fiber optic | +0.33 | DSL | −0.36 |
| senior citizen | +0.21 | phone service on | −0.25 |
| electronic check | +0.21 | online security on | −0.22 |

Logistic regression was chosen over a tree ensemble deliberately. A gradient-boosted model would gain maybe 0.01 AUC and cost the ability to say "this customer scored high because of the contract and the payment method", which is the sentence the retention team actually needs.

## Repo

```
data/churn.csv                  Telco customer data, 7,043 rows, 21 columns
notebooks/churn_project.ipynb   Full analysis: cleaning, EDA, feature importance,
                                evaluation, cross-validation, and the retention
                                budget model behind the tables above

train.py                        Reproduces the final model and writes model.bin

deployment/                     Flask service that scores a single customer

requirements.txt

```

## Running it

```bash
pip install -r requirements.txt
python train.py
```

That prints per-fold AUC, trains on the full training set, evaluates once on the test set, and saves `model.bin`.

To serve predictions:

```bash
cd deployment
python predict.py
```

Then in another terminal, `python predict-test.py` posts a sample customer and prints the risk score.
