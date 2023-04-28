# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The model uses Census data to predict whether or not someone's salary is above or below \$50,000

## Intended Use

The model might be used for exploring features associated with higher salaries and how those change over time

## Training Data

The training data is an 80% split from the overall Census data--- the unit of analysis is an individual

## Evaluation Data

The evaluation is a 20% split from the overall Census data

## Metrics

We examine three accuracy statistics: precision, recall, and the Fbeta measure. The model's values on these statistics are:

- Precision: 0.71
- Recall: 0.25
- Fbeta: 0.37

These values indicate that we are relatively good at avoiding false positives, or predicting a $>$\$50,000 salary in cases where we do not attain one. However, the recall measure shows that we have a non-trivial number of false negatives- or people who attain that salary but who we predict they dont. Overall, the performance metrics indicate the need for substantial improvement.

[Subgroup metrics](https://github.com/rebeccajohnson88/udacity_apideploy/blob/master/slice_output.txt): shows the metrics separately for different levels of educational attainment. We see the models perform best for those with a BA or higher, and especially those who attend professional school.

## Ethical Considerations

The low accuracy for those outside the major buckets of education introduces ethical concerns/the value of future preprocessing into coarser educational categories.

## Caveats and Recommendations

This was trained on a particular cohort of workers so may not have accurate predictions over time as educational and labor market patterns change.
