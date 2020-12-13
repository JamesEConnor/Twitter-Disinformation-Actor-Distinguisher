# Twitter Disinformation Actor Distinguisher

One of the largest problems facing social media platforms, law enforcement agencies, and general uses is the difficulty in attributing information operation accounts to specific  nation state actors. When accounts are identified as being part of an information operation, it takes additional work to identify which nation-state they belong to - a process [described by the Department of Homeland Security](https://www.dhs.gov/sites/default/files/publications/ia/ia_combatting-targeted-disinformation-campaigns.pdf) as a "painstaking process". This project was created with the goal of automating this process.

## Datasets Used

The utilized datasets were obtained from [Twitter's transparency report](https://transparency.twitter.com/en/reports/information-operations.html). The current model follows the below statistics in terms of attributable actors.

![latest dataset: april 2020](https://img.shields.io/badge/latest%20dataset-april%202020-blue)

![actor count: 15](https://img.shields.io/badge/actor%20count-15-blue)

![country count: 13](https://img.shields.io/badge/country%20count-13-blue)

## Accuracy Rate

When compared with predetermined labels in a separate testing dataset, the model achieved an accuracy rate of 90%.
