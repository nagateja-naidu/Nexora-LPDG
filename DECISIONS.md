Main goal : use the available data to rank the 15 gateways that should be considered for a visit each week, using only information that was available before that week's prediction.

## 1. What does "needs a visit" mean?

I defined a gateway as needing a visit when its recent and historical telemetry shows higher operational risk compared with other gateways.

I did not use past field visits as direct labels because visits were not random. Gateways with problems were more likely to be visited, so using visit history directly could add bias.

I also did not treat the engineer review as official ground truth. It is an engineer's opinion recorded on 15 February 2026.

An anomaly-only definition was also considered, but I found that unusual behaviour by itself does not always mean that a gateway should be selected.

## 2. Why did I choose Random Forest?

I first tested an anomaly-based approach that compared a gateway with its own previous four-week behaviour. Its average validation AUC was about 0.49, so I did not use it as the final ranking method.

I then used a Random Forest classifier. It works well with the telemetry features, can capture non-linear relationships, and is simple enough to inspect and explain.

From 16 February onward, the model uses the engineer review as a supervisory signal when that information was available before the prediction week. Earlier weeks use the telemetry-based fallback.

## 3. What data did I use?

The main input is gateway telemetry.

The features include:

- Offline duration
- Disconnections
- Reboots
- Mean, maximum and total values
- Active days
- Days with disconnections and reboots
- Previous-week values
- Four-week averages
- Recent trends
- Overall severity and severity change

These features are chosen to capture both the current condition of a gateway and checking whether behaviour is changing.

The information from the 15 February engineer review was not used for the 2 February or 9 February prediction cutoffs.

## 4. Why exactly 15 gateways?

The challenge allows a maximum of 15 site visits each week. I therefore rank the gateways by score and select the top 15.

I did not use a fixed probability threshold because that could result in a different number of visits each week.

The cost information was also considered. A visit costs €380, while leaving a broken gateway unattended costs €600 per week. During development, I used these values for an offline comparison with the baseline.

The offline comparison was:

- 3-sigma baseline: €330,280
- ML V2: €280,300
- Difference: €49,980
- Reduction: 15.13%

These numbers are a development analysis, not the official challenge score, because the official fault outcomes are hidden.

## 5. Why did I choose Machine Learning as Part 2?

I chose Machine Learning because it matches the work I wanted to explore most deeply: learning useful patterns from telemetry and testing whether the model can generalise.

I used gateway-disjoint validation so that validation gateways were not the same gateways used for training. The mean AUC from 16 February onward was about 0.87.

I also checked later prediction weeks and a network-shift scenario. These tests are important because a model can perform well on known data but behave differently when it sees new gateways or changes in network behaviour.

## What it cannot do

The biggest limitation is that the official hidden fault outcomes are not available. The engineer review is useful, but it is not the same as the final ground truth.

Field visits are also not randomly assigned, which makes them difficult to use as direct labels.

The cost comparison is based on an offline proxy analysis and does not fully represent every possible fault episode or repeated visit.

With another two weeks of data and outcomes, I would check which telemetry patterns actually lead to faults, which high-risk gateways recover without a visit, and whether network changes are affecting the ranking.

## Conclusion

I chose a time-aware Random Forest ranking system with a telemetry fallback.

The main principles were:

1. Use only information available before each prediction week.
2. Rank gateways instead of using an arbitrary probability cutoff.
3. Use engineer review only when it was available.
4. Test on gateways not seen during training.
5. Compare the approach with the provided baseline.
6. Be clear about what the available data cannot prove.
