# AI Usage

I used AI as a support tool during this project, mainly when I was stuck, needed to understand an problem statement, debug an issue, or wanted feedback on my approach. The implementation, testing and final decisions were done by me.

## 1. Where I Used AI

I used AI assistance for:

- Clarifying parts of the challenge requirements
- Discussing possible ML approaches and features
- Getting help with Python errors and debugging
- Reviewing some model results

## 2.  Usage of AI During Model Development

I discussed an anomaly-based approach with AI and then implemented and tested it. The approach compared a gateway's behaviour with its own previous four-week behaviour.

After testing, I got an average AUC of about 0.49 against the engineer review. Based on this result, I decided not to use it and moved to the Random Forest risk-ranking approach.

## 3. One Thing I thought wrong and Changed

During the final review, I noticed that the first reason-generation logic could give many gateways the same reason, such as "high recent disconnections". The logic checked fixed thresholds in order, so it did not always show which issue was strongest for a particular gateway.

I changed the reason logic to compare the relative strength of disconnections, offline duration and reboots and use the strongest one.

This was identified while checking my own output and was changed in the implementation based on the observed results.

## 4. Conclusion

AI was used as a development aid and like a mentor.

I personally implemented and tested the pipeline, checked the prediction cutoffs, verified the 15-gateway limit, checked for future-data leakage, evaluated the model on unseen gateways, compared it with the baseline and looked at network-shift behaviour.

The final model, results and decisions are based on the tests I ran on the challenge data.
