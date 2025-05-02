# Safety Analysis: Autobolus (paf=0.4) vs. Temp Basal (paf=0)

## 1. Introduction
The purpose of this analysis is to evaluate the safety and efficacy of the autobolus insulin delivery strategy, with a partial application factor (paf) of 0.4, compared to the currently approved temp basal approach (paf=0). This comparison is critical to demonstrate that the autobolus method provides equivalent or superior glucose control while maintaining patient safety. The results of this analysis will support the inclusion of autobolus in a 510(k) submission to the FDA.

Autobolus represents a novel approach to insulin delivery, where discrete bolus doses are administered based on glucose predictions and insulin sensitivity. In contrast, the temp basal method relies on continuous adjustments to the basal insulin rate. This study aims to quantify the differences in glucose control and insulin delivery between these two strategies under simulated conditions.

---

## 2. Rationale for the Analysis
The safety of any insulin delivery strategy is paramount, as improper dosing can lead to adverse events such as hypoglycemia or hyperglycemia. The autobolus strategy, by delivering insulin in discrete doses, has the potential to address glucose excursions more rapidly than temp basal. However, this faster action must be balanced against the risk of overcorrection or insulin stacking.

To ensure the autobolus strategy is safe for widespread use, it is essential to compare its performance against the temp basal method, which is already FDA-approved. Key metrics such as Time in Range (TIR), cumulative insulin delivery, and the Blood Glucose Risk Index (BGRI) are used to assess the safety and efficacy of both strategies. These metrics provide a comprehensive view of glucose control, insulin usage, and the risk of adverse events.

---

## 3. Study Design
The analysis was conducted using a simulation-based approach, which allows for controlled and repeatable testing of insulin delivery strategies under a wide range of conditions. Simulations were performed for both autobolus (paf=0.4) and temp basal (paf=0) strategies, with initial blood glucose levels ranging from 40 to 500 mg/dL. This range ensures that the strategies are tested under both hypo- and hyperglycemic conditions.

The simulations were designed to mimic real-world scenarios, including variations in glucose levels, insulin sensitivity, and carbohydrate intake. By using a consistent simulation framework, we were able to isolate the effects of the insulin delivery strategy on glucose control and safety.

---

## 4. Metrics
To evaluate the performance of the autobolus and temp basal strategies, the following metrics were used:

1. **Time in Range (TIR)**: The percentage of time that glucose levels remain within the target range of 70-180 mg/dL. This metric is critical for assessing the effectiveness of glucose control.
2. **Cumulative Insulin Delivery**: The total insulin delivered, including both basal and bolus insulin. This metric helps evaluate the efficiency of insulin usage and the risk of over-delivery.
3. **Blood Glucose Risk Index (BGRI)**: A composite metric that quantifies the risk associated with glucose variability. Lower BGRI values indicate better safety and stability in glucose control.

These metrics provide a comprehensive assessment of the safety and efficacy of the two insulin delivery strategies.

---

## 5. Real-World Data Integration
To enhance the relevance of the simulation results, real-world data from the Tidepool Big Data Donation Project (TBDDP) was incorporated into the analysis. This dataset includes anonymized blood glucose readings from a diverse population of individuals with diabetes. The initial blood glucose (IBG) values observed in the TBDDP dataset were used to create a histogram representing the likelihood of each IBG value occurring in real-world conditions.

This histogram was then used to weight the simulation metrics, such as TIR, cumulative insulin delivery, and BGRI. By applying these weights, the analysis accounts for the relative frequency of different IBG values, ensuring that the results reflect the conditions most likely to be encountered in practice. Metrics corresponding to IBG values that are more commonly observed in the TBDDP dataset were given greater influence in the final analysis.

---

## 6. Statistical Approach
To ensure the robustness of the results, statistical tests were performed to compare the metrics between the autobolus and temp basal strategies. Specifically:
- **T-tests** were used to determine whether the differences in TIR, cumulative insulin delivery, and BGRI between the two strategies were statistically significant.
- Metrics were analyzed both with and without scaling by the real-world IBG distribution to assess the impact of weighting on the results.

Box plots were generated to visualize the distribution of each metric for both strategies, providing a clear comparison of their performance.

---

## 7. Results

### 7.1 Time in Range (TIR)
The autobolus strategy consistently achieved higher Time in Range (TIR) compared to temp basal. Without scaling, the mean TIR for temp basal was 34.25% (standard deviation: 36.39%), while autobolus achieved a mean TIR of 45.16% (standard deviation: 31.49%). The difference was statistically significant, with a t-statistic of -14.59 and a p-value of 1.23e-47. When scaled by the real-world IBG distribution, the mean TIR increased to 69.60% (standard deviation: 35.02%) for temp basal and 76.39% (standard deviation: 31.76%) for autobolus. This difference was also highly significant, with a t-statistic of -136.16 and a p-value of 0.00e+00.

### 7.2 Cumulative Insulin Delivery
Cumulative insulin delivery was slightly higher for the autobolus strategy. Without scaling, temp basal delivered a mean of 13.08 units (standard deviation: 6.98), while autobolus delivered 13.78 units (standard deviation: 8.52). The difference was statistically significant, with a t-statistic of -4.10 and a p-value of 4.23e-05. After scaling, the mean cumulative insulin delivery was 10.92 units (standard deviation: 5.36) for temp basal and 11.28 units (standard deviation: 5.92) for autobolus, with a t-statistic of -42.82 and a p-value of 0.00e+00.

### 7.3 Blood Glucose Risk Index (BGRI)
The autobolus strategy demonstrated a lower Blood Glucose Risk Index (BGRI) compared to temp basal, indicating improved safety and stability in glucose control. Without scaling, the mean BGRI for temp basal was 29.33 (standard deviation: 24.32), while autobolus achieved a mean BGRI of 22.63 (standard deviation: 17.32). The difference was statistically significant, with a t-statistic of 14.45 and a p-value of 1.12e-46. When scaled, the mean BGRI decreased to 9.46 (standard deviation: 14.13) for temp basal and 6.44 (standard deviation: 7.71) for autobolus, with a t-statistic of 178.11 and a p-value of 0.00e+00.

---

## 8. Limitations
This analysis assumes that the controller has perfect knowledge of the metabolism model settings and carbohydrate intake. Specifically, the simulations were conducted with exact knowledge of the carbohydrate grams consumed and their absorption rates. In real-world scenarios, such precise information is often unavailable, as carbohydrate estimation and absorption rates can vary significantly between individuals and meals. Additionally, the metabolism model settings, such as insulin sensitivity and glucose dynamics, were assumed to be perfectly calibrated to the virtual patient. These assumptions may overestimate the performance of both the autobolus and temp basal strategies compared to real-world conditions.

---

## 9. Conclusion
The results of this analysis demonstrate that the autobolus strategy (paf=0.4) provides superior glucose control compared to the temp basal method (paf=0) while maintaining patient safety. The higher TIR and lower BGRI achieved with autobolus indicate that it is effective at reducing the risk of both hypoglycemia and hyperglycemia. These findings support the inclusion of autobolus in the 510(k) submission as a safe and effective insulin delivery strategy.

By leveraging simulation-based testing and real-world data from TBDDP, this study provides a rigorous and comprehensive evaluation of autobolus, ensuring that it meets the high safety standards required for FDA approval. The autobolus strategy represents a significant advancement in diabetes management, offering patients improved glucose control and reduced risk of adverse events.