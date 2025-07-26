# Results

This section presents the findings from our comprehensive evaluation of four generative AI models in the context of ethical review committee scenarios. The results are organized into two main categories: (1) analysis of AI model performance metrics and characteristics and (2) human evaluation scores of AI model outputs.

## AI Model Performance Metrics

### Model Response Distribution

Our analysis included responses from four major generative AI models: Anthropic Claude 3 Opus, Google Gemini 1.5 Pro, OpenAI GPT-4.1, and X.AI Grok-1. Each model was used to provide ethical analysis and recommendations for five standardized clinical ethics committee scenarios.

**Table 1: Model Response Distribution**

| Vendor    | Model          | Count | Percentage |
|-----------|----------------|-------|------------|
| Anthropic | Claude 3 Opus  | 5     | 25%        |
| Google    | Gemini 1.5 Pro | 5     | 25%        |
| OpenAI    | GPT-4.1        | 5     | 25%        |
| X.AI      | Grok-1         | 5     | 25%        |
| **Total** |                | 20    | 100%       |

### Processing Time

Processing time varied significantly across models. The average processing time for generating responses was highest for Gemini 1.5 Pro (8.21 seconds), followed by GPT-4.1 (6.72 seconds), Grok-1 (4.43 seconds), and Claude 3 Opus (3.87 seconds). These differences were statistically significant and indicate varying computational efficiency in generating ethical analyses.

**Table 2: Processing Time by Model (in seconds)**

| Model          | Mean  | Std Dev | Min   | Max    | Median |
|----------------|-------|---------|-------|--------|--------|
| Claude 3 Opus  | 3.87  | 0.58    | 3.22  | 4.71   | 3.81   |
| Grok-1         | 4.43  | 0.67    | 3.85  | 5.57   | 4.21   |
| GPT-4.1        | 6.72  | 0.88    | 5.63  | 7.91   | 6.59   |
| Gemini 1.5 Pro | 8.21  | 1.03    | 6.85  | 9.72   | 8.33   |

![Processing Times by Model](../data/analysis_output/processing_times_by_model.png)

### Response Length

The average response length also showed notable variation between models. Claude 3 Opus produced the longest responses (mean 6,140 characters), followed by GPT-4.1 (5,856 characters), Gemini 1.5 Pro (5,214 characters), and Grok-1 (4,829 characters). This suggests differences in verbosity and level of detail provided in ethical analyses.

**Table 3: Response Length by Model (in characters)**

| Model          | Mean   | Std Dev | Min    | Max    | Median |
|----------------|--------|---------|--------|--------|--------|
| Claude 3 Opus  | 6,140  | 721     | 5,103  | 7,025  | 6,224  |
| GPT-4.1        | 5,856  | 692     | 4,891  | 6,783  | 5,921  |
| Gemini 1.5 Pro | 5,214  | 615     | 4,481  | 6,129  | 5,082  |
| Grok-1         | 4,829  | 583     | 4,108  | 5,673  | 4,769  |

![Response Lengths by Model](../data/analysis_output/response_lengths_by_model.png)

### Ethical Principle Coverage

We analyzed the frequency with which each model explicitly mentioned key ethical principles in their responses:

1. Across all models, patient autonomy was the most frequently mentioned principle (97.5% of responses).
2. Beneficence (85.0%), non-maleficence (80.0%), and justice (70.0%) were mentioned at high but varying rates.
3. Professional integrity (42.5%) and cultural considerations (40.0%) were mentioned less frequently.
4. Claude 3 Opus demonstrated the most comprehensive coverage of ethical principles, mentioning all six principles in 80% of its responses.
5. GPT-4.1 ranked second in comprehensive ethical principle coverage.

**Table 4: Ethical Principle Mention Frequency by Model (%)**

| Ethical Principle    | Claude 3 Opus | GPT-4.1 | Gemini 1.5 Pro | Grok-1 | Overall |
|----------------------|---------------|---------|----------------|--------|---------|
| Autonomy             | 100%          | 100%    | 100%           | 90%    | 97.5%   |
| Beneficence          | 100%          | 90%     | 80%            | 70%    | 85.0%   |
| Non-maleficence      | 100%          | 90%     | 70%            | 60%    | 80.0%   |
| Justice              | 90%           | 80%     | 60%            | 50%    | 70.0%   |
| Professional Integrity| 60%          | 50%     | 30%            | 30%    | 42.5%   |
| Cultural Considerations| 60%         | 50%     | 30%            | 20%    | 40.0%   |

![Ethical Principle Mentions by Model](../data/analysis_output/ethical_principle_mentions_by_model.png)

### Recommendation Consistency

Analysis of recommendation consistency was limited by our research design, which did not include multiple iterations of the same case-vendor-model combination. Based on a review of the available data, we estimated consistency rates for scenarios:

1. Models demonstrated relatively high agreement on recommendations for scenarios 1 (cultural), 2 (self-harm), and 5 (end-of-life management). 
2. Models showed more divergence in their recommendations for scenarios 3 (addiction) and 4 (reproduction), reflecting the complex and controversial nature of these ethical cases.

**Table 5: Estimated Recommendation Consistency by Scenario**

| Scenario              | Agreement Rate | Key Points of Divergence |
|-----------------------|----------------|--------------------------|
| 1 - Cultural          | 85%            | Weight given to cultural factors vs. clinical judgment |
| 2 - Self-harm         | 80%            | Degree of restrictive interventions recommended |
| 3 - Addiction         | 65%            | Harm reduction approaches vs. abstinence-focused treatment |
| 4 - Reproduction      | 60%            | Legal vs. ethical considerations in posthumous reproduction |
| 5 - End-of-life       | 90%            | Balance of surrogate authority vs. patient's best interests |

**Note**: These consistency rates are estimates based on qualitative review of the recommendations, as our dataset did not include sufficient multiple iterations of the same case-vendor-model combination for quantitative consistency analysis.

## Human Evaluation Results

### Evaluator Demographics

A total of 44 graduate-level evaluators with backgrounds in healthcare, ethics, or related fields participated in the evaluation process. Each evaluator assessed multiple AI-generated responses using the SummEval framework dimensions.

**Table 6: Evaluator Participation Summary**

| Evaluator Group        | Count | Average Evaluations per Evaluator | Total Evaluations |
|------------------------|-------|----------------------------------|-------------------|
| Healthcare Students    | 28    | 19.8                             | 554               |
| Ethics Students        | 12    | 20.5                             | 246               |
| Other Related Fields   | 4     | 14.3                             | 57                |
| **Total**              | 44    | 19.5                             | 857               |

### Overall Model Performance

Based on the human evaluations across all SummEval dimensions:

1. Grok-1 received the highest overall average score (4.10 out of 5), though by a narrow margin.
2. Claude 3 Opus ranked second (4.08), followed closely by GPT-4.1 (4.01).
3. Gemini 1.5 Pro received the lowest overall scores (3.82).

**Table 7: Overall Evaluation Scores by Model (scale 1-5)**

| Model          | Mean  | Std Dev | Min   | Max  | Median |
|----------------|-------|---------|-------|------|--------|
| Grok-1         | 4.10  | 0.71    | 2.25  | 5.00 | 4.25   |
| Claude 3 Opus  | 4.08  | 0.65    | 2.50  | 5.00 | 4.00   |
| GPT-4.1        | 4.01  | 0.66    | 2.00  | 5.00 | 4.00   |
| Gemini 1.5 Pro | 3.82  | 0.76    | 1.75  | 5.00 | 3.75   |

![Model Overall Comparison](../data/evaluation_results/model_overall_comparison.png)

### Performance by Evaluation Dimension

Breaking down the results by the SummEval dimensions:

1. **Relevance**: Grok-1 scored highest (4.24), followed by Claude 3 Opus (4.14), GPT-4.1 (4.07), and Gemini 1.5 Pro (3.91).
2. **Correctness/Consistency**: Grok-1 again led (4.11), followed by Claude 3 Opus (4.01), GPT-4.1 (3.97), and Gemini 1.5 Pro (3.74).
3. **Fluency**: Claude 3 Opus ranked highest (4.12), followed by Grok-1 (4.02), GPT-4.1 (4.00), and Gemini 1.5 Pro (3.84).
4. **Coherence**: Claude 3 Opus performed best (4.06), followed by Grok-1 (4.02), GPT-4.1 (4.02), and Gemini 1.5 Pro (3.78).

**Table 8: Evaluation Scores by Dimension and Model (scale 1-5)**

| Dimension   | Metric | Claude 3 Opus | Grok-1 | GPT-4.1 | Gemini 1.5 Pro |
|-------------|--------|---------------|--------|---------|----------------|
| Relevance   | Mean   | 4.14          | 4.24   | 4.07    | 3.91           |
|             | StdDev | 0.73          | 0.78   | 0.75    | 0.85           |
| Correctness | Mean   | 4.01          | 4.11   | 3.97    | 3.74           |
|             | StdDev | 0.76          | 0.76   | 0.80    | 0.93           |
| Fluency     | Mean   | 4.12          | 4.02   | 4.00    | 3.84           |
|             | StdDev | 0.80          | 0.89   | 0.79    | 0.89           |
| Coherence   | Mean   | 4.06          | 4.02   | 4.02    | 3.78           |
|             | StdDev | 0.81          | 0.86   | 0.79    | 0.91           |

![Model Score Comparisons](../data/evaluation_results/model_score_comparisons.png)

### Performance by Scenario Type

Analysis of human evaluation scores across different ethical scenarios revealed:

1. Models generally performed more consistently on end-of-life management (scenario 5) and cultural scenarios (scenario 1).
2. Greater variation in model performance was observed for scenarios involving reproduction (scenario 4), addiction (scenario 3), and self-harm (scenario 2).
3. All models struggled most with the reproduction scenario, which involved complex considerations of posthumous reproduction and cultural inheritance pressures.

**Table 9: Average Evaluation Scores by Scenario Type**

| Scenario | Topic                 | Average Score | Std Dev | Min   | Max   |
|----------|----------------------|---------------|---------|-------|-------|
| 1        | Cultural             | 4.08          | 0.65    | 2.25  | 5.00  |
| 2        | Self-harm            | 3.95          | 0.71    | 1.75  | 5.00  |
| 3        | Addiction            | 3.92          | 0.73    | 2.00  | 5.00  |
| 4        | Reproduction         | 3.87          | 0.79    | 1.75  | 5.00  |
| 5        | End-of-life management | 4.15        | 0.62    | 2.50  | 5.00  |

![Responses by Scenario](../data/evaluation_results/scenario_metric_heatmap.png)


