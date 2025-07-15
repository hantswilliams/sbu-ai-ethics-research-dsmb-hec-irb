# Methods

## Study Design Overview

This study employed a systematic evaluation approach to assess the capabilities of generative AI models in healthcare ethics decision-making contexts. We utilized a standardized prompt-based testing methodology applied to real-world clinical ethics scenarios, with subsequent analysis of AI-generated recommendations compared to documented human ethics committee decisions.

## Ethics Scenario Selection

### Data Source
We selected eleven real-world clinical ethics case studies from publicly documented cases at Brigham and Women's Hospital, available through their Clinical Ethics Case Review repository (https://bwhclinicalandresearchnews.org/clinical-ethics-case-review/). These cases were chosen to represent a diverse range of ethical dilemmas commonly encountered in healthcare settings, including:

- End-of-life care decisions
- Psychiatric emergencies
- Infectious disease management
- Organ transplantation considerations
- Resource allocation challenges
- Pediatric and neonatal care dilemmas

### Selection Criteria
Cases were selected based on:
1. Complexity of ethical considerations
2. Clear documentation of the clinical scenario
3. Availability of the actual ethics committee decision or recommendation
4. Relevance to current healthcare ethics practices
5. Representation of diverse ethical principles (autonomy, beneficence, non-maleficence, justice)

## Generative AI Models

The study evaluated two leading generative AI models:

1. **OpenAI ChatGPT** (version available as of July 2025)
   - Model characteristics: [specific parameters]
   - Access method: OpenAI API

2. **Google Gemini** (version available as of July 2025)
   - Model characteristics: [specific parameters]
   - Access method: Google API

Models were selected based on their widespread use, advanced reasoning capabilities, and different architectural approaches, providing a representative sample of current generative AI technology.

## Prompt Engineering

### Standardized Prompt Framework
We developed a standardized prompt framework to ensure consistent evaluation across scenarios and models. The prompt was structured to:

1. Define the AI's role as an Ethics Advisor embedded within a Hospital Ethics Committee
2. Specify responsibilities including ethical analysis and recommendation generation
3. Establish guiding ethical principles (autonomy, beneficence, non-maleficence, justice)
4. Delineate scope and limitations of AI decision-making
5. Prescribe a structured response format

### Prompt Structure
The standardized prompt instructed the AI to provide responses in the following format:
1. Brief Clinical Scenario Restatement
2. Identification and Explanation of Relevant Ethical Principles and Tensions
3. Systematic Ethical Analysis
4. Explicitly Stated Recommended Clinical Decisions (Ranked Medical Recommendations):
   - Recommended Decision (Best Medical Option)
   - Alternative Decision (Second-Best Medical Option)
   - Least-Recommended Decision (Third Medical Option)

The complete prompt template is available in the supplementary materials.

## Data Collection Procedure

### Testing Protocol
Each ethics scenario was processed through both AI models using the standardized prompt. To account for potential variability in AI responses, we conducted three separate iterations for each scenario-model combination, resulting in a total of 66 AI-generated responses (11 scenarios × 2 models × 3 iterations).

Testing was conducted between [Month-Month 2025], with all prompts submitted through the respective model APIs using consistent parameter settings to ensure reproducibility.

### Response Capture
AI-generated responses were systematically captured in their entirety, preserving all content, formatting, and metadata. Responses were stored in a structured database that included:
- AI Model identifier
- Scenario identifier
- Iteration number
- Timestamp
- Complete response text
- Extracted recommended decisions (primary, secondary, tertiary options)

## Evaluation Framework

### Human Evaluation Using SummEval Framework
To systematically assess the quality of AI-generated ethical analyses and recommendations, we implemented a comprehensive human evaluation process based on the SummEval framework (Fabbri et al., TACL 2021). This widely-used four-way rubric provides a structured approach for evaluating AI-generated content across key dimensions.

#### Evaluator Selection and Training
We recruited 35 graduate students from healthcare ethics, bioethics, and healthcare management programs to serve as evaluators. All evaluators underwent a standardized training session that included:
1. An overview of the SummEval rubric and scoring criteria
2. Review of anchor examples for each dimension and score point
3. Practice evaluations with feedback from clinical ethics experts
4. Instruction on avoiding common biases in AI evaluation

#### Evaluation Procedure
Each evaluator was assigned a subset of AI-generated responses along with the corresponding original case scenario and the actual ethics committee decision. For each AI response, evaluators assessed four key dimensions using a 5-point Likert scale (1=poor, 5=excellent):

1. **Relevance**: The degree to which the AI response addresses the core ethical issues presented in the case scenario.
   - Evaluates whether the AI identifies the most pertinent ethical considerations
   - Assesses appropriateness of recommended actions to the specific scenario
   - Measures alignment with the clinical context and constraints

2. **Consistency/Correctness**: The factual accuracy and ethical soundness of the AI analysis and recommendations.
   - Evaluates alignment with established bioethical principles and clinical guidelines
   - Assesses internal consistency of ethical reasoning
   - Measures accuracy of factual statements and clinical assumptions

3. **Fluency**: The linguistic quality and clarity of the AI-generated text.
   - Evaluates grammatical correctness and appropriate terminology
   - Assesses readability and professional tone
   - Measures appropriate use of medical and ethical terminology

4. **Coherence**: The logical organization and flow of the ethical analysis.
   - Evaluates logical progression of ethical reasoning
   - Assesses clear connections between ethical principles and recommended actions
   - Measures overall structure and framework of the ethical analysis

Each evaluator was required to provide a brief justification for their ratings, particularly for scores at the extremes (1 or 5). Evaluators were blinded to which AI model generated each response to minimize bias.

#### Quality Control
To ensure evaluation reliability and consistency:
1. 20% of responses were randomly assigned to multiple evaluators to calculate inter-rater reliability using Cohen's Kappa
2. Evaluators showing systematic deviation from group norms received additional training
3. Regular calibration meetings were held to address questions and maintain consistent application of the rubric

### Quantitative Assessment
In addition to the SummEval dimensions, responses were evaluated using supplementary quantitative metrics:

1. **Recommendation Concordance**: Degree of alignment between AI recommendations and the actual ethics committee decisions (scored on a 0-3 scale)
2. **Ethical Principle Coverage**: Comprehensiveness in addressing the four core bioethical principles (scored on a 0-4 scale)
3. **Response Consistency**: Variation in recommendations across iterations (calculated as standard deviation)

### Qualitative Analysis
A team of three researchers with backgrounds in clinical ethics and AI conducted thematic analysis of the responses to identify:
1. Patterns in ethical reasoning approaches
2. Strengths and limitations in AI ethical analysis
3. Novel insights or perspectives offered by AI systems
4. Areas of ethical oversight or potential bias
5. Implications for practical implementation in ethics committees

## Data Analysis

### SummEval Metrics Analysis
The SummEval ratings from student evaluators were analyzed as follows:

1. **Descriptive Statistics**:
   - Mean scores and standard deviations for each dimension (Relevance, Consistency/Correctness, Fluency, Coherence)
   - Distribution of scores across the 5-point scale for each dimension
   - Composite scores combining all four dimensions

2. **Comparative Analysis**:
   - Between-model comparisons (OpenAI vs. Google) using independent samples t-tests
   - Between-case comparisons to identify scenario-dependent performance using ANOVA
   - Correlations between dimensions to identify relationships between different aspects of performance

3. **Reliability Analysis**:
   - Inter-rater reliability calculated using Cohen's Kappa for categorical ratings
   - Intraclass correlation coefficient (ICC) for continuous ratings
   - Analysis of evaluator consistency across different case types

4. **Factor Analysis**:
   - Principal component analysis to identify underlying patterns in evaluation dimensions
   - Assessment of whether the four dimensions capture distinct aspects of AI performance

### Additional Statistical Methods
Supplementary quantitative data were analyzed using:

- Mean scores and standard deviations for each evaluation dimension
- Frequency distributions of recommendation concordance
- Comparative analysis between models using paired t-tests
- Correlation analysis between SummEval dimensions and other metrics

### Qualitative Synthesis
Thematic analysis results from both evaluator comments and researcher analysis were synthesized using a structured framework approach to identify:
- Common themes across responses
- Model-specific patterns or tendencies
- Scenario-dependent variations in performance
- Implications for real-world implementation

### Visualization and Reporting
Results were visualized using:
- Radar charts displaying performance across the four SummEval dimensions
- Box plots showing score distributions by model and case type
- Heat maps illustrating correlations between dimensions and other performance metrics

Comprehensive data tables with mean scores, standard deviations, and confidence intervals are provided in the supplementary materials.

## Limitations

This study acknowledges several methodological limitations:

1. The restricted sample of publicly available ethics cases may not represent the full spectrum of healthcare ethics dilemmas
2. Models were evaluated at a specific point in time and may not reflect subsequent improvements
3. The standardized prompt, while comprehensive, represents one of many possible frameworks for AI ethics guidance
4. The comparison to documented ethics committee decisions assumes those decisions were optimal
5. The evaluation framework, while multi-dimensional, cannot capture all aspects of ethical reasoning quality

These limitations are addressed in the discussion section along with their implications for interpretation of findings.
