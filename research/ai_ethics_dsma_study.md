### Study Design: Evaluating Ethical Decision-Making Capabilities of Generative AI for IRB, HEC, DSMB, and DSMA Implementation

#### Background and Rationale
Generative AI (LLMs such as GPT-4, Claude, and Gemini) has potential for augmenting ethical decision-making processes in healthcare settings. Prior research demonstrates mixed success, indicating promise but emphasizing the need for structured testing and validation. This study proposes evaluating several prominent LLMs through structured, prompt-based scenarios relevant to IRBs, Hospital Ethics Committees (HECs), and Data Safety Monitoring Boards (DSMBs), and introduces the concept of a Data Safety Monitoring Agent (DSMA)—an AI-driven continuous monitoring entity.

#### Objectives
1. Evaluate and compare the ethical reasoning capabilities of multiple generative AI models.
2. Assess AI performance across clinically relevant ethical scenarios.
3. Develop preliminary guidelines and criteria for a Data Safety Monitoring Agent (DSMA).

#### Methodology

##### Step 1: Scenario Development
Develop 5 structured, clinically relevant ethical scenarios:
- Organ transplantation allocation dilemma
- Oncology treatment recommendation conflict
- Clinical trial interim safety analysis (oncology)
- Data privacy breach involving genomic data
- End-of-life decision-making in ICU

Each scenario will explicitly reference key ethical principles (autonomy, beneficence, non-maleficence, justice).

##### Step 2: Prompt Engineering
For each scenario, develop three prompt variations to test:
- Direct clinical framing
- Patient-centered framing
- Policy/regulatory framing (relevant to IRB/DSMB/HEC contexts)

This yields 15 unique prompts.

##### Step 3: AI Model Testing
Selected models:
- GPT-4 Turbo
- Claude 3
- Gemini 1.5

Each prompt will be tested across all three models (45 responses total).

##### Step 4: Evaluation Framework
Develop a standardized evaluation rubric (1-5 scale) assessing:
- Ethical alignment (consistency with ethical principles)
- Transparency and explainability (explicit reasoning provided)
- Clinical relevance and appropriateness
- Empathy and consideration of patient context
- Absence of harmful bias or ethical oversight

Two independent human raters (ethicists with clinical experience) will evaluate responses to determine inter-rater reliability and aggregate scores.

#### Data Analysis
Quantitative:
- Mean scores comparison across models and prompts
- Inter-rater reliability (Cohen’s Kappa)

Qualitative:
- Thematic analysis identifying strengths and weaknesses in ethical reasoning

#### Proposed Concept: Data Safety Monitoring Agent (DSMA)
Define and propose criteria for DSMA:
- Continuously analyzes patient safety data streams from clinical trials and patient care
- Flags ethical concerns, biases, or safety issues proactively
- Provides immediate, transparent reasoning for flagged issues
- Supports human decision-makers by reducing cognitive burden and enhancing decision consistency

#### Potential Impact and Next Steps
- Provide foundational insights into ethical AI tool viability for ethics oversight bodies
- Initial validation of DSMA concept and functionality
- Inform future development and large-scale validation studies

#### Ethical Considerations
- No use of real patient data; all scenarios hypothetical
- Clear delineation that AI supplements, not replaces, human decision-makers
- Transparent reporting on AI limitations and ethical implications

