# Generative AI in Ethics Committees and Research

# Oversight

Generative AI (like large language models such as ChatGPT) is increasingly pervasive in healthcare decision-
making, but its role in ethical decision processes remains nascent. Recently, researchers have begun
exploring how AI could assist **hospital ethics committees** (which guide difficult clinical decisions) and **Data
Safety Monitoring Boards (DSMBs)** (which oversee ethics and safety in clinical trials). Below, we summarize
relevant research and tools that support the case for _prompt-driven ethical testing_ – using AI to generate or
evaluate ethical analyses – and identify high-impact scenarios (e.g. transplantation and oncology) where
such testing could be most beneficial.

## Research on Generative AI in Ethics Committees and DSMBs

Early studies indicate both the potential and current limitations of AI in ethical deliberation. For example, a
2024 pilot study tested ChatGPT’s ability to write clinical ethics consultation notes. **At baseline, ChatGPT
performed poorly** , scoring very low on a standard ethics consult quality scale (ECQAT). However,
when the AI was “trained” by providing examples of past ethics consults, its performance **improved
significantly** – the quality rating rose from about 1 (unacceptable) to ~2.5–3 on a 3-point scale in the best
conditions. This showed that _with some guidance and examples, generative AI can produce an ethical
analysis approaching an acceptable level, though results were variable_. The authors concluded that **human
oversight remains essential** : ChatGPT’s ethical reasoning was not on par with human experts unless
carefully steered, and it struggled as case complexity increased.

Another line of research has evaluated **ChatGPT’s “moral competence.”** A JAMIA Open study applied a
moral reasoning test (based on Kohlberg’s stages) to ChatGPT. It found that GPT-4 demonstrated higher
moral reasoning consistency than GPT-3.5, but overall **ChatGPT only showed “medium” moral
competence** in tackling healthcare ethics dilemmas. In other words, the AI could apply basic ethical
principles but still fell short on more nuanced deliberation. These findings reinforce _why_ prompt-driven
ethical testing is interesting – we need to gauge how well AIs understand ethical nuances and where they
might falter.

Encouragingly, there is emerging evidence that generative AI can assist in **ethical review processes for
research**. A 2024 preprint study in Japan had GPT-4 review clinical trial protocols and consent forms, as an
ethics committee member might. GPT-4 reliably extracted key information (study design, risks, etc.) with
high accuracy (80–100% on certain elements). With customized prompts and fine-tuning, the AI’s
consistency and reproducibility improved further. The authors noted that _with refinement, AI could enhance
the consistency and efficiency of ethics committee evaluations_ ** – helping catch issues and standardize reviews

. This suggests that generative AI might one day support DSMBs or Institutional Review Boards by
quickly summarizing complex documents and flagging ethical considerations, reducing human workload
(though not replacing human judgment).

```
1 2
```
```
3
```
```
2
```
```
4 5
```
```
6 7
```
```
7
```

Notably, direct research on AI **within DSMBs** is sparse so far. We did not find specific studies of an AI
“member” of a DSMB. However, the concept aligns with the above IRB/protocol review work – a generative
AI could, for instance, assist a DSMB by analyzing interim trial data summaries or safety reports and
providing an unbiased summary of risks vs. benefits. In practice, DSMBs require careful analysis of patient
safety data, where AI might help spot patterns. Any such use would need **extensive validation** to ensure
the AI’s suggestions are accurate and ethically sound, given the high stakes (patient safety) involved.

## Current Ethical AI Tools and Approaches

Because the idea of AI aiding ethical decisions is relatively new, researchers have been prototyping various
tools and frameworks:

```
MedEthEx and METHAD: One of the earliest attempts (2006) was MedEthEx , a rule-based “medical
ethics advisor.” More recently, Meier et al. proposed METHAD , a machine-learning system using
fuzzy cognitive maps to model ethical principles. METHAD encodes Beauchamp & Childress’s
principlism (autonomy, beneficence, non-maleficence, justice) in machine-readable form and takes
patient data/preferences as input. It outputs a numerical score (0 to 1) indicating support or
opposition to a treatment, essentially evaluating ethical pros and cons. METHAD is still a proof-of-
concept , but it demonstrates how formalizing ethics for AI could work. Notably, the
developers intentionally left out the principle of justice initially, given how context-dependent justice
can be.
```
```
Patient Preference Predictors: Another conceptual tool is the Patient Preference Predictor (PPP)
```
. This AI would use large datasets of patient choices to predict an incapacitated patient’s likely
wishes for care. For example, if an ICU patient cannot speak for themselves, a PPP algorithm could
consider the patient’s characteristics and compare to similar cases to suggest what treatment
approach aligns with that patient’s probable values. Proponents argue this could **enhance patient
autonomy** by increasing the chances that care aligns with the patient’s true preferences. In
fact, studies show surrogate decision-makers often fail to guess patients’ wishes accurately, so a
data-driven tool might do better. On the other hand, critics worry that reducing such profound
decisions to statistics might **endanger autonomy** – for instance, by relying on population averages
and neglecting the individual’s uniqueness.

```
Other Algorithmic Ethics Aids: Table 3 of a recent systematic review lists several hypothetical AI
tools. For example, a “Do Not Attempt Resuscitation” (DNAR) algorithm is proposed to predict whether a
given patient would want CPR in an emergency, based on their profile and how similar patients
answered advance directive questions. A “Surgery algorithm” has been conjectured to help
surgeons make fairer decisions about high-risk operations – it would objectively assess surgical risk
and maybe counteract human biases (e.g. regarding a patient’s socioeconomic status or race).
An “Autonomy algorithm” was even imagined that trawls a patient’s electronic health records and
social media for clues to their values, to guide care when they can’t express consent. Most of
these are still theoretical (“conjectured” as the review says ). They reflect the growing
interest in ethical decision-support systems.
```
Beyond these specific tools, the **broader debate** on AI in ethics is captured by a 2023 systematic review

. Key anticipated benefits include: improving consistency and transparency in ethical decisions ,
reducing the burden and stress on human decision-makers , and extending ethics support to settings

### •

```
8 9
```
```
9
```
```
10 11
```
```
12
```
### •

```
13
```
```
14 15
```
```
16
```
```
17
```
### •

```
18
```
```
19
```
```
20
21 22
```
```
23
24 25
26
```

with no human ethicists (e.g. smaller hospitals or resource-poor areas). In theory, an “always-available”
AI ethicist could help front-line clinicians faced with a quick dilemma, or assist a busy ethics committee by
drafting an initial analysis. There is even speculation that using such AI tools could provide a form of
_“cognitive moral enhancement”_ for clinicians, prompting them to think more systematically about ethics.

However, current **ethical AI tools are far from perfect**. Common concerns include: lack of true empathy or
understanding of context , risk of algorithmic bias (if the AI’s training data carries societal biases)
, and the challenge of encoding moral reasoning which may not be reducible to equations. Experts
emphasize these AIs should **augment, not replace** human judgment. As one commentary put it, even
well-performing algorithms can be unreliable in individual cases, and “complex deliberations are unlikely to
be successfully reduced” to purely computational terms. Thus, current research is focused on finding
the right balance – leveraging AI’s strengths (data processing, consistency) while ensuring human values
and case-by-case nuance aren’t lost.

## High-Impact Scenarios for AI Ethical Testing

If we want to **test generative AI in ethically charged scenarios** , it makes sense to start with domains that
frequently generate complex ethical dilemmas. Two such domains are **organ transplantation** and
**oncology** , as the user suggested, though there are others (end-of-life care, reproductive decisions, etc.).
These fields involve life-and-death decisions, scarce resources, and often conflicting principles – ideal for
exploring the utility and pitfalls of an AI ethics assistant.

```
Organ Transplantation: Allocating scarce organs (like donor livers, hearts, kidneys) is ethically
challenging by nature. Committees must weigh utility (maximizing lives saved or life-years),
urgency, fairness/justice, and sometimes controversial factors (e.g. substance abuse history or social
worth). AI has been proposed to help make these decisions more consistent and data-driven.
In fact, basic algorithms already play a role – for instance, the MELD score ranks liver transplant
candidates by medical urgency using lab values. More advanced machine-learning models could
integrate many more factors to predict outcomes and suggest an optimal organ allocation. A recent
survey of the public in the UK explored attitudes toward AI in liver transplant allocation.
Interestingly, about 69% of respondents found AI-based allocation acceptable , and 73% said they
wouldn’t be less likely to donate organs if AI was involved. People saw potential advantages: they
viewed AI as more consistent and less biased than human committees. The main concerns were
about “dehumanization” of such a sensitive decision and whether an AI could appreciate the nuances
in patients’ stories. Participants valued accuracy, impartiality, and consistency in decision-
making more than having a human’s empathy per se. This suggests an ethical AI that is
transparent and fair might gain public trust in transplant contexts. For prompt-driven testing, one
could imagine giving an AI a scenario like: “Two patients need a liver – one is younger with better
predicted outcome, the other is sicker (more urgent) – how do we decide who gets it?” and
evaluating if the AI’s reasoning aligns with ethical norms (e.g. does it recognize principles of justice
and utility?). Early research indicates there are good reasons to involve AI in transplant decisions, as
long as implementation is careful. Done right, it could improve objectivity and even legitimacy (e.g.
by reducing perceptions of bias or favoritism). But these scenarios would also test an AI’s ability
to handle morally relevant factors that are hard to quantify (quality of life, societal contributions, etc.,
which humans currently debate intensely).
```
```
27
```
```
28
```
```
29 24
17 30
2
```
```
31
```
### •

```
32
```
```
33 34
```
```
35
```
```
36
37
```
```
37
38
```
```
39
```

```
Oncology (Cancer Care): Oncology often presents ethically fraught choices, such as whether to
pursue aggressive treatment vs. palliative care, how to allocate expensive or experimental therapies,
or when to enroll a patient in a trial. Because AI is rapidly being introduced in oncology (for
diagnostics, prognostication, treatment recommendations), oncologists are starting to confront its
ethical implications. A 2024 survey of over 200 oncologists provides insight into what scenarios
concern cancer doctors. Notably, 81% of oncologists felt patients should give informed
consent before AI is used in making their treatment decisions. This underscores that having
AI involved (say, an AI suggests a chemotherapy plan) is seen as significant enough to disclose to
patients as part of ethical practice. Another scenario: when an AI’s recommendation conflicts with the
physician’s, how should that be handled? In the survey, 37% of oncologists said they would present
both the AI’s and their own recommended regimen to the patient for discussion. This scenario is
ripe for prompt-driven testing: we could ask a generative AI how to reconcile such a disagreement or
explain it to a patient. Liability and bias are also key concerns. The majority of oncologists in the
study thought AI developers should bear responsibility if an AI causes harm (e.g. a wrong
treatment choice), though many also felt responsibility would be shared with physicians or hospitals
```
. Additionally, 76% of oncologists agreed they have a duty to **protect patients from biased
AI** outputs, yet only 28% felt confident they could tell when an AI’s model was biased. This
suggests scenarios about **bias detection** – e.g. if an AI tends to undertreat older patients or
minorities, would a human catch it? Testing AI responses on hypothetical oncology cases (with subtle
biases or requiring explaining reasoning) could be very illuminating.

```
Other Critical Scenarios: Beyond transplant and oncology, end-of-life decisions are a common
domain for hospital ethics committees: for example, deciding on withdrawing life support in an ICU,
or handling a family’s disagreement about a Do-Not-Resuscitate order. These scenarios test
empathy, communication, and respect for patient autonomy. An AI might help by summarizing
similar precedent cases or ethical guidelines (e.g. principles of futility), but we would want to test if
its suggestions are compassionate and legally sound. Likewise, clinical trial dilemmas (the province
of DSMBs) are rich testing ground: Should a trial be stopped early because interim results show either
great benefit (making it unethical to deny the drug to controls) or unexpected harm? An AI could be
prompted with interim data trends and ethical arguments on both sides to see if it advises pausing
or continuing a trial in line with human experts. For instance, in an oncology drug trial showing
moderate efficacy but some severe side effects, does the AI weigh patient safety vs. potential future
benefit appropriately? Because DSMBs operate on data and ethics , an AI that can rapidly analyze
data and recall historical trial ethics decisions could be helpful – but only if it consistently upholds
participant welfare.
```
In summary, _prompt-driven ethical testing_ using generative AI can target these high-impact scenarios to
evaluate AI’s performance as a would-be ethics assistant. **Transplant allocation and oncology care** stand
out because they involve high stakes and nuanced value judgments that AI must navigate. Early research
indicates there is genuine interest and some public/professional openness to AI input in these areas – e.g.
support for AI in organ allocation if it improves fairness , or oncologists willing to integrate AI advice as
long as patient consent and bias mitigation are addressed.

## Conclusion

Bringing a GenAI tool into a hospital ethics committee or a DSMB is an intriguing idea that is starting to
gain scholarly attention. The **motivation** is clear: healthcare providers face ethically complex decisions that

### •

```
40 41
42 43
```
```
44
```
```
45 46
47
```
### •

```
48
42 47
```

cause stress and moral distress , and resources like ethics consultation services are not always available
or consistent. A well-designed AI could **provide real-time ethical guidance** , ensure relevant principles
aren’t overlooked, and make decision processes more transparent. Current background research – from
pilot studies of ChatGPT’s ethical reasoning to conceptual AI ethics advisors – supports the _potential_ benefits
of such a tool while also highlighting the pitfalls. **Existing ethical AI tools** (like METHAD or preference
predictors) show that encoding ethics into algorithms is possible, but careful testing is needed to avoid
undermining human values. Therefore, focusing research on concrete scenarios (transplant boards
deciding who gets an organ, oncology teams debating high-risk treatments, etc.) is a fruitful way to **drive
ethical testing of AI**. These scenarios force the AI to grapple with core bioethical principles (autonomy,
justice, beneficence, non-maleficence) in realistic contexts. By observing where the AI performs well or
poorly, we can gain insight into how it might be safely integrated as a _non-voting “advisor”_ on an ethics
committee or DSMB. Ultimately, the goal is not to hand over moral decisions to machines, but to see if
machines can **help humans make better moral decisions** – more informed, unbiased, and aligned with
patients’ values – in some of the toughest areas of medicine.

**Sources:**

```
Benzinger et al., BMC Medical Ethics (2023) – systematic review of reasons for/against AI in clinical
ethical decision-making.
Jenkins et al., J. of Health Ethics (2024) – pilot study on ChatGPT for ethics consultations.
Rashid et al., JAMIA Open (2024) – study evaluating ChatGPT’s moral competence in healthcare ethics
.
Drezga-Kleiminger et al., BMC Med Ethics (2023) – survey on using AI for liver transplant allocation
.
Dana-Farber Cancer Inst. News (2024) summarizing Hantel et al., JAMA Netw Open – oncologists’
views on AI in treatment decisions.
Fukataki et al., medRxiv preprint (Nov 2024) – using GPT-4 to assist clinical research ethics reviews
.
```
Evaluating ChatGPT's moral competence in health care ... - PubMed
https://pubmed.ncbi.nlm.nih.gov/38983845/

"A Pilot Study in ChatGPT's Ethics Ability" by Daniel Jenkins, Christian Vercler et al.
https://aquila.usm.edu/ojhe/vol20/iss1/5/

Evaluating the Moral Competence of ChatGPT in Medical Ethics
https://healthmanagement.org/c/it/news/evaluating-the-moral-competence-of-chatgpt-in-medical-ethics

Ethical review of clinical research with generative AI: Evaluating ChatGPT’s accuracy and
reproducibility | medRxiv
https://www.medrxiv.org/content/10.1101/2024.11.19.24317555v

Should
Artificial Intelligence be used to support clinical ethical decision-making? A systematic review of reasons -
PMC
https://pmc.ncbi.nlm.nih.gov/articles/PMC10327319/

```
49
```
```
27 30
```
### •

```
14 50
```
-^32
-
    4 5
-^51
    39
-
    42 47
-
    6 7

```
1
```
```
2 3
```
```
4 5
```
```
6 7
```
```
8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 49 50
```

Should AI allocate livers for transplant? Public attitudes and ethical
considerations | BMC Medical Ethics | Full Text
https://bmcmedethics.biomedcentral.com/articles/10.1186/s12910-023-00983-

Study provides a first look at oncologists' views on ethical implications of AI in
cancer care | Dana-Farber Cancer Institute
https://www.dana-farber.org/newsroom/news-releases/2024/study-provides-first-look-oncologists-views-ethical-implications-ai-
cancer-care

```
32 33 34 35 36 37 38 39 48 51
```
```
40 41 42 43 44 45 46 47
```

