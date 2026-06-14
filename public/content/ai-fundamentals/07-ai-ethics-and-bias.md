---
title: "AI Ethics and Bias"
difficulty: beginner
topic: ai-fundamentals
order: 7
estimatedTime: "15 minutes"
summary: "Explores types of bias in AI systems, real-world examples of algorithmic harm, fairness definitions and trade-offs, explainability techniques, and the EU AI Act."
---

# AI Ethics and Bias

## Overview

AI systems are increasingly making decisions that affect people's lives — who gets a loan, who gets hired, who goes to prison. When these systems inherit or amplify human biases, the consequences can be severe and systemic. Understanding AI ethics isn't optional — it's essential for anyone building or deploying AI.

### Types of Bias in AI

Bias can enter AI systems at every stage:

**Data Bias** — The most common source. If training data doesn't represent the real world fairly, the model won't either.

- **Historical bias**: Data reflects past discrimination. If historical hiring data shows men were hired more for engineering roles, the model learns to prefer men.
- **Representation bias**: Underrepresentation of certain groups. ImageNet was initially dominated by images from the US and Europe.
- **Measurement bias**: The way data is collected introduces systematic errors. Arrest data doesn't measure crime — it measures policing patterns.

**Algorithmic Bias** — The model architecture or optimization objective itself can amplify biases.

- **Feedback loops**: A model predicts more crime in a neighborhood → more police are sent → more arrests are made → model "confirms" its prediction.
- **Proxy variables**: Even without explicit demographic features, models can use proxies (zip code → race, name → gender).

**Deployment Bias** — A model works well in the lab but fails for certain groups in practice.

### Real-World Examples

**COMPAS Recidivism Algorithm**: Used by US courts to predict reoffending risk. A ProPublica investigation found it was twice as likely to falsely flag Black defendants as high risk compared to white defendants, while being twice as likely to incorrectly label white defendants as low risk.

**Amazon Hiring Tool (2018)**: Amazon built a resume screening AI trained on 10 years of hiring data. It learned to penalize resumes containing the word "women's" (as in "women's chess club") and downgraded graduates of all-women's colleges. Amazon scrapped the tool.

**Healthcare Algorithm (2019)**: A widely used algorithm in US hospitals used healthcare spending as a proxy for health needs. Because Black patients historically had less access to healthcare (and thus lower spending), the algorithm systematically underestimated their needs, affecting millions.

### Fairness Definitions

There is no single definition of fairness — and some definitions are mathematically incompatible:

- **Demographic Parity**: The model's positive prediction rate should be equal across groups. $P(\hat{y}=1|A=a) = P(\hat{y}=1|A=b)$
- **Equalized Odds**: True positive and false positive rates should be equal across groups.
- **Individual Fairness**: Similar individuals should receive similar predictions.
- **Counterfactual Fairness**: The prediction should be the same if the individual's protected attribute were different.

The **impossibility theorem** (Chouldechova, 2017) proves that except in trivial cases, you cannot simultaneously satisfy demographic parity, equalized odds, and predictive parity. Trade-offs are inevitable.

### Explainability (XAI)

If a model denies your loan, you deserve to know why. **Explainable AI** aims to make model decisions interpretable:

- **LIME**: Approximates any model locally with an interpretable one
- **SHAP**: Uses game theory (Shapley values) to assign feature importance
- **Attention visualization**: In Transformers, examine attention weights (though this is debated as a true explanation)
- **Counterfactual explanations**: "Your loan would have been approved if your income were $5K higher"

### EU AI Act

The EU AI Act (2024) is the world's first comprehensive AI regulation. It classifies AI systems by risk:

1. **Unacceptable risk** (banned): Social scoring, real-time biometric surveillance
2. **High risk** (strict requirements): Hiring tools, credit scoring, law enforcement
3. **Limited risk** (transparency obligations): Chatbots must disclose they're AI
4. **Minimal risk** (no restrictions): Spam filters, video games

High-risk systems must maintain documentation, conduct impact assessments, ensure human oversight, and allow affected individuals to contest decisions.

## Key Concepts

- **Data Bias**: Training data that doesn't fairly represent the population
- **Feedback Loop**: Model predictions influencing future data, reinforcing bias
- **Proxy Variable**: A feature that indirectly encodes protected attributes
- **Fairness Trade-offs**: Different fairness definitions are mathematically incompatible
- **Explainability (XAI)**: Making AI decisions interpretable and contestable

## Exercises

1. **Identify the bias**: A facial recognition system has 99% accuracy on light-skinned faces but 65% on dark-skinned faces. What type(s) of bias are at play?
2. **Fairness debate**: A college admissions AI has equal accuracy across racial groups but admits 30% of Group A applicants vs 15% of Group B. Is this fair? Under which definition?
3. **Design task**: You're building a hiring AI. List 5 specific steps you would take to mitigate bias.
4. **Research**: Find one AI ethics guideline document from a major tech company. What principles does it emphasize?

## Further Reading

- Buolamwini, J. & Gebru, T. (2018). "Gender Shades: Intersectional Accuracy Disparities in Commercial Gender Classification"
- O'Neil, C. *Weapons of Math Destruction* (book)
- Chouldechova, A. (2017). "Fair prediction with disparate impact"
- EU AI Act: https://artificialintelligenceact.eu/
