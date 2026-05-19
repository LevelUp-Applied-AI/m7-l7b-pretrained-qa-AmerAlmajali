# QA Evaluation Report — Module 7 Week B

## Dataset Description
~1,000 extractive QA examples loaded from `tech_news_qa.csv` (CNN tech and entertainment news slice), curated so that every gold answer is a literal substring of its context. Each example contains a `qid`, `question`, `context`, and `gold_answer`.

## Model
`distilbert-base-cased-distilled-squad`  
https://huggingface.co/distilbert-base-cased-distilled-squad

## Aggregate Metrics

| Metric | Score |
|--------|-------|
| Exact Match (EM) | 0.34 |
| Token-F1 | 0.46 |

The 12-point gap between EM and F1 indicates the model frequently finds the right region of the context but gets span boundaries wrong — predicting slightly longer or shorter spans than the gold answer. When it is wrong, it is not completely wrong; it tends to land nearby. This pattern is typical of a SQuAD v1.1-trained model on informal news text where gold spans were annotated with varying levels of specificity.

## Failure-Mode Taxonomy

**1. Distractor Entity Selection**  
The model selects a named entity from the context that matches the question type (person, place, title) but is the wrong referent — a different entity appears nearby and outscores the correct one.

> `NEWS_0221_Q5` — *Who is fast-tracking to get married?*  
> Gold: `Jade Goody` | Predicted: `Tweed`  
> The model latches onto a co-occurring proper noun rather than the grammatical subject.

**2. Action-Span Substitution**  
When the gold answer is a specific term or label (medical, descriptive, categorical), the model returns the clause describing what the subject *did* instead — substituting an event span for a noun answer.

> `NEWS_0118_Q5` — *What does John Mayer have?*  
> Gold: `granuloma` | Predicted: `bowed out of a series of concerts`  
> The model picks the most salient clause about Mayer rather than the clinical noun that answers the question.

**3. Summary-Level vs. Span-Level Answer**  
For questions whose gold answer is a long descriptive span, the model collapses to a short title or label — returning the name of a thing rather than the requested description of it.

> `NEWS_0593_Q2` — *What is the film about?*  
> Gold: `Michael Oher, who went from being a homeless inner-city high school student whose father was dead and whose mother was a crack addict to a star lineman at the University of Mississippi`  
> Predicted: `The Blind Side`  
> The model returns the film title (also present in the context) instead of the requested descriptive span.

## Domain Judgment

I would **not** ship this model for **medical literature QA** (e.g., answering clinician questions from research abstracts). The primary blocker is faithfulness calibration: the model always returns a span even when the context does not contain a reliable answer, and the `score` field is not a calibrated probability — a confidently wrong span over a drug dosage or diagnosis criterion carries unacceptable patient-safety risk. A SQuAD v2.0-trained model with explicit no-answer support (e.g., `deepset/roberta-base-squad2`) and a human review layer would be the minimum bar before deployment in that domain.
