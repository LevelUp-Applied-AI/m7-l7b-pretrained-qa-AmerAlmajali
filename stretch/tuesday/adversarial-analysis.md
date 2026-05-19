# Adversarial QA Probe - Analysis Memo

> Replace each placeholder section. Memo target: ~1 page. The TA rubric rewards specificity grounded in your data.

## 1. Hypothesis

State your targeted failure mode operationally:
- **Input pattern:**   The context contains multiple named entities of the same semantic type (person, location, or organization), especially when distractor entities appear close to important relation phrases or inside contrastive sentence structures.

- **Output pattern:**   The model predicts a distractor entity instead of the correct referent, often selecting the entity most strongly associated with nearby keywords rather than the entity that actually satisfies the question relationship.

- **Why you hypothesize this:**   The `distilbert-base-cased-distilled-squad` model appears to rely heavily on shallow lexical matching and local attention patterns learned during SQuAD-style extractive QA training. When several entities of the same type appear near relevant keywords, the QA head tends to overweight proximity and semantic association signals instead of fully resolving grammatical or relational structure. This behavior becomes more visible in adversarial contexts that deliberately place competing entities near the answer span.


## 2. Set Design

- Total examples: 33
- Tags used: 
  - `person-same-sentence` (8)  
  - `person-cross-sentence` (4)  
  - `person-paraphrased` (3)  
  - `location-same-sentence` (4)  
  - `location-cross-sentence` (2)  
  - `location-paraphrased` (2)  
  - `org-same-sentence` (4)  
  - `org-cross-sentence` (2)  
  - `control-person` (2)  
  - `control-location` (1)  
  - `control-org` (1)
- Why these tags: 
  - `same-sentence` tags test whether nearby competing entities in the same sentence confuse the model.  
  - `cross-sentence` tags test whether the model can maintain correct entity relationships across multiple sentences.  
  - `paraphrased` tags test whether the model depends on shallow lexical overlap between question and context.  
  - `control` tags isolate the effect of distractor entities by providing clean non-adversarial examples of the same QA task type.

- Control examples: 
  Four control examples were included. These examples contain straightforward extractive answers without competing distractor entities. The controls confirm that the model can correctly answer standard QA prompts and that the observed failures are specifically tied to distractor-entity patterns rather than general task difficulty.


## 3. Results

- Aggregate EM: 0.8485  ; Aggregate F1: 0.8658  
- Lab 7B baseline (from your `qa_metrics.json`): EM: 0.34; F1: 0.46
- Per-pattern_tag breakdown:

| Pattern | n | EM | F1 | vs. baseline |
|---|---|---|---|---|
| person-same-sentence | 8 | 1.00 | 1.00 | +0.66 EM |
| person-cross-sentence | 4 | 1.00 | 1.00 | +0.66 EM |
| person-paraphrased | 3 | 1.00 | 1.00 | +0.66 EM |
| location-same-sentence | 4 | 0.75 | 0.75 | +0.41 EM |
| location-cross-sentence | 2 | 1.00 | 1.00 | +0.66 EM |
| location-paraphrased | 2 | 1.00 | 1.00 | +0.66 EM |
| org-same-sentence | 4 | 0.50 | 0.50 | +0.16 EM |
| org-cross-sentence | 2 | 1.00 | 1.00 | +0.66 EM |
| control-person | 2 | 1.00 | 1.00 | +0.66 EM |
| control-location | 1 | 1.00 | 1.00 | +0.66 EM |
| control-org | 1 | 1.00 | 1.00 | +0.66 EM |

Cite at least 3 specific (qid, question, gold, predicted) tuples that illustrate the patterns:


- **(ADV_L03)** *Where is the Los Angeles Philharmonic based?*  
  -> gold: `Los Angeles`, predicted: `Walt Disney Concert Hall`.  
  The model selected a nearby landmark entity instead of the broader city-level location requested by the question, suggesting confusion between related location spans.

- **(ADV_O01)** *Which organization found Dr. Murray guilty?*  
  -> gold: `Los Angeles Superior Court`, predicted: `American Medical Association`.  
  The context contained multiple organization entities, and the model incorrectly selected the distractor organization rather than the organization directly associated with the conviction event.

- **(ADV_O04)** *Which organization distributes the distilbert QA model?*  
 -> gold: `Hugging Face`, predicted: `Google`.  
  The model over-associated the phrase fine-tuned by Google with the question and ignored the later clause identifying the actual distributor of the model.


## 4. Production Defense

The most appropriate production defense would be retraining or further fine-tuning the QA model using adversarial distractor-based examples similar to those constructed in this probe. The per-pattern results show that the largest weaknesses appeared in organization and location distractor scenarios, where nearby competing entities caused incorrect span selection despite otherwise clean contexts. Adding adversarial training examples that emphasize relational reasoning and entity disambiguation would likely improve robustness without requiring major architectural changes to the QA pipeline.
