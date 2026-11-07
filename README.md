# re.inform

## Project overview
re.inform is a project aimed at countering the spread of far-right extremism, fascism, racism, climate denialism, and other harmful ideologies online. Using a research-backed approach, we develop targeted counter-narratives based on personality categorization to promote critical thinking and reduce vulnerability to extremist content.

## Mission
Our mission is to better understand what leads certain people to fall for extremist or conspiracist content, and provide effective tools for engagement that can be used by various stakeholders working to combat harmful ideologies.

## Core features
- **Personality categorization**: Evidence-based classification of different personality types and their openness to dialogue
- **Counter-narrative tools**:
  - Summary sheets for each personality category
  - Interactive chatbot that provides tailored conversation strategies
- **Community engagement**: Platform for contributors, users, and researchers to share insights and experiences

## Target audience
- Activists
- Educators
- Journalists
- Family/friends of individuals engaged with extremist content
- Political leaders

## Methodology
Our categorization framework is developed through:

- Integration of existing personality categorization methods
- Collaboration with leading social sciences researchers
- Digital "macroscope" mapping of online concepts and communities
- Community feedback and continuous improvement

## EVS-based French personas (data → personas)

This repository contains a reproducible pipeline that builds **10–20 data-driven “personas”** representing the **French population in EVS5 (2018)**, with two deliverables per persona:

- **System of values**: an interpretable summary built from EVS value-attitude axes.
- **Best format to convince**: a persuasion-format recommendation inferred from those axes.

### Inputs

- EVS France cleaned survey extract: [EVS/data/evs_france_cleaned_cleaned_data.csv](EVS/data/evs_france_cleaned_cleaned_data.csv)
  - 1,870 respondents, already filtered to France (`cntry=250`).
  - Missing data uses EVS negative codes: `-5, -4, -3, -2, -1`.
  - Weights: `gwght` is used for population shares.
- Original EVS Stata file (metadata source): [EVS/data/ZA7505_v5-0-0.dta](EVS/data/ZA7505_v5-0-0.dta)
  - Used to extract variable labels (question texts) for interpretability.

### Output

Running the pipeline writes the selected persona set and diagnostics to [EVS/output/](EVS/output/):

- [EVS/output/fr_personas_k10.csv](EVS/output/fr_personas_k10.csv): the personas table (one row per persona).
- [EVS/output/fr_personas_k10.json](EVS/output/fr_personas_k10.json): same content, JSON.
- [EVS/output/fr_personas_k10_scores.csv](EVS/output/fr_personas_k10_scores.csv): k-selection diagnostics for `k=10..20`.
- [EVS/output/fr_personas_k10_var_labels.json](EVS/output/fr_personas_k10_var_labels.json): variable labels from the EVS `.dta`.

The main deliverable columns in the persona CSV are:

- `population_share_w`: weighted share of the persona in the French sample (using `gwght`).
- `value_system`: short value-system descriptor derived from axis z-scores.
- `persuasion_format`: recommended persuasion format (controlled vocabulary).
- `top_axes`: most distinguishing axes for the persona.

### How it works (high level)

The implementation is in [EVS/bin/french_personas.py](EVS/bin/french_personas.py) and follows these steps:

1) **Cleaning and missing values**
  - Treat EVS missing codes `-5..-1` as missing (`NaN`).
  - Drop columns that are entirely missing-coded for the France extract (mostly WVS-only / follow-up metadata).

2) **Build interpretable “value axes”**
  The model does not cluster on 200+ raw survey items. Instead it builds a compact set of interpretable axes, then clusters on those axes.

  Examples of axes currently implemented:
  
  - **Life priorities** (importance in life): family/friends/leisure/politics/work/religion from `A001..A006` (reoriented so higher = more important).
  - **Interpersonal trust**: `A165` (trust vs “can’t be too careful”).
  - **Conformity/order vs autonomy**: derived from child-quality items `A027..A035`.
  - **Economic ideology**: `E035..E037` (equality, ownership, government responsibility).
  - **Democracy attributes**: `E224..E229` (used to build liberal-democracy vs authoritarian-leaning components).
  - **Religiosity**: identity + belief items `F034, F050, F051, F053, F054, F063`.

3) **Preprocessing for clustering**
  - Missing values in axes are imputed with the median (per axis).
  - Axes are standardized (z-scored).

4) **Choose the number of clusters (k) from 10 to 20**
  For each `k` in `[10..20]`, the script computes:
  - **Silhouette score** (separation/compactness)
  - **Stability** (mean Adjusted Rand Index across multiple random initializations)
  - **Minimum weighted cluster share** (to avoid tiny clusters)

  The pipeline selects the best `k` by silhouette, preferring solutions where all clusters exceed a minimum weighted share (configurable).

5) **Turn clusters into personas**
  For each cluster:
  - Compute **axis z-scores** relative to the French sample.
  - Summarize a **value system** (e.g., “secular, liberal-democratic, egalitarian, autonomy”) based on strong z-score signals.
  - Assign a **persuasion format** from a rule-based mapping on the same z-scores.

### Persuasion formats

EVS does not directly ask “do you prefer calm discussion vs constraint”, so the pipeline infers persuasion formats from value-attitude signals.

Current controlled vocabulary (see [EVS/bin/french_personas.py](EVS/bin/french_personas.py)):

- `calm_discussion_evidence`
- `calm_discussion`
- `example_social_proof`
- `moral_care_fairness`
- `moral_tradition_order`
- `authority_rules`
- `constraint_enforcement`

These are intentionally simple, transparent heuristics meant to produce usable “first-pass” persona guidance. They can be refined once you decide which behaviors you want to optimize for (e.g., climate persuasion vs anti-conspiracy de-radicalization).

### Reproducing the personas

This project assumes a conda environment. On your machine this was run with:

- `conda activate /Users/lucie/miniforge3/envs/re_inform`

Dependencies used by the pipeline:

- `numpy`, `pandas`, `pyreadstat`, `scikit-learn`

Run:

- `cd EVS`
- `python bin/french_personas.py --k-min 10 --k-max 20 --min-share 0.03`

### Notes and limitations

- Cluster quality depends heavily on the choice of axes. The current axes prioritize interpretability and stable coverage in the France extract.
- Silhouette scores are expected to be modest in survey-attitude clustering (overlapping human populations); stability and interpretability matter more than a single metric.
- The persona “value_system” and “persuasion_format” are derived from z-scores and rules; they should be treated as *actionable summaries*, not clinical/psychometric diagnoses.

## Technologies
- **Frontend**: HTML, CSS, JavaScript (with potential React integration)
- **Backend**: Under evaluation (Node.js/Express or Python with Flask/Django)
- **Chatbot**: Large Language Model (LLM) implementation

## Current status
The project is in prototype development stage, with focus on building a functional "vitrine" website before scaling.

## Minimum viable product
Our initial release will include:

- Project manifesto outlining our values and approach
- Description of contribution opportunities
- Reference library of related resources
- Community access ([Discord](https://discord.gg/fu5aveEJ))
- Feedback collection mechanism

## Contributing/Contact
We welcome contributions from researchers, developers, content creators, and community organizers. Join our [Discord](https://discord.gg/fu5aveEJ) community to learn more about how you can get involved.
User feedback is essential to our project's evolution. Share your thoughts and experiences through:

- Our [Discord](https://discord.gg/fu5aveEJ) community
- Website feedback form
- Direct engagement with project contributors

## License
[To be determined]
