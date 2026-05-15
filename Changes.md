\# Changes.md
\## Project title
Early-Exit Deep Neural Networks for Distorted Images

\## improvment 1  members
\- Mehdi Samanipour
\- Peter Normann Diekema

\## Brief description of changes made to the original code
In this project, we extended the original early-exit image classification code with new experiments and improvements focused on branch placement, threshold tuning, and robustness against distorted images.
The original code used a fixed early-exit configuration. We modified the implementation so that the early-exit branch positions can be changed and tested systematically. This allowed us to run a grid search over several branch placements in MobileNetV2.

\## Main changes
\### 1. Configurable early-exit branch placement
We changed the early-exit model so that the branch positions are no longer fixed. Instead, the user can define different exit positions in the MobileNetV2 feature extractor.
This was added because the position of an early exit strongly affects the trade-off between classification accuracy and computational cost. Early exits save more computation, but they may use weaker features. Later exits use stronger semantic features, but they save less computation.

\### 2. Grid search over branch positions
We added a grid search experiment to compare several different early-exit placements.

The tested branch placements were:

\- `(2, 5, 10)`

\- `(3, 6, 13)`

\- `(4, 8, 14)`

\- `(5, 9, 15)`

\- `(6, 10, 16)`

\- `(7, 12, 17)`


The best placement found in our experiments was `(7, 12, 17)`. This placement gave the best validation accuracy-cost trade-off among the tested configurations.

\### 3. Confidence threshold sweep


We added threshold sweeping for the early-exit policy. The model checks the confidence score at each exit. If the confidence is high enough, the image exits early. Otherwise, it continues to a deeper branch or the final classifier.
We tested several threshold combinations to study the trade-off between accuracy and computational cost.

Example threshold settings:

\- `0.99 / 0.97 / 0.95`

\- `0.90 / 0.85 / 0.80`

\- `0.80 / 0.75 / 0.70`

\- `0.70 / 0.60 / 0.50`

This helped us understand how aggressive or conservative early exiting affects performance.

\### 4. Cost-ratio measurement

We added cost-ratio calculation to estimate how much computation is used compared to running the full model.

The cost ratio is based on which exit is used. If many samples exit early, the average cost ratio becomes lower. This allowed us to compare accuracy and computational saving together.

\### 5. Exit distribution analysis
We added analysis of how many samples exit at each branch.

This shows whether the early-exit system actually uses the intermediate branches or whether most samples still continue to the final classifier.

\### 6. Distortion experiments

We added experiments for distorted images, especially:



\- Gaussian blur

\- Gaussian noise
These experiments were used to evaluate how robust the early-exit model is when the input images are degraded.

\### 7. Visualization and output plots
We added code to save plots and figures for the report, including:

\- Gaussian blur results

\- Gaussian noise results

\- Best exit distribution

\- Loss curves and validation accuracy

\- Cost ratio

\- Accuracy-cost trade-off

\- Mixed-distortion accuracy

\- Edge-inference probability

These figures were used in the final report.

\## Files modified or added

The main code files are:

\- `project15\_gridsearch\_early\_exit.py`

\- `requirements.txt`


The main output folders generated during experiments were:

\- `outputs`

\- `outputs\_gridsearch`

\- `outputs\_best\_7\_12\_17`

\- `outputs\_best\_7\_12\_17\_blur`

\- `outputs\_best\_7\_12\_17\_noise`

The output folders contain experiment results and figures. Trained model files are not required for submission.

\## How to run

Install dependencies:


```bash

pip install -r requirements.txt
