# CASE_exemplar_bandits

Code base for the novel gap-index based exemplar selection approach

Code is adopted from the repository https://github.com/clreda/linear-top-m, Code associated with AISTATS 2021 paper "Top-m identification for linear bandits". We extended their setup to implement our algorithm CASE with related changes for synthetic experiments and implementatiosn for ICl exempalr selection for large langauge models. We thank the authors for making their code open source.
<p>
<img src="CASE_overview.png"> </img>
</p>

## TL;DR

+ Install necessary packages:

```bash
python -m pip install --force-reinstall pip==9.0.1
pip install -r requirements.txt
```

## Command options

- default argument values are written in JSON in file *args.json*. Necessary arguments are:

	+ **small\_N** for **data**=linear or logistic
	+ **small\_K** for **data**=classic or linear or logistic
	+ **vr** for **data**=linear or logistic
	+ **omega** for **data**=classic
	+ **T\_init** for **bandit**=TrueUniform
	+ **bandit**
	+ **data**
	+ **m**
	+ **beta**

# To Run CASE and replicate main result sin paper use 

python main.py --small_K 20 --data linear --omega 0.1 --sigma 0.5 --bandit CASE --is_greedy 1 --m 3 --n_simu 10 --epsilon 0. --delta 0.05 --small_N 3 --beta Heuristic --mode recommendation

# To Run LinGIFA and replicate main result sin paper use 
python main.py --small_K 20 --data linear --omega 0.1 --sigma 0.5 --bandit LinGIFA --is_greedy 1 --m 3 --n_simu 10 --epsilon 0. --delta 0.05 --small_N 3 --beta Heuristic --mode recommendation


- Adding "--verbose 1" allows to display comments and description of selected arms and obtained rewards.

## Modifying the code

### Add a new bandit algorithm

- Check file *bandits.py*. You need to create a sub-instance of **ExploreMBandit** and add it to the *bandit\_factory*. Then you will be able to invoke it from the terminal with option "--bandit".

### Add a new type of data

- Check file *data.py*. You need to modify function *create\_scores\_features*. Then you will be able to invoke it from the terminal with option "--data".

### Add a new threshold function

- Check file *betas.py*. You need to create a function similar to the **AlphaDependentBeta**, **LUCB1Beta**, ... and add it to the *beta\_factory*. Then you will be able to invoke it from the terminal with option "--beta".

### Add a new problem type

- Check file *problems.py*. You need to create a sub-instance of **GenericProblem** and add it to the *problem\_factory*. Then you will be able to invoke it from the terminal with option "--problem".


# Real World Experimets for LLM exemplar selection

The exemplar selection experiments can be found at Code/LLM_experiments

For instance to select exemplars for tabmwp run
```
python Code/LLM_experiments/CASE_tabmwp_selection.py
```
```
Alternatively you can directly use the selected exemplars released by us in ICLR_CASE/CASE_exemplar_bandit/Code/LLM_experiments/output using both Llama2, mistral and run inference using gpt-3.5 or other models on the test set of benchmarks to reproduce the main results in our paper. We call this the exemplar reuse setting as stated in the paper
```

All data for the 5 benchmarks can be found at Code/LLM_exepriments/data
