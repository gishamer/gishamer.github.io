---
title: "Hyperparameter search with Optuna"
layout: post
---
In this post, I want to give you a short introduction to hyperparameter search with Optuna. I came across Optuna a few months ago when I was looking for a framework that allowed me to optimize hyperparameters across ml-frameworks that enabled me to record the results in a nice and organized way.

Some of the listings in this tutorial are a bit shortened, and some functions are left out to keep the text concise and clean. You can find the complete code example on this [github-repository](https://github.com/gishamer/optuna-example).

# What is Optuna?

When you train a machine learning model, some parameters are optimized with respect to an objective function. And then there are hyperparameters, things like regularization parameters, activation functions, etc. In a broader sense, you could also view preprocessing steps as hyperparameters. For instance, in a setting where you want to use texts as features, you might decide to either use tf-idf document vectors or just simple word count vectors. You could then decide that you wanted to search for the optimal vectorization approach for the problem.  

Optuna is a framework for automatic hyperparameter optimization. To find the best values for the parameters passed to Optuna for optimization, it uses the *Tree-structured Parzen Estimator* algorithm by default, about which you can read more [here](https://proceedings.neurips.cc/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf). Still, it allows you to choose many different algorithms. Optuna also enables you to use pruning algorithms that automatically stop runs that are not yielding promising results; think early-stopping. I will not go into depth on the topic of pruning, but if you want to learn more about it, you can find additional information [here](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html).  

Now what I consider one of the most interesting features of Optuna is its pythonic search space, meaning that you can define your strategy for choosing values like regular python code. You might think that this could be a disadvantage, because why not do it declaratively? But in my experience, this approach is very convenient since it still allows you to define all your parameter ranges in a configuration file while giving you the freedom to extend Optunas default behavior with some fancy logic, implemented by: you!

# Using Optuna for hyperparameter search
You can install Optuna with either Pip or Conda. However, on my m1 MacBook, Pip did not work. Therefore I had to use Conda, with the following command:
```
conda install -c conda-forge Optuna
```
If you want to try Optuna with the repository I linked in the introduction, feel free to use the `environment.yml` file to set up a conda environment containing all the necessary dependencies.

## Study object

To start a hyperparameter search, you must first create a study object. A study, in turn, uses a storage backend to persist its results.
There are multiple options to choose from, such as PostgreSQL, MySQL, or SQLite, which I will use in this article:

```python
study_name = 'scikit-learn-study'                                                             
storage_name = f'sqlite:///{study_name}.db'                                                   
study = optuna.create_study(study_name=study_name, storage=storage_name, direction='maximize')
```
In this case, Optuna will create a file named `scikit-learn-study.db`, a SQLite database you can use for querying directly, but the study object is very convenient to retrieve the results instead.  

## Optimize method

After creating a study, you start optimizing your classifier by calling the study's optimized method. The optimize method's first argument is a function that accepts a single argument, a so-called trial object.

In the context of Optuna, the function argument is called the objective function. It is a regular python function where you set up your classifier and everything needed for your experiment.  

Now let's say you defined your parameterized objective function with more than one parameter than a very convenient
way to use it with optimize (which accepts only one parameter) is to use a lambda function.  

The second parameter of the optimize function is called `n_trials`, where you specify the maximum number of runs
Optuna should perform to optimize the hyperparameters for you.  

```python
study.optimize(lambda trial: svm_objective( 
    df,                                     
    trial                                   
), n_trials=50)                             
```

## Objective function

Below you can see the objective function. Firstly you assemble the pipeline by passing the `trial` object `create_pipeline` function, which returns a Scikit-learn pipeline object.

The next step is to create a stratified k-fold iterator, which allows evaluating a classifier on multiple folds to increase the reliability of the obtained results.  

To obtain the actual scores, you can employ Scikit-learn's `cross_val_score`, which returns a list of scores you can
use to get the mean score, which Optuna uses for optimization.

```python
def svm_objective(df: pd.DataFrame, trial: optuna.Trial) -> float:   
    classifier_obj = create_pipeline(trial)                          
                                                                        
    skf = StratifiedKFold(n_splits=5)                                
                                                                        
    scores = cross_val_score(                                        
        classifier_obj, df['text'], df['label'], cv=skf, scoring='f1'
    )                                                                
                                                                        
    return scores.mean()                                             
```

## Trials and the suggest methods

The following listing shows the part of the `create_pipeline` function responsible for setting the SVM classifiers hyperparameters. Here, you can use the `trial` object to have Optuna suggest values at each run. For example, looking at the `C` and the `kernel` parameter, you will notice that the first one uses `trial.suggest_float`, and the second one uses `trial.suggest_categorical`.

Categorical can be used for categorical values in the classical sense, such as `['fish', 'insect', 'mammal']`, but also to limit the search space to a set of fixed values when using integers or floats, such as `[64, 128, 256]`.

When using the `suggest` methods for float and int, you have to specify a range by setting the `low` and `high` parameters from which Optuna will then choose a value at each trial.

The `suggest` functions always have a name as their first parameter, which is very important. Optuna uses this string as an identifier, and additionally, it is the name used in the reports that we'll come to later.

```python
pipeline.extend([                                                        
    ('classifier', SVC(                                                  
        max_iter=10_000,                                                 
        C=trial.suggest_float('svc_c', low=2e-5, high=2e15),             
        kernel=trial.suggest_categorical('kernel', ['rbf', 'linear']),   
        gamma=trial.suggest_float('svc_gamma', low=2e-15, high=2e-3)))
])
```

A neat trick to choose between different branches is to use boolean values in `suggest_categorial`. For example, let's say you wanted Optuna to determine whether to choose preprocessing `function_a` or `function_b`. You then could easily do it like this:

```python
if trial.suggest_categorical('preprocessing_function', [True, False]):
    preprocessed_data = function_a(raw_data)
else:
    preprocessed_data = function_b(raw_data)
```

## Evaluation

After Optuna completes a run, you need a way to evaluate the results. In Optuna, you would use the study object to do so, which has the great advantage of not having to compute all the metrics you're interested
in immediately after your experiments since all of the results are persisted in your storage backend of choice. To load the results of your experiments back into memory, you can tell Optuna to load your study:

```python
study = optuna.load_study(study_name=study_name, storage=storage_name)
``` 

There are some methods on the study object that help you quickly gather the essential information of your experiments. A very convenient one is `study.best_trial.params()`, which returns the parameters and the corresponding values of the best run with respect to the objective used. Another one is `study.plot_param_importances()`, which creates a Plotly plot depicting which parameters had the most significant influence on your results.

```python
def evaluate_results(study_name: str):                                    
    study = optuna.load_study(study_name=study_name, storage=storage_name)
                                                                          
    best_values = study.best_trial.params                                 
    param_importances = optuna.visualization.plot_param_importances(study)
                                                                          
    print(best_values)                                                    
    param_importances.show()                                              
```

The output of `study.best_trial.params` is a simple dict that is easy to serialize as JSON for further processing. Note how I mentioned earlier how important it is to give your parameters descriptive names? They are the keys in the output of `study.best_tria.params` and the labels in the plots you create.

```json
{
    "kernel": "linear",
    "ngram_range": "(1,1)",
    "stop_words": "None",
    "sublinear_tf": "True",
    "svc_c": 1989652561500881.0,
    "svc_gamma": 0.0006103416712680179,
    "vectorizer": "tfidf"
}
```

Below you see the parameter importance plot. There are many other plots Optuna can create for you, such as *intermediate values*, *parallel coordinates*, and *contour plots* among others, about which you can find more information [here](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/005_visualization.html). I haven't been able to test it myself, but I saw there is even a real-time [dashboard](https://github.com/optuna/optuna-dashboard) that lets you inspect and view the plots in a web interface. 

![param importances](/assets/images/hyperparameter_search_param_import.png)