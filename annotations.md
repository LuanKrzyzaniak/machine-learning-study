## Machine Learning Annotations
Luan Alecxander Krzyzaniak

Applications of machine learning include problems where the decision logic bypass human capability or demand too much work to manually build and mantain.
Some of these problems include face, handwriting or even medical conditions recognition. In these cases, it's difficult for the human mind to identify clear rules or patterns and manually translate them to a algorithm, and that's where machine learning comes in hand. <br>
Of course, it doensn't mean that we can't apply machine learning to more basic problems.

#### ✏️ General concepts

**General roadmap for model building**
|ID|Passo|Descrição|
---|---|---|
|1|Dataset|Identify data, study the set, decide on the project bases (oriented/unoriented)|
|2|Pre-processing|Data treatment: remove incomplete or incorrect data, colect statistics, plotting, building relationship tables, normalize data (apply scale)|
|3|Split the data|Prepare training and testing sets|
|4|Choose the algorithm|Define the algorithm (regression or classification) and its hiperparameters|
|5|Test the model|Apply the test set and collect avaliation metrics|
|6|Fix the model|In case of bad performance, we can revisit the hiperparameters, change algorithms or revise our data, identifying bad parameters and experimenting with new ones|

**Training and test sets:** to avoid creating a bias in the pattern recognition, it's important to have a separate set of data for tests. If we test our machines with the same (or part of the) training data, we risk it struggling with different patterns of data in real applications

**Overfitting:** when your model cannot properly generalize, and become too accustomed to the training dataset. In this case, the accuracy ratings become imprecise, as the model fails when presented with new data.This can happen when your dataset is too small, has a lot of noisy data (irrelevant data), trains for too long with the same dataset, or the model is so complex that the model learn the noisy data. <br>
**Underfitting:** Every model that cannot properly identify what it's supposed to is classified as underfitting.

**Oriented training:** we provide data and expected results to the machine. In this type of training, data quality becomes of upmost importance, given that the performance of the pattern recognition depends on it. <br>
**Unoriented training:** it's used when we cannot determine an expected output, either because we don't know its contents or size. An example include grouping social media users by their media consumption: in this case, we don't know what media categories we're dealing with, so we cannot provide a precise output.

**Regression algorithms:** deals with continuous results, like predicting temperatures or guessing a person's height. <br>
**Classification algorithms:** deals with a specified classification, like classifying a temperature as 'hot' or 'cold' or  even a person as 'short' or 'tall'.

#### 📘 Introduction for machine learning with python

**Why Python?** Python is a general-use language that's easy to use and include a variety of useful features and libraries. It has support for object-oriented programming, web applications and GUI design, as well as various libraries for data treatment that allow us to model over any type of data, be it image, text, or value. Also, Python supports some script languages, which allow us to quickly interact with our code via terminal or other tools. <br>

**scikit-learn** is the most used Python library for machine learning. User guide: <https://scikit-learn.org/stable/user_guide.html> <br>
Install it via **ANACONDA**, a prepackage that includes numPy, SciPy, matplotlib, pandas, IPython, Jupyter Notebook, and scikit-learn, or via **PIP** $ pip install numpy scipy matplotlib ipython scikit-learn pandas

#### ❔ Questions

*EVALUATION STEP: how does this apply to unoriented training? Do i need to evalueate it manually, since i don't have an example output?*

#### 🗒️ References

C. Müller, Andreas; Guido, Sarah. Introduction to Machine Learning with Python: a guide for data scientists. First edition. USA: O'Reilly, 2017.

FreeCodeCamp. Machine Learning Principles Explained. Febuary 1st, 2020. Available at: <https://www.freecodecamp.org/news/machine-learning-principles-explained/#:~:text=The%20three%20components%20that%20make,to%20look%20at%20your%20data.>. Access in 22nd of September, 2023.

AWS Amazon. What is ovefitting? Available at: <https://aws.amazon.com/pt/what-is/overfitting/#:~:text=Overfitting%20is%20an%20undesirable%20machine,on%20a%20known%20data%20set.>. Access in 22nd of September, 2023.
