## Machine Learning Annotations
Luan Alecxander Krzyzaniak - 2023

Applications of machine learning include problems where the decision logic bypass human capability or demand too much work to manually build and mantain.
Some of these problems include face, handwriting or even tumors recognition. In these cases, it's difficult for the human mind to identify clear rules to the patterns and manually translate them to a algorithm; and that's where machine learning comes in hand. <br>
Of course, it doensn't mean that we can't apply machine learning to more basic problems.

A general structure for a machine learning algorithm: <br>
**Representation** - this step is where you build your T dataset. Here, you'll decide how to see and organize your data. You can organize it individualy, or in tables, for example. <br>
**Evaluation** - Using an evaluation function, this step consists of ranking the effetiveness of an algorithm. It's used in oriented trainings, so you know how well your learner treated the data and how you need to adjust it. Accuracy and squared error are some metrics.<br>
**Optimization** - That's where you find the best learner and research/apply modifications to improve it. Greedy search and gradient descent are some techniques.

**Training and test sets:** to avoid creating a bias in the pattern recognition, it's important to have a separate set of data for tests. If we test our machines with the same (or part of the) training data, we risk it struggling with different patterns of data in real applications.

**Overfitting:** when your model cannot properly generalize, and become too accustomed to the training dataset. In this case, the accuracy ratings become imprecise, as the model fails when presented with new data.<br>
It can happen when your dataset is too small, has a lot of noisy data (irrelevant data), trains for too long with the same dataset, or the model is so complex that the model learn the noisy data. <br>
**Underfitting:** Every model that cannot properly identify what it's supposed to is classified as underfitting.

**Oriented training:** we provide data and expected results to the machine. In this type of training, data quality becomes of upmost importance, given that the performance of the pattern recognition depends on it.

### Questions

*EVALUATION STEP: how does this apply to unoriented training?*

### References

C. Müller, Andreas; Guido, Sarah. Introduction to Machine Learning with Python: a guide for data scientists. First edition. USA: O'Reilly, 2017.

FreeCodeCamp. Machine Learning Principles Explained. Febuary 1st, 2020. Available at: <https://www.freecodecamp.org/news/machine-learning-principles-explained/#:~:text=The%20three%20components%20that%20make,to%20look%20at%20your%20data.>. Access in 22nd of September, 2023.

AWS Amazon. What is ovefitting? Available at: <https://aws.amazon.com/pt/what-is/overfitting/#:~:text=Overfitting%20is%20an%20undesirable%20machine,on%20a%20known%20data%20set.>. Access in 22nd of September, 2023.
