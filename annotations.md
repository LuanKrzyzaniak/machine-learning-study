## Machine Learning Annotations

Applications of machine learning include (but are not limited to) problems that bypass human capability or demand an unhuman amount of work to build or mantain. Of course, it doensn't mean that we can't apply machine learning to more basic problems. Machine learning is a useful tool.

A general structure for a machine learning algorithm: <br>
**Representation** - this step is where you build your T. Here, you'll decide how to see and organize your data. You can organize it individualy, or in tables, for example. <br>
**Evaluation** - Using an evaluation function, this step consists of ranking the effetiveness of an algorithm. It's used in oriented trainings, so you know how well your learner treated the data and how you need to adjust it. Accuracy and squared error are some metrics.<br>
**Optimization** - That's where you find the best learner and research/apply modifications to improve it. Greedy search and gradient descent are some techniques.

Reference: <https://www.freecodecamp.org/news/machine-learning-principles-explained/#:~:text=The%20three%20components%20that%20make,to%20look%20at%20your%20data.>.

**Training and test sets:** to avoid creating a bias in the pattern recognition, it's important to have a separate set of data for tests. If we test our machines with the same (or part of) the training data, we risk it struggling with different patterns of data in real applications.

**Overfitting:** when your evaluation function is too 'fit' for your training set. In this case, the accuracy ratings become redundant.

**Oriented training:** we provide data and expected results to the machine. In this type of training, data quality becomes of upmost importance, given that the performance of the pattern recognition depends on it.

### Questions

*EVALUATION STEP: how does this apply to unoriented training?*
