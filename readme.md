# Introduction

On literally a rainy Sunday afternoon I set out to implement a Dirichlet Process mixture model (DPM). My main reason? __To find clusters on streaming data.__
At one point, every data scientist faces this question: _how many clusters do I model?_. This question gets more cumbersome once you work with streaming data. As more data enters, you probably add more clusters. You can do that with heuristics, but the DPM solves it naturally. 

So the DPM covers two of our concerns:

  * We don't know how many clusters to model
  * As more data streams in, we allow more clusters

# Alternatives
Definitely, there exist alternatives for the first concern, _to pick the number of clusters_. Among others:
PCA, MDL, AIC, BIC. Even [Wikipedia](https://en.wikipedia.org/wiki/Determining_the_number_of_clusters_in_a_data_set) dedicates a full page to [the question](https://en.wikipedia.org/wiki/Determining_the_number_of_clusters_in_a_data_set)

# Generative model
The best way to understand the Dirichlet process is to understand how it generates data. Later on, we will discuss how to learn the model.

A Dirichlet process assumes it has infinitely many clusters to draw from. The best intuition I've heard compares this to entering a Chinese restaurant, based on the seemingly infinite supply of tables at Chinese restaurants. The analogy works as follows: The tables represent clusters, and the customers represent our data. When a person enters the restaurant, he chooses to join an existing table with probability proportional to the number of people already sitting at this table (the `N_k`); otherwise, he may choose to sit at a new table k, with probability proportional to `alpha`. Like the following formulas

![equation]( 
https://latex.codecogs.com/gif.latex?p(pick&space;\&space;table&space;\&space;k)=\frac{N_k}{\alpha&space;&plus;&space;N}&space;\&space;,\&space;\&space;p(pick&space;\&space;new\&space;table)&space;=&space;\frac{\alpha}{\alpha&space;&plus;&space;N} )

To narrow this metaphor down to our case of data points. For every new point, it picks a cluster with the above formula. And every cluster has a Gaussian distribution associated to it. So in the code `dpm.z[i]` is an integer that indicates your cluster number. Let's call it `dpm.z[i] = k`, then `dpm.q[k]` is a Gaussian distribution over the datapoints for that cluster.

# Find the number of clusters
The DPM does not require explicitly choose the number of mixtures. That solves our first concern. Unaivoidably, the DPM does require a hyperparameter that controls the number of clusters. It is called `alpha`. Now alpha relates both the numbers of clusters and the number of observations by the following formula:
![equation](https://latex.codecogs.com/gif.latex?E[numClusters]&space;=&space;\alpha&space;log(&space;1&space;&plus;&space;\frac{N}{\alpha}))

Fortunately, the variance is quite big. So changing the alpha does not influence our results too much. ([Source](https://www.stats.ox.ac.uk/~teh/research/npbayes/Teh2010a.pdf))

![equation](https://latex.codecogs.com/gif.latex?var[numClusters]&space;=&space;\alpha&space;log(&space;1&space;&plus;&space;\frac{N}{\alpha}))

To give you some intuition, this plots shows the expected number of clusters when `N=100`.
[alpha_expected_num_clusters](https://github.com/RobRomijnders/dpm/blob/master/clust_dp/alpha_numClust.png?raw=true)


# Learning the DPM
We want to know two things:

  * For each point, the assiginment to a cluster
  * For each assigned cluster, the parameters of its Gaussian distribution

Now we can calculate the mean and variance of a cluster by simple [formulas](https://en.wikipedia.org/wiki/Normal_distribution#Estimation_of_parameters): the sample mean and sample variance.

 _But how to find the cluster assignments?_

The answer is Gibbs sampling: it is too hard to infer all cluster assignments at one time. Remember, we could have infinitely many. So we iteratively assign each point a new cluster. When that iteration becomes stable, we conlude that we found a _valid sample_ of the cluster assignments. 

Honestly, [people write complete books and teach entire courses](https://stats.stackexchange.com/questions/5885/good-sources-for-learning-markov-chain-monte-carlo-mcmc/5889) on Gibbs sampling. Or rather its overarching concept: Markov Chain Monte Carlo sampling. So I'll make no attempt to explain it formally. For now, we assume that as we iterate long enough over our cluster assignments, we find a valid sample. 

Also note that Expectation Maximization won't help us here. The Dirichlet process is a non-parametric model that assumes an infinite amount of clusters. EM only deals with the finite case.

# Results
This is Gibbs sampling learning on a fixed dataset
![dpm](https://github.com/RobRomijnders/dpm/blob/master/clust_dp/im/dpm_100.gif?raw=true)

The reason for doing this project was the case of streaming data. Here are two examples
![dpm](https://github.com/RobRomijnders/dpm/blob/master/clust_dp/im/dpm100_30.gif?raw=true)
![dpm](https://github.com/RobRomijnders/dpm/blob/master/clust_dp/im/dpm100_60.gif?raw=true)

You'll see that as data comes in, the DPM learns more clusters. For the clusters it found, the streaming data makes it more confident.

This is only a small example of a stream. Concerning the stream, you can play with

  * How many points to add? (`main_dpm.py >> points_to_add`)
  * How often to add points? (`main_dpm.py >> interval_to_add`)
  * How many points to start? (`main_dpm.py >> N_start`)

## Technical detail on the Gibbs sampler
You'll notice that the Gibbs sampler can make seemingly irrelevant clusters. In the gif. you'll see a cluster with only few points assigned to it. In the logs, from line to line, you'll see a sudden increase in clusters. This is becuase of the sampling that Gibbs sampling does. Let say we do a Gibbs sample on points `x_i`, then we sample from the conditional probability `p(z_i| ... )` *see line 78 `dpm_class.py >> k_new = ...`. By this sampling, we can have sudden clusters with only few data points. 
This also has a positive side to it: these redundant clusters allow the DPM to escape poor local minima. As opposed to EM (for the finite case) which often gets stuck in poor local minima. (Read Murphy chapter 25.2.4 for more information)


#Further reading

  * Learning the alpha hyperparameter: [Murphy](https://mitpress.mit.edu/books/machine-learning-0) section 25.2.4 on _Fitting a DP mixture model_
  * [Murphy](https://mitpress.mit.edu/books/machine-learning-0) on the entire Dirichlet process mixture model: section 25.2. (Note, I paraphrased his explanation of the Chinese restaurants)
  * Videolecture by Mike Jordan on the Dirichlet process and its famous Chinese restaurant process @ICML 05. [Here](http://videolectures.net/icml05_jordan_dpcrp/)

As always, I am curious to any comments and questions. Reach me at romijndersrob@gmail.com