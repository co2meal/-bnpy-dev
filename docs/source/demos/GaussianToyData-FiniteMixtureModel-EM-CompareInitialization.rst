
Comparison of initialization methods for Gaussian mixtures
==========================================================

Goal
----

Solution quality for the standard coordinate ascent algorithms like EM
depend heavily on initialization quality. Here, we'll see how bnpy can
be used to run an experiment comparing two initialization methods, one
smarter than the other.

.. code:: python

    import bnpy
    import os

.. code:: python

    %pylab inline
    from bnpy.viz.PlotUtil import ExportInfo
    bnpy.viz.PlotUtil.ConfigPylabDefaults(pylab)


.. parsed-literal::

    Populating the interactive namespace from numpy and matplotlib


Toy dataset : ``AsteriskK8``
----------------------------

We'll use a simple dataset of 2D points, drawn from 8 well-separated
Gaussian clusters.

.. code:: python

    import AsteriskK8
    Data = AsteriskK8.get_data()

We can visualize this dataset as follows:

.. code:: python

    pylab.plot(Data.X[:,0], Data.X[:,1], 'k.');
    pylab.axis('image'); 
    pylab.xlim([-1.75, 1.75]); pylab.xticks([-1, 0, 1]);
    pylab.ylim([-1.75, 1.75]); pylab.yticks([-1, 0, 1]);



.. image:: /demos/GaussianToyData-FiniteMixtureModel-EM-CompareInitialization_8_0.png


Initialization Methods
----------------------

Our intended task is to train a Gaussian mixture model using expectation
maximization (EM) with a maximum likelihood criterion.

We'll consider two methods here to initialize the global parameters
(means and covariances) of the Gaussian mixture model.

For more background on possible initializations, see the `Initialization
documentation TODO <../Code/Initialization.md>`__.

Naive initialization: "select examples uniformly at random"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To initialize K clusters, we select K items uniformly at random from all
N data items, and initialize the model as if each item was the only
member of its corresponding component.

This procedure is called ``randexamples`` in **bnpy**. Note: this is the
default initialization.

Smart initialization: "select examples at random, biased by Euclidean distance"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One problem with the naive method is that it doesn't account for
distances between selected points. When using Gaussian observation
models, it can be beneficial for initialized clusters to be spread out
so a diverse set of points are likely to be represented.

Concretely, we could modify the above procedure to choose K items in a
distance-biased way, instead of uniformly at random. We pick the first
item at random from the data, and then for each successive component
select an item n with probability proportional to its distance from the
nearest chosen item among the :math:`k` previously chosen items.

This procedure is called ``randexamplesbydist`` in **bnpy**.

Running the experiment with **bnpy**
------------------------------------

We'll do 25 separate runs for each of the two initialization methods.
Each run gets at most 50 laps through the data, and uses 10 clusters.

The **initname** argument specifies which initialization method to use,
while the **jobname** is a human-readable name for the experiment.

25 runs from naive initialization: ``randexamples``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    # ExpectedRunTime=130sec
    bnpy.run('AsteriskK8', 'FiniteMixtureModel', 'Gauss', 'EM', 
              K=8, initname='randexamples', jobname='compareinit-K=8-randexamples',
              nLap=100, minLaps=50, nTask=25, printEvery=100);


.. parsed-literal::

    Asterisk Toy Data. 8 true clusters.
      size: 25000 units (single observations)
      dimension: 2
    Allocation Model:  Finite mixture with K=8. Dir prior param 1.00
    Obs. Data  Model:  Gaussian with full covariance.
    Obs. Data  Prior:  Gauss-Wishart on each mean/prec matrix pair: mu, Lam
      E[ mu[k] ]     = [ 0.  0.]
      E[ CovMat[k] ] = 
      [[ 1.  0.]
       [ 0.  1.]]
    Learn Alg: EM
    Trial  1/25 | alg. seed: 7451264 | data order seed: 8541952
    savepath: /results/AsteriskK8/compareinit-K=8-randexamples/1
            1/100 after      0 sec. | K    8 | ev -1.049632959e+07 |  
            2/100 after      0 sec. | K    8 | ev -7.512763667e-01 | Ndiff  390.461 
           78/100 after      9 sec. | K    8 | ev -3.833696280e-01 | Ndiff    0.047 
    ... done. converged.
    Trial  2/25 | alg. seed: 5565568 | data order seed: 7673856
    savepath: /results/AsteriskK8/compareinit-K=8-randexamples/2
            1/100 after      0 sec. | K    8 | ev -1.306152559e+07 |  
            2/100 after      0 sec. | K    8 | ev -8.778910794e-01 | Ndiff  188.995 
           50/100 after      5 sec. | K    8 | ev -1.412879535e-01 | Ndiff    0.034 
    ... done. converged.
    Trial  3/25 | alg. seed: 2559616 | data order seed: 7360256
    savepath: /results/AsteriskK8/compareinit-K=8-randexamples/3
            1/100 after      0 sec. | K    8 | ev -4.937402099e+06 |  
            2/100 after      0 sec. | K    8 | ev -7.401727383e-01 | Ndiff  380.507 
          100/100 after     13 sec. | K    8 | ev -2.275133606e-01 | Ndiff    0.190 
    ... done. not converged. max laps thru data exceeded.
    Trial  4/25 | alg. seed: 7606528 | data order seed: 900864
    savepath: /results/AsteriskK8/compareinit-K=8-randexamples/4
            1/100 after      0 sec. | K    8 | ev -6.396954645e+06 |  
            2/100 after      0 sec. | K    8 | ev -5.124506370e-01 | Ndiff  441.456 
          100/100 after     14 sec. | K    8 | ev -3.892437651e-01 | Ndiff    0.879 
    ... done. not converged. max laps thru data exceeded.
    Trial  5/25 | alg. seed: 543872 | data order seed: 6479872
    savepath: /results/AsteriskK8/compareinit-K=8-randexamples/5
            1/100 after      0 sec. | K    8 | ev -4.707830652e+06 |  
            2/100 after      0 sec. | K    8 | ev -4.475765731e-01 | Ndiff  267.692 
           50/100 after      8 sec. | K    8 | ev  1.084158115e-01 | Ndiff    0.000 
    ... done. converged.
    Trial  6/25 | alg. seed: 8294272 | data order seed: 9149952
    savepath: /results/AsteriskK8/compareinit-K=8-randexamples/6
            1/100 after      0 sec. | K    8 | ev -5.503295677e+06 |  
            2/100 after      0 sec. | K    8 | ev -6.354122899e-01 | Ndiff  667.694 
          100/100 after     12 sec. | K    8 | ev -1.379228250e-01 | Ndiff    0.359 
    ... done. not converged. max laps thru data exceeded.
    Trial  7/25 | alg. seed: 6597632 | data order seed: 3441280
    savepath: /results/AsteriskK8/compareinit-K=8-randexamples/7
            1/100 after      0 sec. | K    8 | ev -5.841809889e+06 |  
            2/100 after      0 sec. | K    8 | ev -6.566263379e-01 | Ndiff  624.176 
          100/100 after     10 sec. | K    8 | ev -1.413068313e-01 | Ndiff    0.169 
    ... done. not converged. max laps thru data exceeded.
    Trial  8/25 | alg. seed: 5652864 | data order seed: 899584
    savepath: /results/AsteriskK8/compareinit-K=8-randexamples/8
            1/100 after      0 sec. | K    8 | ev -6.473739355e+06 |  
            2/100 after      0 sec. | K    8 | ev -7.095056013e-01 | Ndiff  437.260 
           50/100 after      5 sec. | K    8 | ev  1.084158115e-01 | Ndiff    0.000 
    ... done. converged.
    Trial  9/25 | alg. seed: 478720 | data order seed: 3785600
    savepath: /results/AsteriskK8/compareinit-K=8-randexamples/9
            1/100 after      0 sec. | K    8 | ev -6.220224351e+06 |  
            2/100 after      0 sec. | K    8 | ev -7.906995346e-01 | Ndiff  272.741 
          100/100 after     10 sec. | K    8 | ev -1.401775596e-01 | Ndiff    0.175 
    ... done. not converged. max laps thru data exceeded.
    Trial 10/25 | alg. seed: 955776 | data order seed: 6801920
    savepath: /results/AsteriskK8/compareinit-K=8-randexamples/10
            1/100 after      0 sec. | K    8 | ev -4.936372804e+06 |  
            2/100 after      1 sec. | K    8 | ev -6.524215431e-01 | Ndiff  356.245 
           50/100 after      7 sec. | K    8 | ev  1.084158115e-01 | Ndiff    0.000 
    ... done. converged.
    Trial 11/25 | alg. seed: 3296640 | data order seed: 2531072
    savepath: /results/AsteriskK8/compareinit-K=8-randexamples/11
            1/100 after      0 sec. | K    8 | ev -4.702913998e+06 |  
            2/100 after      0 sec. | K    8 | ev -7.248465073e-01 | Ndiff  709.896 
          100/100 after     11 sec. | K    8 | ev -1.376285839e-01 | Ndiff    0.049 
    ... done. converged.
    Trial 12/25 | alg. seed: 2183296 | data order seed: 3886080
    savepath: /results/AsteriskK8/compareinit-K=8-randexamples/12
            1/100 after      0 sec. | K    8 | ev -4.736837291e+06 |  
            2/100 after      0 sec. | K    8 | ev -7.222251942e-01 | Ndiff  535.503 
          100/100 after     11 sec. | K    8 | ev -1.401671553e-01 | Ndiff    0.131 
    ... done. not converged. max laps thru data exceeded.
    Trial 13/25 | alg. seed: 9082752 | data order seed: 8818688
    savepath: /results/AsteriskK8/compareinit-K=8-randexamples/13
            1/100 after      0 sec. | K    8 | ev -2.402842928e+07 |  
            2/100 after      0 sec. | K    8 | ev -7.838083057e-01 | Ndiff  471.542 
          100/100 after     12 sec. | K    8 | ev -3.880923096e-01 | Ndiff    0.340 
    ... done. not converged. max laps thru data exceeded.
    Trial 14/25 | alg. seed: 1826176 | data order seed: 3528320
    savepath: /results/AsteriskK8/compareinit-K=8-randexamples/14
            1/100 after      0 sec. | K    8 | ev -8.121699638e+06 |  
            2/100 after      0 sec. | K    8 | ev -7.848461533e-01 | Ndiff  773.627 
          100/100 after     11 sec. | K    8 | ev -1.383908151e-01 | Ndiff    0.181 
    ... done. not converged. max laps thru data exceeded.
    Trial 15/25 | alg. seed: 2865664 | data order seed: 1024640
    savepath: /results/AsteriskK8/compareinit-K=8-randexamples/15
            1/100 after      0 sec. | K    8 | ev -1.041035933e+07 |  
            2/100 after      0 sec. | K    8 | ev -5.779321685e-01 | Ndiff  146.472 
           55/100 after      6 sec. | K    8 | ev -1.412383845e-01 | Ndiff    0.050 
    ... done. converged.
    Trial 16/25 | alg. seed: 6036480 | data order seed: 8819712
    savepath: /results/AsteriskK8/compareinit-K=8-randexamples/16
            1/100 after      0 sec. | K    8 | ev -1.035807373e+07 |  
            2/100 after      0 sec. | K    8 | ev -5.325935776e-01 | Ndiff  578.055 
          100/100 after     11 sec. | K    8 | ev -4.783631765e-01 | Ndiff    0.207 
    ... done. not converged. max laps thru data exceeded.
    Trial 17/25 | alg. seed: 8729088 | data order seed: 9034368
    savepath: /results/AsteriskK8/compareinit-K=8-randexamples/17
            1/100 after      0 sec. | K    8 | ev -7.870993308e+06 |  
            2/100 after      0 sec. | K    8 | ev -8.532105430e-01 | Ndiff  321.068 
           83/100 after     11 sec. | K    8 | ev -1.379814604e-01 | Ndiff    0.048 
    ... done. converged.
    Trial 18/25 | alg. seed: 8933248 | data order seed: 9882240
    savepath: /results/AsteriskK8/compareinit-K=8-randexamples/18
            1/100 after      0 sec. | K    8 | ev -6.622626036e+06 |  
            2/100 after      0 sec. | K    8 | ev -5.434679187e-01 | Ndiff  459.092 
          100/100 after     12 sec. | K    8 | ev -3.858258541e-01 | Ndiff    0.258 
    ... done. not converged. max laps thru data exceeded.
    Trial 19/25 | alg. seed: 793600 | data order seed: 3803392
    savepath: /results/AsteriskK8/compareinit-K=8-randexamples/19
            1/100 after      0 sec. | K    8 | ev -6.097546211e+06 |  
            2/100 after      0 sec. | K    8 | ev -7.452996663e-01 | Ndiff  689.808 
           50/100 after      5 sec. | K    8 | ev  1.084158115e-01 | Ndiff    0.000 
    ... done. converged.
    Trial 20/25 | alg. seed: 6725120 | data order seed: 1715072
    savepath: /results/AsteriskK8/compareinit-K=8-randexamples/20
            1/100 after      0 sec. | K    8 | ev -8.373704736e+06 |  
            2/100 after      0 sec. | K    8 | ev -8.210094835e-01 | Ndiff  507.022 
          100/100 after     11 sec. | K    8 | ev -1.376522130e-01 | Ndiff    0.177 
    ... done. not converged. max laps thru data exceeded.
    Trial 21/25 | alg. seed: 4116864 | data order seed: 6033536
    savepath: /results/AsteriskK8/compareinit-K=8-randexamples/21
            1/100 after      0 sec. | K    8 | ev -5.684427307e+06 |  
            2/100 after      0 sec. | K    8 | ev -5.823918832e-01 | Ndiff  280.307 
          100/100 after     14 sec. | K    8 | ev -1.405226764e-01 | Ndiff    0.490 
    ... done. not converged. max laps thru data exceeded.
    Trial 22/25 | alg. seed: 4644096 | data order seed: 8644096
    savepath: /results/AsteriskK8/compareinit-K=8-randexamples/22
            1/100 after      0 sec. | K    8 | ev -7.021386421e+06 |  
            2/100 after      0 sec. | K    8 | ev -8.282700188e-01 | Ndiff  402.709 
          100/100 after     12 sec. | K    8 | ev -3.856614345e-01 | Ndiff    0.143 
    ... done. not converged. max laps thru data exceeded.
    Trial 23/25 | alg. seed: 9808000 | data order seed: 2513920
    savepath: /results/AsteriskK8/compareinit-K=8-randexamples/23
            1/100 after      0 sec. | K    8 | ev -7.022194203e+06 |  
            2/100 after      0 sec. | K    8 | ev -6.500824597e-01 | Ndiff  338.034 
           50/100 after      6 sec. | K    8 | ev  1.084158115e-01 | Ndiff    0.000 
    ... done. converged.
    Trial 24/25 | alg. seed: 447360 | data order seed: 6039296
    savepath: /results/AsteriskK8/compareinit-K=8-randexamples/24
            1/100 after      0 sec. | K    8 | ev -4.619402382e+06 |  
            2/100 after      0 sec. | K    8 | ev -6.925151605e-01 | Ndiff  674.341 
           50/100 after      6 sec. | K    8 | ev  1.084158115e-01 | Ndiff    0.000 
    ... done. converged.
    Trial 25/25 | alg. seed: 818944 | data order seed: 7907200
    savepath: /results/AsteriskK8/compareinit-K=8-randexamples/25
            1/100 after      0 sec. | K    8 | ev -6.660144158e+06 |  
            2/100 after      0 sec. | K    8 | ev -8.483899931e-01 | Ndiff  248.956 
          100/100 after     10 sec. | K    8 | ev -3.886293461e-01 | Ndiff    0.667 
    ... done. not converged. max laps thru data exceeded.


25 runs from smart initialization: ``randexamplesbydist``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    # ExpectedRunTime=130sec
    bnpy.run('AsteriskK8', 'FiniteMixtureModel', 'Gauss', 'EM', 
              K=8, initname='randexamplesbydist', jobname='compareinit-K=8-randexamplesbydist',
              nLap=100,  minLaps=50, nTask=25, printEvery=100);


.. parsed-literal::

    Asterisk Toy Data. 8 true clusters.
      size: 25000 units (single observations)
      dimension: 2
    Allocation Model:  Finite mixture with K=8. Dir prior param 1.00
    Obs. Data  Model:  Gaussian with full covariance.
    Obs. Data  Prior:  Gauss-Wishart on each mean/prec matrix pair: mu, Lam
      E[ mu[k] ]     = [ 0.  0.]
      E[ CovMat[k] ] = 
      [[ 1.  0.]
       [ 0.  1.]]
    Learn Alg: EM
    Trial  1/25 | alg. seed: 7451264 | data order seed: 8541952
    savepath: /results/AsteriskK8/compareinit-K=8-randexamplesbydist/1
            1/100 after      0 sec. | K    8 | ev -3.600496754e+06 |  
            2/100 after      0 sec. | K    8 | ev -4.445192734e-01 | Ndiff  575.970 
           50/100 after      6 sec. | K    8 | ev  1.084158115e-01 | Ndiff    0.000 
    ... done. converged.
    Trial  2/25 | alg. seed: 5565568 | data order seed: 7673856
    savepath: /results/AsteriskK8/compareinit-K=8-randexamplesbydist/2
            1/100 after      0 sec. | K    8 | ev -6.853334508e+06 |  
            2/100 after      0 sec. | K    8 | ev -6.669116077e-01 | Ndiff  243.889 
           50/100 after      5 sec. | K    8 | ev  1.084158115e-01 | Ndiff    0.000 
    ... done. converged.
    Trial  3/25 | alg. seed: 2559616 | data order seed: 7360256
    savepath: /results/AsteriskK8/compareinit-K=8-randexamplesbydist/3
            1/100 after      0 sec. | K    8 | ev -4.936890683e+06 |  
            2/100 after      0 sec. | K    8 | ev -3.712569073e-01 | Ndiff  232.021 
           50/100 after      5 sec. | K    8 | ev  1.084158115e-01 | Ndiff    0.000 
    ... done. converged.
    Trial  4/25 | alg. seed: 7606528 | data order seed: 900864
    savepath: /results/AsteriskK8/compareinit-K=8-randexamplesbydist/4
            1/100 after      0 sec. | K    8 | ev -5.027655215e+06 |  
            2/100 after      0 sec. | K    8 | ev -6.451053927e-01 | Ndiff  718.659 
          100/100 after     11 sec. | K    8 | ev -1.383876555e-01 | Ndiff    0.200 
    ... done. not converged. max laps thru data exceeded.
    Trial  5/25 | alg. seed: 543872 | data order seed: 6479872
    savepath: /results/AsteriskK8/compareinit-K=8-randexamplesbydist/5
            1/100 after      0 sec. | K    8 | ev -5.253121018e+06 |  
            2/100 after      0 sec. | K    8 | ev -6.485960982e-01 | Ndiff  546.664 
          100/100 after     10 sec. | K    8 | ev -3.855585329e-01 | Ndiff    0.381 
    ... done. not converged. max laps thru data exceeded.
    Trial  6/25 | alg. seed: 8294272 | data order seed: 9149952
    savepath: /results/AsteriskK8/compareinit-K=8-randexamplesbydist/6
            1/100 after      0 sec. | K    8 | ev -5.152164018e+06 |  
            2/100 after      0 sec. | K    8 | ev -6.003062966e-01 | Ndiff  647.372 
           50/100 after      5 sec. | K    8 | ev  1.084158115e-01 | Ndiff    0.000 
    ... done. converged.
    Trial  7/25 | alg. seed: 6597632 | data order seed: 3441280
    savepath: /results/AsteriskK8/compareinit-K=8-randexamplesbydist/7
            1/100 after      0 sec. | K    8 | ev -4.338306924e+06 |  
            2/100 after      0 sec. | K    8 | ev -6.295960373e-01 | Ndiff  921.600 
          100/100 after      9 sec. | K    8 | ev -1.428707497e-01 | Ndiff    0.665 
    ... done. not converged. max laps thru data exceeded.
    Trial  8/25 | alg. seed: 5652864 | data order seed: 899584
    savepath: /results/AsteriskK8/compareinit-K=8-randexamplesbydist/8
            1/100 after      0 sec. | K    8 | ev -3.867496114e+06 |  
            2/100 after      0 sec. | K    8 | ev -3.557416977e-01 | Ndiff 1004.426 
           55/100 after      7 sec. | K    8 | ev -1.412329469e-01 | Ndiff    0.048 
    ... done. converged.
    Trial  9/25 | alg. seed: 478720 | data order seed: 3785600
    savepath: /results/AsteriskK8/compareinit-K=8-randexamplesbydist/9
            1/100 after      0 sec. | K    8 | ev -5.970583018e+06 |  
            2/100 after      0 sec. | K    8 | ev -7.953409006e-01 | Ndiff  209.212 
          100/100 after     14 sec. | K    8 | ev -3.860541178e-01 | Ndiff    0.738 
    ... done. not converged. max laps thru data exceeded.
    Trial 10/25 | alg. seed: 955776 | data order seed: 6801920
    savepath: /results/AsteriskK8/compareinit-K=8-randexamplesbydist/10
            1/100 after      0 sec. | K    8 | ev -3.662867873e+06 |  
            2/100 after      0 sec. | K    8 | ev -3.546524947e-01 | Ndiff  337.773 
           50/100 after      5 sec. | K    8 | ev  1.084158115e-01 | Ndiff    0.000 
    ... done. converged.
    Trial 11/25 | alg. seed: 3296640 | data order seed: 2531072
    savepath: /results/AsteriskK8/compareinit-K=8-randexamplesbydist/11
            1/100 after      0 sec. | K    8 | ev -3.444473594e+06 |  
            2/100 after      0 sec. | K    8 | ev -2.283710622e-01 | Ndiff  477.753 
           50/100 after      7 sec. | K    8 | ev  1.084158115e-01 | Ndiff    0.000 
    ... done. converged.
    Trial 12/25 | alg. seed: 2183296 | data order seed: 3886080
    savepath: /results/AsteriskK8/compareinit-K=8-randexamplesbydist/12
            1/100 after      0 sec. | K    8 | ev -2.842650163e+06 |  
            2/100 after      0 sec. | K    8 | ev -3.555177094e-01 | Ndiff  322.300 
           50/100 after      6 sec. | K    8 | ev -1.379815027e-01 | Ndiff    0.009 
    ... done. converged.
    Trial 13/25 | alg. seed: 9082752 | data order seed: 8818688
    savepath: /results/AsteriskK8/compareinit-K=8-randexamplesbydist/13
            1/100 after      0 sec. | K    8 | ev -3.295738203e+06 |  
            2/100 after      0 sec. | K    8 | ev -2.612920344e-01 | Ndiff  699.990 
           50/100 after      5 sec. | K    8 | ev  1.084158115e-01 | Ndiff    0.000 
    ... done. converged.
    Trial 14/25 | alg. seed: 1826176 | data order seed: 3528320
    savepath: /results/AsteriskK8/compareinit-K=8-randexamplesbydist/14
            1/100 after      0 sec. | K    8 | ev -4.555194762e+06 |  
            2/100 after      0 sec. | K    8 | ev -5.831747094e-01 | Ndiff  278.429 
           50/100 after      8 sec. | K    8 | ev  1.084158115e-01 | Ndiff    0.000 
    ... done. converged.
    Trial 15/25 | alg. seed: 2865664 | data order seed: 1024640
    savepath: /results/AsteriskK8/compareinit-K=8-randexamplesbydist/15
            1/100 after      0 sec. | K    8 | ev -3.854665487e+06 |  
            2/100 after      0 sec. | K    8 | ev -4.541478251e-01 | Ndiff  455.371 
           50/100 after      5 sec. | K    8 | ev -1.412841024e-01 | Ndiff    0.044 
    ... done. converged.
    Trial 16/25 | alg. seed: 6036480 | data order seed: 8819712
    savepath: /results/AsteriskK8/compareinit-K=8-randexamplesbydist/16
            1/100 after      0 sec. | K    8 | ev -4.048566381e+06 |  
            2/100 after      0 sec. | K    8 | ev -2.703185683e-01 | Ndiff 1381.804 
           50/100 after      5 sec. | K    8 | ev  1.084158115e-01 | Ndiff    0.000 
    ... done. converged.
    Trial 17/25 | alg. seed: 8729088 | data order seed: 9034368
    savepath: /results/AsteriskK8/compareinit-K=8-randexamplesbydist/17
            1/100 after      0 sec. | K    8 | ev -2.969749464e+06 |  
            2/100 after      0 sec. | K    8 | ev -3.077916036e-01 | Ndiff  258.354 
           50/100 after      6 sec. | K    8 | ev -1.405859435e-01 | Ndiff    0.045 
    ... done. converged.
    Trial 18/25 | alg. seed: 8933248 | data order seed: 9882240
    savepath: /results/AsteriskK8/compareinit-K=8-randexamplesbydist/18
            1/100 after      0 sec. | K    8 | ev -3.524794548e+06 |  
            2/100 after      0 sec. | K    8 | ev -5.699028806e-01 | Ndiff  231.840 
           50/100 after      6 sec. | K    8 | ev  1.084158115e-01 | Ndiff    0.000 
    ... done. converged.
    Trial 19/25 | alg. seed: 793600 | data order seed: 3803392
    savepath: /results/AsteriskK8/compareinit-K=8-randexamplesbydist/19
            1/100 after      0 sec. | K    8 | ev -4.298673001e+06 |  
            2/100 after      0 sec. | K    8 | ev -3.379167934e-01 | Ndiff  446.261 
          100/100 after     12 sec. | K    8 | ev -1.405840224e-01 | Ndiff    0.171 
    ... done. not converged. max laps thru data exceeded.
    Trial 20/25 | alg. seed: 6725120 | data order seed: 1715072
    savepath: /results/AsteriskK8/compareinit-K=8-randexamplesbydist/20
            1/100 after      0 sec. | K    8 | ev -2.859909724e+06 |  
            2/100 after      1 sec. | K    8 | ev -3.804304871e-01 | Ndiff  194.074 
           50/100 after      6 sec. | K    8 | ev  1.084158115e-01 | Ndiff    0.000 
    ... done. converged.
    Trial 21/25 | alg. seed: 4116864 | data order seed: 6033536
    savepath: /results/AsteriskK8/compareinit-K=8-randexamplesbydist/21
            1/100 after      0 sec. | K    8 | ev -4.387150964e+06 |  
            2/100 after      0 sec. | K    8 | ev -3.944707405e-01 | Ndiff  624.849 
           50/100 after      5 sec. | K    8 | ev  1.084158115e-01 | Ndiff    0.000 
    ... done. converged.
    Trial 22/25 | alg. seed: 4644096 | data order seed: 8644096
    savepath: /results/AsteriskK8/compareinit-K=8-randexamplesbydist/22
            1/100 after      0 sec. | K    8 | ev -3.180605574e+06 |  
            2/100 after      0 sec. | K    8 | ev -2.893347054e-01 | Ndiff  560.490 
          100/100 after     11 sec. | K    8 | ev -1.405566725e-01 | Ndiff    0.278 
    ... done. not converged. max laps thru data exceeded.
    Trial 23/25 | alg. seed: 9808000 | data order seed: 2513920
    savepath: /results/AsteriskK8/compareinit-K=8-randexamplesbydist/23
            1/100 after      0 sec. | K    8 | ev -4.241417917e+06 |  
            2/100 after      0 sec. | K    8 | ev -6.152510519e-01 | Ndiff  644.980 
           50/100 after      6 sec. | K    8 | ev  1.084158115e-01 | Ndiff    0.000 
    ... done. converged.
    Trial 24/25 | alg. seed: 447360 | data order seed: 6039296
    savepath: /results/AsteriskK8/compareinit-K=8-randexamplesbydist/24
            1/100 after      0 sec. | K    8 | ev -4.778659182e+06 |  
            2/100 after      0 sec. | K    8 | ev -3.904695311e-01 | Ndiff  677.820 
           55/100 after      8 sec. | K    8 | ev -1.412334189e-01 | Ndiff    0.046 
    ... done. converged.
    Trial 25/25 | alg. seed: 818944 | data order seed: 7907200
    savepath: /results/AsteriskK8/compareinit-K=8-randexamplesbydist/25
            1/100 after      0 sec. | K    8 | ev -2.999018493e+06 |  
            2/100 after      0 sec. | K    8 | ev  3.957736786e-02 | Ndiff  159.477 
           50/100 after      6 sec. | K    8 | ev  1.084158115e-01 | Ndiff    0.000 
    ... done. converged.


Performance comparison: training objective as more data is seen
---------------------------------------------------------------

Using **bnpy**'s built-in visualization tools, we can easily make a plot
comparing the two methods' performance at recovering the ideal set of 8
clusters.

This plot shows that across many runs, the ``randexamplesbydist``
procedure often reaches better objective function values than the
simpler, more naive baseline. Of course, poor luck in the random
initialization can still cause both methods to reach very poor objective
values, which correspond to clusterings that group several real clusters
together. However, this happens much less frequently with a good
initialization.

.. code:: python

    bnpy.viz.PlotELBO.plotJobsThatMatchKeywords('AsteriskK8/compareinit-K=8-*')
    pylab.ylim([-1, 0.2]);
    pylab.xlim([1, 50]);
    pylab.legend(loc='lower right');
    pylab.xlabel('num pass thru data');
    pylab.ylabel('train objective');



.. image:: /demos/GaussianToyData-FiniteMixtureModel-EM-CompareInitialization_17_0.png


Discovered clusters: naive initialization
-----------------------------------------

Here we show the discovered clusters for each of the 25 runs. The plot
shows the runs in ranked order, from highest to lowest final objective
function value.

Clearly, the best runs with this method do find all 8 true clusters. In
fact, 6 of the 25 runs do. But, this means that **19 of the 25 runs did
not find the ideal clustering**.

.. code:: python

    figH, axH = pylab.subplots(nrows=5, ncols=5, figsize=(15,15))
    for plotID, rank in enumerate(range(1,26)):
        pylab.subplot(5,5, plotID+1)
        taskidstr = '.rank%d' % (rank)
        bnpy.viz.PlotComps.plotCompsForJob('AsteriskK8/compareinit-K=8-randexamples/', taskids=[taskidstr], figH=figH);
        ELBOpath = os.path.expandvars('$BNPYOUTDIR/AsteriskK8/compareinit-K=8-randexamples/%s/evidence.txt' % (taskidstr))
        finalELBOval = np.loadtxt(ELBOpath)[-1]
        pylab.axis('image'); pylab.xlim([-1.75, 1.75]); pylab.xticks([-1, 0, 1]); pylab.ylim([-1.75, 1.75]); pylab.yticks([-1, 0, 1]);
        pylab.title('Rank %d/25 : %.2f' % (rank, finalELBOval))
    pylab.tight_layout()
    
    # Ignore this block. Only needed for auto-generation of documentation.
    if ExportInfo['doExport']:
        W_in, H_in = pylab.gcf().get_size_inches()
        figpath100 = '../docs/source/_static/GaussianToyData_FiniteMixtureModel_EM_CompareInitialization_%dx%d.png' % (100, 100)
        pylab.savefig(figpath100, bbox_inches=0, pad_inches=0, dpi=ExportInfo['dpi']/W_in);


.. parsed-literal::

    SKIPPED 1 comps with size below 0.00



.. image:: /demos/GaussianToyData-FiniteMixtureModel-EM-CompareInitialization_19_1.png


Discovered clusters: smart initialization
-----------------------------------------

Here, we show the same plots for the smarter, initialize-by-distance
runs.

Many more of the runs have discovered the ideal set of 8 clusters.
However, still only 14 of the 25 runs find all 8 clusters. Clearly,
smarter initialization helps, but we still need to take the best of many
runs to get ideal performance.

.. code:: python

    figH, axH = pylab.subplots(nrows=5, ncols=5, figsize=(15,15))
    for plotID, rank in enumerate(range(1,26)):
        pylab.subplot(5,5, plotID+1)
        taskidstr = '.rank%d' % (rank)
        bnpy.viz.PlotComps.plotCompsForJob('AsteriskK8/compareinit-K=8-randexamplesbydist/', taskids=[taskidstr], figH=figH);
        ELBOpath = os.path.expandvars('$BNPYOUTDIR/AsteriskK8/compareinit-K=8-randexamplesbydist/%s/evidence.txt' % (taskidstr))
        finalELBOval = np.loadtxt(ELBOpath)[-1]
        pylab.axis('image'); pylab.xlim([-1.75, 1.75]); pylab.xticks([-1, 0, 1]); pylab.ylim([-1.75, 1.75]); pylab.yticks([-1, 0, 1]);
        pylab.title('Rank %d/25 : %.2f' % (rank, finalELBOval))
    pylab.tight_layout()



.. image:: /demos/GaussianToyData-FiniteMixtureModel-EM-CompareInitialization_21_0.png


