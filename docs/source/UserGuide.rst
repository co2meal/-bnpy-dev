============
User Guide
============

**bnpy** is designed for:

* *affordable* training on large datasets
* *easy* comparison of different models
* *easy* comparison of learning algorithms.

This page defines the key concepts and terminology (like 'allocation model' or 'global parameter') that make this general inference framework possible.

TODO:

* explain all possible kwarg settings 
* resource for explaining variational inference
* resource for explaining merges/deletes/births
* practical overview of how the code works
* concept overview for key terms

Prerequisites
-------------
We assume the reader has seen mixture models before.

Modular Representations for Probabilistic Models
====================================

Allocation and Data-Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Consider the following diagrams for standard "building-block" models used throughout machine learning: a mixture, a topic model, and a hidden Markov model.

TODO NICE COMPACT FIGURE HERE

These diagrams are more like model templates than actual models. Each of these models can admit several possible ways to generate data. For example, x could be discrete (like a single word out of a fixed vocabulary), or real-valued (like a Gaussian observation).

We say that each template above is really just a specification of how clusters become assigned or *allocated* to specific data units. For this reason, we call these *allocation models*. 

It is not a coincidence that the observation model 

Global vs. Local Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~

Fundamental operations of variational inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Dataset Representation
=================


Learning Algorithms: Loops of common operations
====================================
