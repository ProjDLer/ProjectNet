#Learning Better Word Embedding by Asymmetric Low-Rank Projection of Knowledge Graph

==========

This is the implementation of the ProjectNet algorithm described in the paper [1]. ProjectNet is an algorithm to learn better word embedding vectors by leveraging knowledge graph such as FreeBase.

Build
----------
The ProjectNet is implemented in C++. Use Visual Studio 2013 to open ./src/Solution/Solution.sln and then build the project. It has no dependencies with other libs.

Please refer to ./scripts/perf_test.pl to see how to configure the hyperparameters.

Dataset
----------
The training/testing knowledge graph dataset is stored in ./dataset. The test set is also the new analogical reasoning dataset we described in the paper.

Reference
----------
[1] Learning Better Word Embedding by Asymmetric Low-Rank Projection of Knowledge Graph[J]. submitted to JCST.