# paper_SIGDR

This is the PyTorch implementation for paper "Sign-aware Graph Contrastive Learning for Drug Repositioning".

## Introduction

Recently, growing efforts are devoted to applying graph neural networks (GNNs) for effectively modeling drug-disease associations (DDAs). However, current GNN-based methods are generally designed for unsigned graphs and fail to gain complementary insights provided by negative links. Despite the proposal of sign-aware GNNs in general fields, there exist two intractable challenges when indiscriminately deploying prior solutions into drug repositioning. (i) How to explicitly connect the nodes within the same set (disease-disease and drug-drug)? (ii) How to design the contrastive learning objective for signed graphs? To this end, we propose a novel sign-aware graph contrastive learning approach, namely SIGDR, which takes both the positive and negative links from signed biological networks into consideration to identify underlying DDAs. To handle the first challenge, we measure the drug and disease similarity and form signed unipartite graphs according to similarity scores. For the second challenge, a signed bipartite graph is then constructed from the annotated dataset. 

In this paper, we highlight the critical role of integrating complementary negative links into graph-based learning framework. To the best of our knowledge, it is the first time that link signs in biological networks are sufficiently characterized for drug repositioning.
