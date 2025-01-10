# FedProto: Federated Prototype Learning across Heterogeneous Clients

Yue Tan1,
Guodong Long1,
Lu Liu1,
Tianyi Zhou2,3,
Qinghua Lu4,
Jing Jiang1,
Chengqi Zhang 1


![Figure 1: An overview of FedProto in the heterogeneous setting. For example, the first client is to recognize the digits 2,3,4234{2,3,4}, while the m𝑚m-th client is to recognize the digits 4,545{4,5}. First, the clients update their local prototype sets by minimizing the loss of classification error ℒSsubscriptℒ𝑆\mathcal{L}_{S} and the distance between global prototypes and local prototypes ℒRsubscriptℒ𝑅\mathcal{L}_{R}. Then, the clients send their prototypes to the central server. The central server generates global prototypes and returns them to all clients to regularize the training of local models.](/html/2105.00243/assets/x1.png)

Heterogeneity across clients in federated learning (FL) usually hinders the optimization convergence and generalization performance when the aggregation of clients’ knowledge occurs in the gradient space.
For example, clients may differ in terms of data distribution, network latency, input/output space, and/or model architecture, which can easily lead to the misalignment of their local gradients.
To improve the tolerance to heterogeneity, we propose a novel federated prototype learning ( ) framework in which the clients and server communicate the abstract class prototypes instead of the gradients. aggregates the local prototypes collected from different clients, and then sends the global prototypes back to all clients to regularize the training of local models.
The training on each client aims to minimize the classification error on the local data while keeping the resulting local prototypes sufficiently close to the corresponding global ones.
Moreover, we provide a theoretical analysis to the convergence rate of under non-convex objectives.
In experiments, we propose a benchmark setting tailored for heterogeneous FL, with outperforming several recent FL approaches on multiple datasets.

## Introduction

Federated learning (FL) is widely used in multiple applications to enable collaborative learning across a variety of clients without sharing private data. It aims at training a global model on a centralized server while all data are distributed over many local clients and cannot be freely transmitted for privacy or communication concerns . The iterative process of FL has two steps: (1) each local client is synchronized by the global model and then trained using its local data; and (2) the server updates the global model by aggregating all the local models. Considering that the model aggregation occurs in the gradient space, traditional FL still has some practical challenges caused by the heterogeneity of and  . Efficient algorithms suitable to overcome both these two challenges have not yet been fully developed or systematically examined.

To tackle the of data distributions, one straightforward solution is to maintain multiple global models for different local distributions, e.g., the works for clustered FL . Another widely studied strategy is personalized FL where a personalized model is generated for each client by leveraging both global and local information. Nevertheless, most of these methods depend on gradient-based aggregation, resulting in high communication costs and heavy reliance on homogeneous local models.

However, in real-world applications, is common because of varying hardware and computation capabilities across clients . Knowledge Distillation (KD)-based FL addresses this challenge by transferring the teacher model’s knowledge to student models with different model architectures. However, these methods require an extra public dataset to align the student and teacher models’ outputs, increasing the computation costs. Moreover, the performance of KD-based FL can significantly degrade with the increase in the distribution divergence between the public dataset and on-client datasets that are usually non-IID.

Inspired by prototype learning, merging the prototypes over heterogeneous datasets can effectively integrate the feature representations from diverse data distributions . On-client intelligent agents in the FL system can share knowledge by exchanging information in terms of representations, despite statistical and model heterogeneity . For example, when we talk about “dog”, different people will have a unique “imagined picture” or “prototype” to represent the concept “dog”. Their prototypes may be slightly diverse due to different life experience and visual memory.
Exchanging these concept-specific prototypes across people enables them to acquire more knowledge about the concept “dog”. Treating each FL client as a human-like intelligent agent, the core idea of our method is to exchange prototypes rather than share model parameters or raw data, which can naturally match the knowledge acquisition behavior of humans.

In this paper, we propose a novel prototype aggregation-based FL framework where only prototypes are transmitted between the server and clients. The proposed solution does not require model parameters or gradients to be aggregated, so it has a huge potential to be a robust framework for various heterogeneous FL scenarios. Concretely, each client can have different model architectures and input/output spaces, but they can still exchange information by sharing prototypes. Each abstract prototype represents a class by the mean representations transformed from the observed samples belonging to the same class. Aggregating the prototypes allows for efficient communication across heterogeneous clients.

Our main contributions can be summarized as follows:

We propose a benchmark setting tailored for heterogeneous FL that considers a more general heterogeneous scenario across local clients.

We present a novel FL method that significantly improves the communication efficiency in the heterogeneous setting. To the best of our knowledge, we are the first to propose prototype aggregation-based FL.

We theoretically provide a convergence guarantee for our method and carefully derive the convergence rate under non-convex conditions.

Extensive experiments show the superiority of our proposed method in terms of communication efficiency and test performance in several benchmark datasets.

## Related Work

### Heterogeneous Federated Learning

Statistical heterogeneity across clients (also known as the non-IID problem) is the most important challenge of FL.  proposed a local regularization term to optimize each client’s local model. Some recent studies train personalized models to leverage both globally shared information and the personalized part .
The third solution is to provide multiple global models by clustering the local models into multiple groups or clusters. Recently, self-supervised learning strategies are incorporated into the local training phase to handle the heterogeneity challenges . applies meta-training strategy for personalized FL.

Heterogeneous model architecture is another major challenging scenario of FL. The recently proposed KD-based FL can serve as an alternative solution to address this challenge. In particular, with the assumption of adding a shared toy dataset in the federated setting, these KD-based FL methods can distill knowledge from a teacher model to student models with different model architectures.
Some recent studies have also attempted to combine the neural architecture search with federated learning , which can be applied to discover a customized model architecture for each group of clients with different hardware capabilities and configurations. A collective learning platform is proposed to handle heterogeneous architectures without access to the local training data and architectures in . Moreover, functionality-based neural matching across local models can aggregate neurons with similar functionality regardless of the variance of the model architectures.

However, most of these mentioned FL methods focus on only one heterogeneous challenging scenario. All of them use gradient-based aggregation methods which will raise concerns about communication efficiency and gradient-based attacks .

### Prototype Learning

The concept of prototypes (the mean of multiple features) has been explored in a variety of tasks. In image classification, a prototype can be a proxy of a class and is calculated as the mean of the feature vectors within every class . In action recognition, the features of a video in different timestamps can be averaged to serve as the representation of the video . Aggregated local features can serve as descriptors for image retrieval .
Averaging word embeddings as the representation of a sentence can achieve competitive performance on multiple NLP benchmarks .
The authors in use prototypes to represent task-agnostic information in distributed machine learning and propose a new fusion paradigm to integrate those prototypes to generate a new model for a new task. In , prototype margins are used to optimize visual feature representations for FL.
In our paper, we borrow the concept of prototypes to represent one class and apply prototype aggregation in the setting of heterogeneous FL.

In general, prototypes are widely used in learning scenarios with a limited number of training samples . This learning scenario is consistent with the latent assumption of cross-client FL: that each client has a limited number of instances to independently train a model with the desired performance. The assumption has been widely supported by the FL-based benchmark datasets and in related applications, such as healthcare and street image object detection .

${2,3,4}$$m$${4,5}$$\mathcal{L}_{S}$$\mathcal{L}_{R}$## Problem Setting

### Heterogeneous Federated Learning Setting

In federated learning, each client owns a local private dataset ${D}_{i}$ drawn from distribution $\mathbb{P}_{i}(x,y)$ , where $x$ and $y$ denote the input features and corresponding class labels, respectively. Usually, clients share a model $\mathcal{F}(\omega;x)$ with the same architecture and hyperparameters. This model is parameterized by learnable weights $\omega$ and input features $x$ . The objective function of  is:

${D}_{i}$$\mathbb{P}_{i}(x,y)$$x$$y$$\mathcal{F}(\omega;x)$$\omega$$x$
$$\operatorname*{arg\,min}_{\omega}\sum_{i=1}^{m}\frac{|D_{i}|}{N}\mathcal{L}_{S}(\mathcal{F}(\omega;x),y),$$

$$\operatorname*{arg\,min}_{\omega}\sum_{i=1}^{m}\frac{|D_{i}|}{N}\mathcal{L}_{S}(\mathcal{F}(\omega;x),y),$$
where $\omega$ is the global model’s parameters, $m$ denotes the number of clients, $N$ is the total number of instances over all clients, $\mathcal{F}$ is the shared model, and $\mathcal{L}_{S}$ is a general definition of any supervised learning task (e.g., a cross-entropy loss).

$\omega$$m$$N$$\mathcal{F}$$\mathcal{L}_{S}$However, in a real-world FL environment, each client may represent a mobile phone with a specific user behavior pattern or a sensor deployed in a particular location, leading to statistical and/or model heterogeneous environment. In the statistical heterogeneity setting, $\mathbb{P}_{i}$ varies across clients, indicating heterogeneous input/output space for $x$ and $y$ . For example, $\mathbb{P}_{i}$ on different clients can be the data distributions over different subsets of classes. In the model heterogeneity setting, $\mathcal{F}_{i}$ varies across clients, indicating different model architectures and hyperparameters. For the $i$ -th client, the training procedure is to minimize the loss as defined below:

$\mathbb{P}_{i}$$x$$y$$\mathbb{P}_{i}$$\mathcal{F}_{i}$$i$
$$\operatorname*{arg\,min}_{\omega_{1},\omega_{2},\dots,\omega_{m}}\sum_{i=1}^{m}\frac{|D_{i}|}{N}\mathcal{L}_{S}(\mathcal{F}_{i}(\omega_{i};x),y).$$

$$\operatorname*{arg\,min}_{\omega_{1},\omega_{2},\dots,\omega_{m}}\sum_{i=1}^{m}\frac{|D_{i}|}{N}\mathcal{L}_{S}(\mathcal{F}_{i}(\omega_{i};x),y).$$
Most existing methods cannot well handle the heterogeneous settings above. In particular, the fact that $\mathcal{F}_{i}$ has a different model architecture would cause $\omega_{i}$ to have a different format and size. Thus, the global model’s parameter $\omega$ cannot be optimized by averaging $\omega_{i}$ . To tackle this challenge, we propose to communicate and aggregate prototypes in FL.

$\mathcal{F}_{i}$$\omega_{i}$$\omega$$\omega_{i}$### Prototype-Based Aggregation Setting

Heterogeneous FL focuses on the robustness to tackle heterogeneous input/output spaces, distributions and model architectures. For example, the datasets $D_{i}$ and $D_{k}$ on two clients $i$ and $k$ may take different statistical distributions of labels. This is common for a photo classification APP installed on mobile clients, where the server needs to recognize many classes $\mathbb{C}=\{{C}^{(1)},{C}^{(2)},\dots\}$ , while each client only needs to recognize a few classes that constitute a subset of $\mathbb{C}$ . The class set can vary across clients, though there are overlaps.

$D_{i}$$D_{k}$$i$$k$$\mathbb{C}=\{{C}^{(1)},{C}^{(2)},\dots\}$$\mathbb{C}$In general, the deep learning-based models comprise two parts: (1) representation layers (a.k.a. embedding functions) to transform the input from the original feature space to the embedding space; and (2) decision layers to make a classification decision for a given learning task.

##### Representation layers

The embedding function of the $i$ -th client is $f_{i}(\phi_{i})$ parameterized by $\phi_{i}$ . We denote $h_{i}=f_{i}(\phi_{i};x)$ as the embeddings of $x$ .

$i$$f_{i}(\phi_{i})$$\phi_{i}$$h_{i}=f_{i}(\phi_{i};x)$$x$##### Decision layers

Given a supervised learning task, a prediction for $x$ can be generated by the function $g_{i}(\nu_{i})$ parameterized by $\nu_{i}$ . So, the labelling function can be written as $\mathcal{F}_{i}(\phi_{i},\nu_{i})=g_{i}(\nu_{i})\circ f_{i}(\phi_{i})$ , and we use $\omega_{i}$ to represent $(\phi_{i},\nu_{i})$ for short.

$x$$g_{i}(\nu_{i})$$\nu_{i}$$\mathcal{F}_{i}(\phi_{i},\nu_{i})=g_{i}(\nu_{i})\circ f_{i}(\phi_{i})$$\omega_{i}$$(\phi_{i},\nu_{i})$##### Prototype

We define a prototype ${C}^{(j)}$ to represent the $j$ -th class in $\mathbb{C}$ . For the $i$ -th client, the prototype is the mean value of the embedding vectors of instances in class $j$ ,

${C}^{(j)}$$j$$\mathbb{C}$$i$$j$
$$C_{i}^{(j)}=\frac{1}{|D_{i,j}|}\sum_{(x,y)\in D_{i,j}}f_{i}(\phi_{i};x),$$

$$C_{i}^{(j)}=\frac{1}{|D_{i,j}|}\sum_{(x,y)\in D_{i,j}}f_{i}(\phi_{i};x),$$
where $D_{i,j}$ , a subset of the local dataset $D_{i}$ , is comprised of training instances belonging to the $j$ -th class.

$D_{i,j}$$D_{i}$$j$

##### Prototype-based model inference

In the inference stage of the learning task, we can simply predict the label $\hat{y}$ to an instance $x$ by measuring the L2 distance between the instance’s representational vector $f(\phi;x)$ and the prototype ${C}^{(j)}$ as follows:

$\hat{y}$$x$$f(\phi;x)$${C}^{(j)}$
$$\hat{y}=\operatorname*{arg\,min}_{j}||f(\phi;x)-C^{(j)}||_{2}.$$

$$\hat{y}=\operatorname*{arg\,min}_{j}||f(\phi;x)-C^{(j)}||_{2}.$$
## Methodology

We propose a solution for heterogeneous FL that uses prototypes as the key component for exchanging information across the server and the clients.

An overview of the proposed framework is shown in Figure . The central server receives local prototype sets $C_{1},C_{2},\ldots,C_{m}$ from $m$ local clients, and then aggregates the prototypes by averaging them. In the heterogeneous FL setting, these prototype sets overlap but are not the same. Taking the MNIST dataset as an example, the first client is to recognize the digits ${2,3,4}$ , while another client is to recognize the digits ${4,5}$ . These are two different handwritten digits set; nonetheless, there is an overlap. The server automatically aggregates prototypes from the overlapping class space across the clients.

$C_{1},C_{2},\ldots,C_{m}$$m$${2,3,4}$${4,5}$Using prototypes in FL, we do not need to exchange gradients or model parameters, which means that the proposed solution can tackle heterogeneous model architectures. Moreover, the prototype-based FL does not require each client to provide the same classes, meaning the heterogeneous class spaces are well supported. Thus, heterogeneity challenges in FL can be addressed.

### Optimization Objective

The objective of is to solve a joint optimization problem on a distributed network. applies prototype-based communication, which allows a local model to align its prototypes with other local models while minimizing the sum of loss for all clients’ local learning tasks. The objective of federated prototype learning across heterogeneous clients can be formulated as

$\displaystyle\operatorname*{arg\,min}_{\left\{\bar{C}^{(j)}\right\}_{j=1}^{|\mathbb{C}|}}\sum_{i=1}^{m}\frac{|D_{i}|}{N}\mathcal{L}_{S}(\mathcal{F}_{i}(\omega_{i};x),y)+$$\displaystyle\lambda\cdot\sum_{j=1}^{|\mathbb{C}|}\sum_{i=1}^{m}\frac{|D_{i,j}|}{N_{j}}\mathcal{L}_{R}(\bar{C}_{i}^{(j)},C_{i}^{(j)}),$where $\mathcal{L}_{S}$ is the loss of supervised learning (as defined in Eq. ( )) and $\mathcal{L}_{R}$ is a regularization term that measures the distance (we use L2 distance) between a local prototype $C^{(j)}$ and the corresponding global prototypes $\bar{C}^{(j)}$ . $N$ is the total number of instances over all clients, and $N_{j}$ is the number of instances belonging to class $j$ over all clients.

$\mathcal{L}_{S}$$\mathcal{L}_{R}$$C^{(j)}$$\bar{C}^{(j)}$$N$$N_{j}$$j$The optimization problem can be addressed by alternate minimization that iterates the following two steps: (1) minimization w.r.t. each $\omega_{i}$ with $\bar{C}_{i}^{(j)}$ fixed; and (2) minimization w.r.t. $\bar{C}_{i}^{(j)}$ with all $\omega_{i}$ fixed. In a distributed setting, step (1) reduces to conventional supervised learning on each client using its local data, while step (2) aggregates local prototypes from local clients on the server end. Further details concerning these two steps can be seen in Algorithm .

$\omega_{i}$$\bar{C}_{i}^{(j)}$$\bar{C}_{i}^{(j)}$$\omega_{i}$$D_{i},\omega_{i},i=1,\cdots,m$

$D_{i},\omega_{i},i=1,\cdots,m$$\left\{\bar{C}^{(j)}\right\}$$T=1,2,...$$i$$C_{i}\leftarrow$$\left(i,\bar{C}_{i}\right)$$C_{i}$$\{\bar{C}^{(j)}\}$:

$\left(i,\bar{C}_{i}\right)$$\left(x_{i},y_{i}\right)\in D_{i}$$C^{(i)}$### Global Prototype Aggregation

Given the data and model heterogeneity in the participating clients, the optimal model parameters for each client are not the same. This means that gradient-based communication cannot sufficiently provide useful information to each client. However, the same label space allows the participating clients to share the same embedding space and information can be efficiently exchanged across heterogeneous clients by aggregating prototypes according to the classes they belong to.

Given a class $j$ , the server receives prototypes from a set of clients that have class $j$ . A global prototype $\bar{C}^{(j)}$ for class $j$ is generated after the prototype aggregating operation,

$j$$j$$\bar{C}^{(j)}$$j$
$$\bar{C}^{(j)}=\frac{1}{\left|\mathcal{N}_{j}\right|}\sum_{i\in{\mathcal{N}_{j}}}\frac{|D_{i,j}|}{N_{j}}C^{(j)}_{i},$$

$$\bar{C}^{(j)}=\frac{1}{\left|\mathcal{N}_{j}\right|}\sum_{i\in{\mathcal{N}_{j}}}\frac{|D_{i,j}|}{N_{j}}C^{(j)}_{i},$$
where $C_{i}^{(j)}$ denotes the prototype of class $j$ from client $i$ , and $\mathcal{N}_{j}$ denotes the set of clients that have class $j$ .

$C_{i}^{(j)}$$j$$i$$\mathcal{N}_{j}$$j$### Local Model Update

The client needs to update the local model to generate a consistent prototype across the clients. To this end, a regularization term is added to the local loss function, enabling the local prototypes $C_{i}^{(j)}$ to approach global prototypes $\bar{C}_{i}^{(j)}$ while minimizing the loss of the classification error. In particular, the loss function is defined as follows:

$C_{i}^{(j)}$$\bar{C}_{i}^{(j)}$
$$\mathcal{L}(D_{i},\omega_{i})=\mathcal{L}_{S}(\mathcal{F}_{i}(\omega_{i};x),y)+\lambda\cdot\mathcal{L}_{R}\left(\bar{C}_{i},C_{i}\right),$$

$$\mathcal{L}(D_{i},\omega_{i})=\mathcal{L}_{S}(\mathcal{F}_{i}(\omega_{i};x),y)+\lambda\cdot\mathcal{L}_{R}\left(\bar{C}_{i},C_{i}\right),$$
where $\lambda$ is an importance weight, and $\mathcal{L}_{R}$ is the regularization term that can be defined as:

$\lambda$$\mathcal{L}_{R}$
$$\mathcal{L}_{R}=\sum_{j}d(C_{i}^{(j)},\bar{C}_{i}^{(j)}),$$

$$\mathcal{L}_{R}=\sum_{j}d(C_{i}^{(j)},\bar{C}_{i}^{(j)}),$$
where $d$ is a distance metric of local generated prototypes $C^{(j)}$ and global aggregated prototypes $\bar{C}^{(j)}$ . The distance measurement can take a variety of forms, such as L1 distance, L2 distance, and earth mover’s distance.

$d$$C^{(j)}$$\bar{C}^{(j)}$### Convergence Analysis

We provide insights into the convergence analysis for . We denote the local objective function defined in Eq. as $\mathcal{L}$ with a subscript indicating the number of iterations and make the following assumptions similar to existing general frameworks .

$\mathcal{L}$(Lipschitz Smooth).

$L_{1}$$L_{1}$$\displaystyle\|\nabla\mathcal{L}_{{t_{1}}}$$\displaystyle-\nabla\mathcal{L}_{{t_{2}}}\|_{2}\leq L_{1}\|\omega_{{i,t_{1}}}-\omega_{{i,t_{2}}}\|_{2},$$\displaystyle\forall t_{1},t_{2}>0,i\in\{1,2,\dots,m\}.$$\displaystyle\mathcal{L}_{{t_{1}}}-\mathcal{L}_{{t_{2}}}$$\displaystyle\leq\langle\nabla\mathcal{L}_{{t_{2}}},(\omega_{{i,t_{1}}}-\omega_{{i,t_{2}}})\rangle+\frac{L_{1}}{2}\|\omega_{{i,t_{1}}}-\omega_{{i,t_{2}}}\|_{2}^{2},$$\displaystyle\forall t_{1},t_{2}>0,\quad i\in\{1,2,\dots,m\}.$(Unbiased Gradient and Bounded Variance).

$g_{i,t}=\nabla\mathcal{L}(\omega_{t},\xi_{t})$
$${\mathbb{E}}_{\xi_{i}\sim D_{i}}[g_{i,t}]=\nabla\mathcal{L}(\omega_{i,t})=\nabla\mathcal{L}_{t},\forall i\in\{1,2,\dots,m\},$$

$${\mathbb{E}}_{\xi_{i}\sim D_{i}}[g_{i,t}]=\nabla\mathcal{L}(\omega_{i,t})=\nabla\mathcal{L}_{t},\forall i\in\{1,2,\dots,m\},$$
$\sigma^{2}$
$${\mathbb{E}}[{\|g_{i,t}-\nabla\mathcal{L}(\omega_{i,t})\|}_{2}^{2}]\leq\sigma^{2},\forall i\in\{1,2,\dots,m\},\sigma^{2}\geq 0.$$

$${\mathbb{E}}[{\|g_{i,t}-\nabla\mathcal{L}(\omega_{i,t})\|}_{2}^{2}]\leq\sigma^{2},\forall i\in\{1,2,\dots,m\},\sigma^{2}\geq 0.$$
(Bounded Expectation of Euclidean norm of Stochastic Gradients).

$G$
$${\mathbb{E}}[\|g_{i,t}\|_{2}]\leq G,\forall i\in\{1,2,\dots,m\}.$$

$${\mathbb{E}}[\|g_{i,t}\|_{2}]\leq G,\forall i\in\{1,2,\dots,m\}.$$
(Lipschitz Continuity).

$L_{2}$$\displaystyle\|f_{i}(\phi_{i,t_{1}})$$\displaystyle-f_{i}(\phi_{i,t_{2}})\|\leq L_{2}\|\phi_{i,t_{1}}-\phi_{i,t_{2}}\|_{2},$$\displaystyle\forall t_{1},t_{2}>0,i\in\{1,2,\dots,m\}.$Based on the above assumptions, we present the theoretical results for the non-convex problem. The expected decrease per round is given in Theorem . We denote $e\in\{1/2,1,2,\dots,E$ } as the local iteration, and $t$ as the global communication round. Moreover, $tE$ represents the time step before prototype aggregation, and $tE+1/2$ represents the time step between prototype aggregation and the first iteration of the current round.

$e\in\{1/2,1,2,\dots,E$$t$$tE$$tE+1/2$(One-round deviation).

$\displaystyle{\mathbb{E}}[\mathcal{L}_{(t+1)E+1/2}]\leq$$\displaystyle\mathcal{L}_{tE+1/2}-\left(\eta-\frac{L_{1}\eta^{2}}{2}\right)\sum_{e=1/2}^{E-1}\|\nabla\mathcal{L}_{tE+e}\|_{2}^{2}$$\displaystyle+\frac{L_{1}E\eta^{2}}{2}\sigma^{2}+\lambda L_{2}\eta EG.$Theorem indicates the deviation bound of the local objective function for an arbitrary client after each communication round. Convergence can be guaranteed when there is a certain expected one-round decrease, which can be achieved by choosing appropriate $\eta$ and $\lambda$ .

$\eta$$\lambda$(Non-convex convergence).

$\mathcal{L}$
$$\eta_{e^{\prime}}<\frac{2(\sum_{e=1/2}^{e^{\prime}}\|\nabla\mathcal{L}_{tE+e}\|_{2}^{2}-\lambda L_{2}EG)}{L_{1}(\sum_{e=1/2}^{e^{\prime}}\|\nabla\mathcal{L}_{tE+e}\|_{2}^{2}+E\sigma^{2})},$$

$$\eta_{e^{\prime}}<\frac{2(\sum_{e=1/2}^{e^{\prime}}\|\nabla\mathcal{L}_{tE+e}\|_{2}^{2}-\lambda L_{2}EG)}{L_{1}(\sum_{e=1/2}^{e^{\prime}}\|\nabla\mathcal{L}_{tE+e}\|_{2}^{2}+E\sigma^{2})},$$
$e^{\prime}=1/2,1,\dots,E-1$
$$\lambda_{t}<\frac{\|\nabla\mathcal{L}_{tE+1/2}\|_{2}^{2}}{L_{2}EG}.$$

$$\lambda_{t}<\frac{\|\nabla\mathcal{L}_{tE+1/2}\|_{2}^{2}}{L_{2}EG}.$$
Corollary is to ensure the expected deviation of $\mathcal{L}$ to be negative, so the loss function converges. It can guide the choice of appropriate values for the learning rate $\eta$ and the importance weight $\lambda$ to guarantee the convergence.

$\mathcal{L}$$\eta$$\lambda$(Non-convex convergence rate of )

$\Delta=\mathcal{L}_{0}-\mathcal{L}^{*}$$\mathcal{L}^{*}$$\epsilon>0$
$$T=\frac{2\Delta}{E\epsilon(2\eta-L_{1}\eta^{2})-E\eta(L_{1}\eta\sigma^{2}+2\lambda L_{2}G)}$$

$$T=\frac{2\Delta}{E\epsilon(2\eta-L_{1}\eta^{2})-E\eta(L_{1}\eta\sigma^{2}+2\lambda L_{2}G)}$$

$$\frac{1}{TE}\sum_{t=0}^{T-1}\sum_{e=1/2}^{E-1}{\mathbb{E}}[\|\nabla\mathcal{L}_{tE+e}\|_{2}^{2}]<\epsilon,$$

$$\frac{1}{TE}\sum_{t=0}^{T-1}\sum_{e=1/2}^{E-1}{\mathbb{E}}[\|\nabla\mathcal{L}_{tE+e}\|_{2}^{2}]<\epsilon,$$

$$\eta<\frac{2(\epsilon-\lambda L_{2}G)}{L_{1}(\epsilon+\sigma^{2})}\ and\ \ \lambda<\frac{\epsilon}{L_{2}G}.$$

$$\eta<\frac{2(\epsilon-\lambda L_{2}G)}{L_{1}(\epsilon+\sigma^{2})}\ and\ \ \lambda<\frac{\epsilon}{L_{2}G}.$$
Theorem provides the convergence rate, which can confine the expected L2-norm of gradients to any bound, denoted as $\epsilon$ , after carefully selecting the number of communication rounds $T$ and hyperparameters including $\eta$ and $\lambda$ . The smaller $\epsilon$ is, the larger $T$ is, which means that the tighter the bound is, more communication rounds is required. A detailed proof and analysis are given in Appendix B.

$\epsilon$$T$$\eta$$\lambda$$\epsilon$$T$## Discussion

In this section, we discuss the superiority of from three perspectives: model inference, communication efficiency, and privacy preserving.

### Model Inference

Unlike many FL methods, the global model in FedProto is not a classifier but a set of class prototypes. When a new client is added to the network, one can initialize its local model with the representation layers of a pre-trained model, e.g. a ResNet18 on ImageNet, and random decision layers. Then, the local client will download the global prototypes of the classes covered in its local dataset and fine-tune the local model by minimizing the local objective. This can support new clients with novel model architectures and spend less time fine-tuning the model on heterogeneous datasets.

### Communication Efficiency

Our proposed method only transmits prototypes between the server and clients. In general, the size of the prototypes is usually much smaller than the size of the model parameters. Taking MNIST as an example, the prototype size is 50 for each class, while the number of model parameters is 21,500. More details can be found in the experimental section.

$\times 10^{3}$$n=3$$n=4$$n=5$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$### Privacy Preserving

The proposed requires the exchange of prototypes rather than model parameters between the server and the clients. This property brings benefits to FL in terms of privacy preserving. First, prototypes naturally protect the data privacy, because they are 1D-vectors generated by averaging the low-dimension representations of samples from the same class, which is an irreversible process. Second, attackers cannot reconstruct raw data from prototypes without the access to local models. Moreover, can be integrated with various privacy-preserving techniques to further enhance the reliability of the system.

## Experiments

### Training Setups

##### Datasets and local models

We implement the typical federated setting where each client owns its local data and transmits/receives information to/from the central server. We use three popular benchmark datasets: MNIST , FEMNIST and CIFAR10 . We consider a multi-layer CNN which consists of 2 convolutional layers then 2 fully connected layers for both MNIST and FEMNIST, and ResNet18 for CIFAR10.

##### Local tasks

Each client learns a supervised learning task. In particular, to illustrate the local task, we borrow the concept of $n$ -way $k$ -shot from few-shot learning where $n$ controls the number of classes and $k$ controls the number of training instances per class. To mimic the heterogeneous scenario, we randomly change the value of $n$ and $k$ in different clients. We define an average value for $n$ and $k$ , and then add a random noise to each user’s $n$ as well as $k$ . The purpose of the variance of $n$ is to control the heterogeneity of the class space, while the purpose of the variance of $k$ is to control the imbalance in data size.

$n$$k$$n$$k$$n$$k$$n$$k$$n$$k$$n$$k$##### Baselines of FL

We study the performance of under both the statistical and model heterogeneous settings ( ) and make comparisons with baselines, including where an individual model is trained for each client without any communication with others,  ,  ,  ,  , and  .


![(a) FedProto](/html/2105.00243/assets/x2.png)


![(a) FedProto](/html/2105.00243/assets/x2.png)


![](/html/2105.00243/assets/x2.png)


![(b) FedAvg](/html/2105.00243/assets/x3.png)


![](/html/2105.00243/assets/x3.png)


![(c) FeSEM](/html/2105.00243/assets/x4.png)


![](/html/2105.00243/assets/x4.png)


![(d) FedPer](/html/2105.00243/assets/x5.png)


![](/html/2105.00243/assets/x5.png)

$n=3$##### Implementation Details

We implement and the baseline methods in PyTorch. We use 20 clients for all datasets and all clients are sampled in each communication round. The average size of each class in each client is set to be 100. For MNIST and FEMNIST dataset, our initial set of hyperparameters was taken directly from the default set of hyperparamters in . For CIFAR10, ResNet18 pre-trained on ImageNet is used as the initial model. The initial average test accuracy of the pre-trained network on CIFAR10 is 27.55 $\%$ . A detailed setup including the choice of hyperparameters is given in Appendix A.

$\%$### Performance in Non-IID Federated Setting

We compare with other baseline methods that are either classical FL methods or FL methods with an emphasis on statistical heterogeneity. All methods are adapted to fit this heterogeneous setting.

##### Statistical heterogeneity simulations

In our setting, we assume that all clients perform learning tasks with heterogeneous statistical distributions. In order to simulate different levels of heterogeneity, we fix the standard deviation to be 1 or 2, aiming to create heterogeneity in both class spaces and data sizes, which is common in real-world scenarios.

##### Model heterogeneity simulations

For the model heterogeneous setting, we consider minor differences in model architectures across clients. In MNIST and FEMNIST, the number of output channels in the convolutional layers is set to either 18, 20 or 22, while in CIFAR10, the stride of convolutional layers is set differently across different clients. This kind of model heterogeneity brings about challenges for model parameter averaging because the parameters in different clients are not always the same size.

The average test accuracy over all clients is reported in Table . It can be seen that achieves the highest accuracy and the least variance in most cases, ensuring uniformity among heterogeneous clients.

##### Communication efficiency

Communication costs have always been posed as a challenge in FL, considering several limitations in existing communication channels. Therefore, we also report the number of communication rounds required for convergence and the number of parameters communicated per round in Table . It can be seen that the number of parameters communicated per round in is much lower than in the case of . Furthermore, requires the fewest communication rounds for the local optimization. This suggests that when the heterogeneity level is high across the clients, sharing more parameters does not always lead to better results. It is more important to identify which part to share in order to benefit the current system to a great extent. More performance results are shown in Appendix A.

##### Visualization of prototypes achieved by FedProto

We visualize the samples in MNIST test set by t-SNE . In Figure , small points in different colors represent samples in different classes, with large points representing corresponding global prototypes. In Figure , and , the points in different colors refer to the representations of samples belonging to different classes. Better generalization means that there are more samples within the same class cluster in the same area, which can be achieved in a centralized setting, while better personalization means that it is easier to determine to which client the samples belong. It can be seen that samples within the same class but from various clients are close but separable in . This indicates that is more successful in achieving the balance between generalization and personalization, while other methods lacks either the generalization or the personalization ability.

##### Scalability of FedProto on varying number of samples

Figure shows that can scale to scenarios with fewer samples available on clients. The test accuracy consistently decreases when there are fewer samples for training, but drops more slowly than as a result of its adaptability and scalability on various data sizes.

##### FedProto under varying λ𝜆\lambda

$\lambda$Figure shows the varying performance under different values of $\lambda$ in Eq. ( ). We tried a set of values selected from $[0,4]$ and reported the average test accuracy and proto distance loss with $n$ =3, $k$ =100 in FEMNIST dataset. The best value of $\lambda$ is $1$ in this scenario. As $\lambda$ increases, the proto distance loss (regularization term) decreases, while the average test accuracy experiences a sharp rise from $\lambda$ =0 to $\lambda$ =1 before a drop in the number of 6 $\%$ , demonstrating the efficacy of prototype aggregation.

$\lambda$$[0,4]$$n$$k$$\lambda$$1$$\lambda$$\lambda$$\lambda$$\%$$\lambda$## Conclusion

In this paper, we propose a novel prototype aggregation-based FL method to tackle challenging FL scenarios with heterogeneous input/output spaces, data distributions, and model architectures. The proposed method collaboratively trains intelligent models by exchanging prototypes rather than gradients, which offers new insights for designing prototype-based FL. The effectiveness of the proposed method has been comprehensively analyzed from both theoretical and experimental perspectives.

# References

We present the related supplements in following sections.

## Experimental Details and Extra Results

### Experimental Details

Local clients are trained by SGD optimizer, with a learning rate of $0.01$ and momentum of $0.5$ .
Regarding the crucial hyperparameter $\lambda$ , we tune the best $\lambda$ from a limited candidate set by . The best $\lambda$ values for MNIST, FEMNIST and CIFAR10 are $1$ , $1$ and $0.1$ , respectively. The number of local epochs and local batch size are set to be 1 and 8, respectively, for all datasets. The heterogeneity level of clients is controlled by the standard deviation of $n$ . The higher this is, the more heterogeneous the clients are.

$0.01$$0.5$$\lambda$$\lambda$$\lambda$$1$$1$$0.1$$n$### Extra Results

The complete experimental results show the performance of and on three benchmark datasets MNIST, FEMNIST, and CIFAR10. Compared with existing FL methods, yields higher test accuracy while resulting in lower communication costs under different heterogeneous settings. Additionally, it can be used in model heterogeneous scenarios and achieves performance similar to that in homogeneous scenarios.

For MNIST, we evaluate local test sets and report the evaluation results in Table . It appears that achieves strong performance with low communication cost. The local average test accuracy of is greater than for the , , and algorithms in all the settings.

For FEMNIST, the evaluation results are reported in Table . We consider the standard deviation of $n$ to be 1 and 2. The results show that, for , the variance of the accuracy across clients is much smaller than for other FL methods, thus ensuring uniformity among heterogeneous clients. allows us to better utilize the local FEMNIST dataset distribution while using around $0.025\%$ of the total parameters communicated.

$n$$0.025\%$For CIFAR10, as can be seen in Table , converges faster in the presence of heterogeneity in most cases. In and , the number of parameters communicated per round is much lower than the baseline methods, meaning greatly reduced communication costs.

$n$$\times 10^{3}$$n=3$$n=4$$n=5$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$n$$\times 10^{3}$$n=3$$n=4$$n=5$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$n$$\times 10^{4}$$n=3$$n=4$$n=5$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\times 10^{4}$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\times 10^{4}$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\times 10^{4}$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\times 10^{4}$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\times 10^{4}$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$$\pm$## Convergence Analysis for FedProto

### Additional Notation

Here, additional variables are introduced to better represent the process of local model update.
Let $f_{i}(\phi_{i}):{\mathbb{R}}^{d_{x}}\rightarrow{\mathbb{R}}^{d_{c}}$ be the embedding function of the $i$ -th client, which can be different regarding to different clients. $d_{x}$ and $d_{c}$ represent the dimension of the input $x$ and the prototype, respectively. They should be the same for all clients. $g_{i}(\nu_{i}):{\mathbb{R}}^{d_{c}}\rightarrow{\mathbb{R}}^{d_{y}}$ is the decision function for all clients, in which $d_{y}$ represents the dimension of output $y$ . So the labelling function can be written as $\mathcal{F}_{i}(\phi_{i},\nu_{i})=g_{i}(\nu_{i})\circ f_{i}(\phi_{i})$ , and sometimes we use $\omega_{i}$ to represent $(\phi_{i},\nu_{i})$ for short.
In the theoretical analysis, we omit the label $(j)$ of prototype $C^{(j)}$ for convenience, which does not affect the proof. We also use $q_{i}$ to represent the weight of the prototype for $i$ -th client, and $p_{i}$ to represent the weight of the loss function for the $i$ -th client for short.

$f_{i}(\phi_{i}):{\mathbb{R}}^{d_{x}}\rightarrow{\mathbb{R}}^{d_{c}}$$i$$d_{x}$$d_{c}$$x$$g_{i}(\nu_{i}):{\mathbb{R}}^{d_{c}}\rightarrow{\mathbb{R}}^{d_{y}}$$d_{y}$$y$$\mathcal{F}_{i}(\phi_{i},\nu_{i})=g_{i}(\nu_{i})\circ f_{i}(\phi_{i})$$\omega_{i}$$(\phi_{i},\nu_{i})$$(j)$$C^{(j)}$$q_{i}$$i$$p_{i}$$i$Therefore, the local loss function of client $i$ can be written as:

$i$
$$\mathcal{L}(\phi_{i},\nu_{i};x,y)=\mathcal{L}_{S}(\mathcal{F}_{i}(\phi_{i},\nu_{i};x),y)+\lambda\|f_{i}(\phi_{i};x)-\bar{C}\|_{2}^{2},$$

$$\mathcal{L}(\phi_{i},\nu_{i};x,y)=\mathcal{L}_{S}(\mathcal{F}_{i}(\phi_{i},\nu_{i};x),y)+\lambda\|f_{i}(\phi_{i};x)-\bar{C}\|_{2}^{2},$$
in which the global prototype


$$\bar{C}=\sum_{i=1}^{m}q_{i}C_{i}$$

$$\bar{C}=\sum_{i=1}^{m}q_{i}C_{i}$$
with


$$\sum_{i=1}^{m}q_{i}=\sum_{i=1}^{m}\frac{|D_{i}|}{N}=1$$

$$\sum_{i=1}^{m}q_{i}=\sum_{i=1}^{m}\frac{|D_{i}|}{N}=1$$
and


$$C_{i}=\frac{1}{|D_{i}|}\sum_{(x,y)\in D_{i}}f_{i}(\phi_{i};x),$$

$$C_{i}=\frac{1}{|D_{i}|}\sum_{(x,y)\in D_{i}}f_{i}(\phi_{i};x),$$
and it is a constant in $\mathcal{L}$ , changing $\mathcal{L}$ every communication round, which makes the convergence analysis complex.

$\mathcal{L}$$\mathcal{L}$As for the iteration notation system, we use $t$ to represent the communication round, $e\in\{1/2,1,2,\dots,E$ } to represent the local iterations. There are $E$ local iterations in total, so $tE+e$ refers to the $e$ -th local iteration in the communication round $t+1$ . Moreover, $tE$ represents the time step before prototype aggregation at the server, and $tE+1/2$ represents the time step between prototype aggregation at the server and starting the first iteration on the local model.

$t$$e\in\{1/2,1,2,\dots,E$$E$$tE+e$$e$$t+1$$tE$$tE+1/2$### Assumptions

(Lipschitz Smooth).

$L_{1}$$L_{1}$$\displaystyle\|\nabla\mathcal{L}_{{t_{1}}}$$\displaystyle-\nabla\mathcal{L}_{{t_{2}}}\|_{2}\leq L_{1}\|\omega_{{i,t_{1}}}-\omega_{{i,t_{2}}}\|_{2},$$\displaystyle\forall t_{1},t_{2}>0,i\in\{1,2,\dots,m\},$$\displaystyle\mathcal{L}_{{t_{1}}}-\mathcal{L}_{{t_{2}}}$$\displaystyle\leq\langle\nabla\mathcal{L}_{{t_{2}}},(\omega_{{i,t_{1}}}-\omega_{{i,t_{2}}})\rangle+\frac{L_{1}}{2}\|\omega_{{i,t_{1}}}-\omega_{{i,t_{2}}}\|_{2}^{2},\quad\forall t_{1},t_{2}>0,i\in\{1,2,\dots,m\}.$(Unbiased Gradient and Bounded Variance).

$g_{i,t}=\nabla\mathcal{L}(\omega_{i,t},\xi_{t})$$\displaystyle{\mathbb{E}}_{\xi_{i}\sim D_{i}}{[}g_{i,t}{]}=\nabla\mathcal{L}(\omega_{i,t})=\nabla\mathcal{L}_{t},\quad\forall i\in{1,2,\dots,m},$$\sigma^{2}$$\displaystyle{\mathbb{E}}{[}{\|g_{i,t}-\nabla\mathcal{L}(\omega_{i,t})\|}_{2}^{2}{]}\leq\sigma^{2},\quad\forall i\in\{1,2,\dots,m\},\sigma^{2}\geq 0.$(Bounded Expectation of Euclidean norm of Stochastic Gradients).

$G$
$${\mathbb{E}}{[}\|g_{i,t}\|_{2}{]}\leq G,\quad\forall i\in\{1,2,\dots,m\}.$$

$${\mathbb{E}}{[}\|g_{i,t}\|_{2}{]}\leq G,\quad\forall i\in\{1,2,\dots,m\}.$$
(Lipschitz Continuity).

$L_{2}$
$$\left\|f_{i}(\phi_{i,t_{1}})-f_{i}(\phi_{i,t_{2}})\right\|\leq L_{2}\|\phi_{i,t_{1}}-\phi_{i,t_{2}}\|_{2},\quad\forall t_{1},t_{2}>0,i\in\{1,2,\dots,m\}.$$

$$\left\|f_{i}(\phi_{i,t_{1}})-f_{i}(\phi_{i,t_{2}})\right\|\leq L_{2}\|\phi_{i,t_{1}}-\phi_{i,t_{2}}\|_{2},\quad\forall t_{1},t_{2}>0,i\in\{1,2,\dots,m\}.$$
Assumption is a little strong, but we only use it in a very narrow domain with width of $E$ steps of SGD in Lemma .

$E$### Key Lemmas

$t+1$
$$\begin{split}{\mathbb{E}}{[}\mathcal{L}_{(t+1)E}{]}\leq\mathcal{L}_{tE+1/2}-(\eta-\frac{L_{1}\eta^{2}}{2})\sum_{e=1/2}^{E-1}\|\nabla\mathcal{L}_{tE+e}\|_{2}^{2}+\frac{L_{1}E\eta^{2}}{2}\sigma^{2}.\end{split}$$

$$\begin{split}{\mathbb{E}}{[}\mathcal{L}_{(t+1)E}{]}\leq\mathcal{L}_{tE+1/2}-(\eta-\frac{L_{1}\eta^{2}}{2})\sum_{e=1/2}^{E-1}\|\nabla\mathcal{L}_{tE+e}\|_{2}^{2}+\frac{L_{1}E\eta^{2}}{2}\sigma^{2}.\end{split}$$
Due to the fact that this lemma is for an arbitrary client, so client notation $i$ is omitted. Let $\omega_{t+1}=\omega_{t}-\eta g_{t}$ , then

$i$$\omega_{t+1}=\omega_{t}-\eta g_{t}$
$$\begin{split}\mathcal{L}_{tE+1}&\stackrel{{\scriptstyle(a)}}{{\leq}}\mathcal{L}_{tE+1/2}+\langle\nabla\mathcal{L}_{tE+1/2},(\omega_{tE+1}-\omega_{tE+1/2})\rangle+\frac{L_{1}}{2}\|\omega_{tE+1}-\omega_{tE+1/2}\|_{2}^{2}\\
&=\mathcal{L}_{tE+1/2}-\eta\langle\nabla\mathcal{L_{1}}_{tE+1/2},g_{tE+1/2}\rangle+\frac{L_{1}}{2}\|\eta g_{tE+1/2}\|_{2}^{2},\end{split}$$

$$\begin{split}\mathcal{L}_{tE+1}&\stackrel{{\scriptstyle(a)}}{{\leq}}\mathcal{L}_{tE+1/2}+\langle\nabla\mathcal{L}_{tE+1/2},(\omega_{tE+1}-\omega_{tE+1/2})\rangle+\frac{L_{1}}{2}\|\omega_{tE+1}-\omega_{tE+1/2}\|_{2}^{2}\\
&=\mathcal{L}_{tE+1/2}-\eta\langle\nabla\mathcal{L_{1}}_{tE+1/2},g_{tE+1/2}\rangle+\frac{L_{1}}{2}\|\eta g_{tE+1/2}\|_{2}^{2},\end{split}$$
where (a) follows from the quadratic $L_{1}$ -Lipschitz smooth bound in Assumption . Taking expectation of both sides of the above equation on the random variable $\xi_{tE+1/2}$ , we have

$L_{1}$$\xi_{tE+1/2}$$\displaystyle{\mathbb{E}}{[}\mathcal{L}_{tE+1}{]}$$\displaystyle\leq\mathcal{L}_{tE+1/2}-\eta{\mathbb{E}}{[}\langle\nabla\mathcal{L_{1}}_{tE+1/2},g_{tE+1/2}\rangle{]}+\frac{L_{1}\eta^{2}}{2}{\mathbb{E}}{[}\|g_{tE+1/2}\|_{2}^{2}{]}$$\displaystyle\stackrel{{\scriptstyle(b)}}{{=}}\mathcal{L}_{tE+1/2}-\eta\|\nabla\mathcal{L}_{tE+1/2}\|_{2}^{2}+\frac{L_{1}\eta^{2}}{2}{\mathbb{E}}{[}\|g_{i,tE+1/2}\|_{2}^{2}{]}$$\displaystyle\stackrel{{\scriptstyle(c)}}{{\leq}}\mathcal{L}_{tE+1/2}-\eta\|\nabla\mathcal{L}_{tE+1/2}\|_{2}^{2}+\frac{L_{1}\eta^{2}}{2}(\|\nabla\mathcal{L}_{tE+1/2}\|_{2}^{2}+Var(g_{i,tE+1/2}))$$\displaystyle=\mathcal{L}_{tE+1/2}-(\eta-\frac{L_{1}\eta^{2}}{2})\|\nabla\mathcal{L}_{tE+1/2}\|_{2}^{2}+\frac{L_{1}\eta^{2}}{2}Var(g_{i,tE+1/2})$$\displaystyle\stackrel{{\scriptstyle(d)}}{{\leq}}\mathcal{L}_{tE+1/2}-(\eta-\frac{L_{1}\eta^{2}}{2})\|\nabla\mathcal{L}_{tE+1/2}\|_{2}^{2}+\frac{L_{1}\eta^{2}}{2}\sigma^{2},$where (b) follows from Assumption , (c) follows from $Var(x)={\mathbb{E}}{[}x^{2}{]}-({\mathbb{E}{[}x{]}})^{2}$ , (d) follows from Assumption . Take expectation of $\omega$ on both sides. Then, by telescoping of $E$ steps, we have,

$Var(x)={\mathbb{E}}{[}x^{2}{]}-({\mathbb{E}{[}x{]}})^{2}$$\omega$$E$
$$\begin{split}{\mathbb{E}}{[}\mathcal{L}_{(t+1)E}{]}\leq\mathcal{L}_{tE+1/2}-(\eta-\frac{L_{1}\eta^{2}}{2})\sum_{e=1/2}^{E-1}\|\nabla\mathcal{L}_{tE+e}\|_{2}^{2}+\frac{L_{1}E\eta^{2}}{2}\sigma^{2}.\end{split}$$

$$\begin{split}{\mathbb{E}}{[}\mathcal{L}_{(t+1)E}{]}\leq\mathcal{L}_{tE+1/2}-(\eta-\frac{L_{1}\eta^{2}}{2})\sum_{e=1/2}^{E-1}\|\nabla\mathcal{L}_{tE+e}\|_{2}^{2}+\frac{L_{1}E\eta^{2}}{2}\sigma^{2}.\end{split}$$
∎


$${\mathbb{E}}{[}\mathcal{L}_{(t+1)E+1/2}{]}\leq\mathcal{L}_{(t+1)E}+\lambda L_{2}\eta EG$$

$${\mathbb{E}}{[}\mathcal{L}_{(t+1)E+1/2}{]}\leq\mathcal{L}_{(t+1)E}+\lambda L_{2}\eta EG$$
$\displaystyle\mathcal{L}_{(t+1)E+1/2}$$\displaystyle=\mathcal{L}_{(t+1)E}+\mathcal{L}_{(t+1)E+1/2}-\mathcal{L}_{(t+1)E}$$\displaystyle\stackrel{{\scriptstyle(a)}}{{=}}\mathcal{L}_{(t+1)E}+\lambda\|f_{i}(\phi_{i,(t+1)E})-\bar{C}_{t+2}\|_{2}-\lambda\|f_{i}(\phi_{i,(t+1)E})-\bar{C}_{t+1}\|_{2}$$\displaystyle\stackrel{{\scriptstyle(b)}}{{\leq}}\mathcal{L}_{(t+1)E}+\lambda\|\bar{C}_{t+2}-\bar{C}_{t+1}\|_{2}$$\displaystyle\stackrel{{\scriptstyle(c)}}{{=}}\mathcal{L}_{(t+1)E}+\lambda\|\sum_{i=1}^{m}q_{i}C_{i,(t+1)E}-\sum_{i=1}^{m}q_{i}C_{i,tE}\|_{2}$$\displaystyle=\mathcal{L}_{(t+1)E}+\lambda\|\sum_{i=1}^{m}q_{i}(C_{i,(t+1)E}-C_{i,tE})\|_{2}$$\displaystyle\stackrel{{\scriptstyle(d)}}{{=}}\mathcal{L}_{(t+1)E}+\lambda\|\sum_{i=1}^{m}q_{i}\frac{1}{|D_{i}|}\sum_{k=1}^{|D_{i}|}(f_{i}(\phi_{i,(t+1)E};x_{i,k})-f_{i}(\phi_{i,tE};x_{i,k})\|_{2}$$\displaystyle\stackrel{{\scriptstyle(e)}}{{\leq}}\mathcal{L}_{(t+1)E}+\lambda\sum_{i=1}^{m}\frac{q_{i}}{|D_{i}|}\sum_{k=1}^{|D_{i}|}\|f_{i}(\phi_{i,(t+1)E};x_{i,k})-f_{i}(\phi_{i,tE};x_{i,k})\|_{2}$$\displaystyle\stackrel{{\scriptstyle(f)}}{{\leq}}\mathcal{L}_{(t+1)E}+\lambda L_{2}\sum_{i=1}^{m}{q_{i}}\|\phi_{i,(t+1)E}-\phi_{i,tE}\|_{2}$$\displaystyle\stackrel{{\scriptstyle(g)}}{{\leq}}\mathcal{L}_{(t+1)E}+\lambda L_{2}\sum_{i=1}^{m}{q_{i}}\|\omega_{i,(t+1)E}-\omega_{i,tE}\|_{2}$$\displaystyle=\mathcal{L}_{(t+1)E}+\lambda L_{2}\eta\sum_{i=1}^{m}q_{i}\|\sum_{e=1/2}^{E-1}g_{i,tE+e}\|_{2}$$\displaystyle\stackrel{{\scriptstyle(h)}}{{\leq}}\mathcal{L}_{(t+1)E}+\lambda L_{2}\eta\sum_{i=1}^{m}q_{i}\sum_{e=1/2}^{E-1}\|g_{i,tE+e}\|_{2}$Take expectations of random variable $\xi$ on both sides, then

$\xi$$\displaystyle{\mathbb{E}}{[}\mathcal{L}_{(t+1)E+1/2}{]}$$\displaystyle{\leq}\mathcal{L}_{(t+1)E}+\lambda L_{2}\eta\sum_{i=1}^{m}q_{i}\sum_{e=1/2}^{E-1}{\mathbb{E}}{[}\|g_{i,tE+e}\|_{2}{]}$$\displaystyle\stackrel{{\scriptstyle(i)}}{{\leq}}\mathcal{L}_{(t+1)E}+\lambda L_{2}\eta EG,$where (a) follows from the definition of local loss function in Eq. , (b) follows from $\|a-b\|_{2}-\|a-c\|_{2}\leq\|b-c\|_{2}$ , (c) follows from the definition of global prototype in Eq. , (d) follows from the definition of local prototype in Eq. , (e) and (h) follow from $\|\sum a_{i}\|_{2}\leq\sum{\|a_{i}\|_{2}}$ , (f) follows from $L_{2}$ -Lipschitz continuity in Assumption , (g) follows from the fact that $\phi_{i}$ is a subset of $\omega_{i}$ , (i) follows from Assumption .
∎

$\|a-b\|_{2}-\|a-c\|_{2}\leq\|b-c\|_{2}$$\|\sum a_{i}\|_{2}\leq\sum{\|a_{i}\|_{2}}$$L_{2}$$\phi_{i}$$\omega_{i}$### Theorems

(One-round deviation).


$$\begin{split}{\mathbb{E}}[\mathcal{L}_{(t+1)E+1/2}]\leq\mathcal{L}_{tE+1/2}-(\eta-\frac{L_{1}\eta^{2}}{2})\sum_{e=1/2}^{E-1}\|\nabla\mathcal{L}_{tE+e}\|_{2}^{2}+\frac{L_{1}E\eta^{2}}{2}\sigma^{2}+\lambda L_{2}\eta EG.\end{split}$$

$$\begin{split}{\mathbb{E}}[\mathcal{L}_{(t+1)E+1/2}]\leq\mathcal{L}_{tE+1/2}-(\eta-\frac{L_{1}\eta^{2}}{2})\sum_{e=1/2}^{E-1}\|\nabla\mathcal{L}_{tE+e}\|_{2}^{2}+\frac{L_{1}E\eta^{2}}{2}\sigma^{2}+\lambda L_{2}\eta EG.\end{split}$$
(Non-convex convergence).

$\mathcal{L}$
$$\eta_{e^{\prime}}<\frac{2(\sum_{e=1/2}^{e^{\prime}}\|\nabla\mathcal{L}_{tE+e}\|_{2}^{2}-\lambda L_{2}EG)}{L_{1}(\sum_{e=1/2}^{e^{\prime}}\|\nabla\mathcal{L}_{tE+e}\|_{2}^{2}+E\sigma^{2})},\ e^{\prime}=1/2,1,\dots,E-1$$

$$\eta_{e^{\prime}}<\frac{2(\sum_{e=1/2}^{e^{\prime}}\|\nabla\mathcal{L}_{tE+e}\|_{2}^{2}-\lambda L_{2}EG)}{L_{1}(\sum_{e=1/2}^{e^{\prime}}\|\nabla\mathcal{L}_{tE+e}\|_{2}^{2}+E\sigma^{2})},\ e^{\prime}=1/2,1,\dots,E-1$$

$$\lambda_{t}<\frac{\|\nabla\mathcal{L}_{tE+1/2}\|_{2}^{2}}{L_{2}EG}.$$

$$\lambda_{t}<\frac{\|\nabla\mathcal{L}_{tE+1/2}\|_{2}^{2}}{L_{2}EG}.$$
(Non-convex convergence rate of )

$\Delta=\mathcal{L}_{0}-\mathcal{L}^{*}$$\epsilon>0$
$$T=\frac{2\Delta}{E\epsilon(2\eta-L_{1}\eta^{2})-E\eta(L_{1}\eta\sigma^{2}+2\lambda L_{2}G)}$$

$$T=\frac{2\Delta}{E\epsilon(2\eta-L_{1}\eta^{2})-E\eta(L_{1}\eta\sigma^{2}+2\lambda L_{2}G)}$$

$$\frac{1}{TE}\sum_{t=0}^{T-1}\sum_{e=1/2}^{E-1}{\mathbb{E}}[\|\nabla\mathcal{L}_{tE+e}\|_{2}^{2}]<\epsilon,$$

$$\frac{1}{TE}\sum_{t=0}^{T-1}\sum_{e=1/2}^{E-1}{\mathbb{E}}[\|\nabla\mathcal{L}_{tE+e}\|_{2}^{2}]<\epsilon,$$

$$\eta<\frac{2(\epsilon-\lambda L_{2}G)}{L_{1}(\epsilon+\sigma^{2})},$$

$$\eta<\frac{2(\epsilon-\lambda L_{2}G)}{L_{1}(\epsilon+\sigma^{2})},$$

$$\lambda<\frac{\epsilon}{L_{2}G}.$$

$$\lambda<\frac{\epsilon}{L_{2}G}.$$
### Completing the Proof of Theorem 1 and Corollary 1

Taking expectation of $\omega$ on both sides in Lemma and , then sum them, we can easily get

$\omega$
$$\begin{split}{\mathbb{E}}[\mathcal{L}_{(t+1)E+1/2}]\leq\mathcal{L}_{tE+1/2}-(\eta-\frac{L_{1}\eta^{2}}{2})\sum_{e=1/2}^{E-1}\|\nabla\mathcal{L}_{tE+e}\|_{2}^{2}+\frac{L_{1}E\eta^{2}}{2}\sigma^{2}+\lambda L_{2}\eta EG\end{split}$$

$$\begin{split}{\mathbb{E}}[\mathcal{L}_{(t+1)E+1/2}]\leq\mathcal{L}_{tE+1/2}-(\eta-\frac{L_{1}\eta^{2}}{2})\sum_{e=1/2}^{E-1}\|\nabla\mathcal{L}_{tE+e}\|_{2}^{2}+\frac{L_{1}E\eta^{2}}{2}\sigma^{2}+\lambda L_{2}\eta EG\end{split}$$
Then, to make sure $-(\eta-\frac{L_{1}\eta^{2}}{2})\sum_{e=1/2}^{E-1}\|\nabla\mathcal{L}_{tE+e}\|_{2}^{2}+\frac{L_{1}E\eta^{2}}{2}\sigma^{2}+\lambda L_{2}\eta EG\leq 0$ , we get

$-(\eta-\frac{L_{1}\eta^{2}}{2})\sum_{e=1/2}^{E-1}\|\nabla\mathcal{L}_{tE+e}\|_{2}^{2}+\frac{L_{1}E\eta^{2}}{2}\sigma^{2}+\lambda L_{2}\eta EG\leq 0$
$$\eta<\frac{2(\sum_{e=1/2}^{E-1}\|\nabla\mathcal{L}_{tE+e}\|_{2}^{2}-\lambda L_{2}EG)}{L_{1}(\sum_{e=1/2}^{E-1}\|\nabla\mathcal{L}_{tE+e}\|_{2}^{2}+E\sigma^{2})},$$

$$\eta<\frac{2(\sum_{e=1/2}^{E-1}\|\nabla\mathcal{L}_{tE+e}\|_{2}^{2}-\lambda L_{2}EG)}{L_{1}(\sum_{e=1/2}^{E-1}\|\nabla\mathcal{L}_{tE+e}\|_{2}^{2}+E\sigma^{2})},$$
and


$$\lambda<\frac{\sum_{e=1/2}^{E-1}\|\nabla\mathcal{L}_{tE+e}\|_{2}^{2}}{L_{2}EG}.$$

$$\lambda<\frac{\sum_{e=1/2}^{E-1}\|\nabla\mathcal{L}_{tE+e}\|_{2}^{2}}{L_{2}EG}.$$
In practice, we use


$$\eta_{e^{\prime}}<\frac{2(\sum_{e=1/2}^{e^{\prime}}\|\nabla\mathcal{L}_{tE+e}\|_{2}^{2}-\lambda L_{2}EG)}{L_{1}(\sum_{e=1/2}^{e^{\prime}}\|\nabla\mathcal{L}_{tE+e}\|_{2}^{2}+E\sigma^{2})},\ e^{\prime}=1/2,1,\dots,E-1$$

$$\eta_{e^{\prime}}<\frac{2(\sum_{e=1/2}^{e^{\prime}}\|\nabla\mathcal{L}_{tE+e}\|_{2}^{2}-\lambda L_{2}EG)}{L_{1}(\sum_{e=1/2}^{e^{\prime}}\|\nabla\mathcal{L}_{tE+e}\|_{2}^{2}+E\sigma^{2})},\ e^{\prime}=1/2,1,\dots,E-1$$
and


$$\lambda_{t}<\frac{\|\nabla\mathcal{L}_{tE+1/2}\|_{2}^{2}}{L_{2}EG}.$$

$$\lambda_{t}<\frac{\|\nabla\mathcal{L}_{tE+1/2}\|_{2}^{2}}{L_{2}EG}.$$
So, the convergence of $\mathcal{L}$ holds.
∎

$\mathcal{L}$### Completing the Proof of Theorem 2

Take expectation of $\omega$ on both sides in Eq. , then telescope considering the communication round from $t=0$ to $t=T-1$ with the timestep from $e=1/2$ to $t=E$ in each communication round, we have

$\omega$$t=0$$t=T-1$$e=1/2$$t=E$$\displaystyle\frac{1}{TE}\sum_{t=0}^{T-1}\sum_{e=1/2}^{E-1}{\mathbb{E}}[\|\nabla\mathcal{L}_{tE+e}\|_{2}^{2}]$$\displaystyle\leq\frac{\frac{1}{TE}\sum_{t=0}^{T-1}(\mathcal{L}_{tE+1/2}-{\mathbb{E}}[\mathcal{L}_{(t+1)E+1/2}])+\frac{L_{1}\eta^{2}}{2}\sigma^{2}+\lambda L_{2}\eta G}{\eta-\frac{L_{1}\eta^{2}}{2}}.$Given any $\epsilon>0$ , let

$\epsilon>0$
$$\frac{\frac{1}{TE}\sum_{t=0}^{T-1}(\mathcal{L}_{tE+1/2}-{\mathbb{E}}[\mathcal{L}_{(t+1)E+1/2}])+\frac{L_{1}\eta^{2}}{2}\sigma^{2}+\lambda L_{2}\eta G}{\eta-\frac{L_{1}\eta^{2}}{2}}<\epsilon,$$

$$\frac{\frac{1}{TE}\sum_{t=0}^{T-1}(\mathcal{L}_{tE+1/2}-{\mathbb{E}}[\mathcal{L}_{(t+1)E+1/2}])+\frac{L_{1}\eta^{2}}{2}\sigma^{2}+\lambda L_{2}\eta G}{\eta-\frac{L_{1}\eta^{2}}{2}}<\epsilon,$$
that is


$$\frac{\frac{2}{TE}\sum_{t=0}^{T-1}(\mathcal{L}_{tE+1/2}-{\mathbb{E}}[\mathcal{L}_{(t+1)E+1/2}])+{L_{1}\eta^{2}}\sigma^{2}+2\lambda L_{2}\eta G}{2\eta-{L_{1}\eta^{2}}}<\epsilon.$$

$$\frac{\frac{2}{TE}\sum_{t=0}^{T-1}(\mathcal{L}_{tE+1/2}-{\mathbb{E}}[\mathcal{L}_{(t+1)E+1/2}])+{L_{1}\eta^{2}}\sigma^{2}+2\lambda L_{2}\eta G}{2\eta-{L_{1}\eta^{2}}}<\epsilon.$$
Let $\Delta=\mathcal{L}_{0}-\mathcal{L}^{*}$ . Since $\sum_{t=0}^{T-1}(\mathcal{L}_{tE+1/2}-{\mathbb{E}}[\mathcal{L}_{(t+1)E+1/2}])\leq\Delta$ , the above equation holds when

$\Delta=\mathcal{L}_{0}-\mathcal{L}^{*}$$\sum_{t=0}^{T-1}(\mathcal{L}_{tE+1/2}-{\mathbb{E}}[\mathcal{L}_{(t+1)E+1/2}])\leq\Delta$$\displaystyle\frac{\frac{2\Delta}{TE}+{L_{1}\eta^{2}}\sigma^{2}+2\lambda L_{2}\eta G}{2\eta-{L_{1}\eta^{2}}}<\epsilon,$that is

$\displaystyle T>\frac{2\Delta}{E\epsilon(2\eta-L_{1}\eta^{2})-E\eta(L_{1}\eta\sigma^{2}+2\lambda L_{2}G)}.$So, we have


$$\frac{1}{TE}\sum_{t=0}^{T-1}\sum_{e=1/2}^{E-1}{\mathbb{E}}[\|\nabla\mathcal{L}_{tE+e}\|_{2}^{2}]<\epsilon,$$

$$\frac{1}{TE}\sum_{t=0}^{T-1}\sum_{e=1/2}^{E-1}{\mathbb{E}}[\|\nabla\mathcal{L}_{tE+e}\|_{2}^{2}]<\epsilon,$$
when


$$\eta<\frac{2(\epsilon-\lambda L_{2}G)}{L_{1}(\epsilon+\sigma^{2})},$$

$$\eta<\frac{2(\epsilon-\lambda L_{2}G)}{L_{1}(\epsilon+\sigma^{2})},$$
and


$$\lambda<\frac{\epsilon}{L_{2}G}.$$

$$\lambda<\frac{\epsilon}{L_{2}G}.$$
∎


![](/assets/ar5iv.png)


![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)

