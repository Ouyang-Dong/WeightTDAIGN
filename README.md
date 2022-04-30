# WeightTDAIGN
Many studies have indicated miRNAs lead to the occurrence and development of diseases through a variety of underlying mechanisms. Meanwhile, computational models can save time, minimize cost and discover potential associations on a large scale. However, most existing computational models based on matrix or tensor decomposition cannot recover positive samples well. Moreover, the high noise of biological similarity networks and how to preserve these similarity relationships in low-dimensional space are also challenges. To this end, we propose a novel computational framework, called WeightTDAIGN, to identify potential multiple types of miRNA-disease associations. WeightTDAIGN can recover positive samples well and improve prediction performance by weighting positive samples. WeightTDAIGN integrates more auxiliary information related to miRNAs and diseases into the tensor decomposition framework, focus on learning low-rank tensor space, and constrain projection matrices by using L2,1 norm to reduce the impact of redundant information on the model. In addition, WeightTDAIGN can preserve the local structure information in the biological similarity network by introducing graph Laplacian regularization. Our experimental results show that the more sparse datasets, the more satisfactory performance of WeightTDAIGN can be obtained. And case studies results further illustrate that WeightTDAIGN can accurately predict the associations of miRNA-disease-type.
# The workflow of WeightTDAIGN model
![The workflow of WeightTDAIGN model](https://github.com/Ouyang-Dong/WeightTDAIGN/blob/master/idea.png)
# Environment Requirement
The code has been tested running under Python 3.8.5. The required packages are as follows:
1. tensorly == 0.5.1
2. numpy == 1.21.5
3. pandas == 1.3.5
# Notice
Before running the code, you need to change the path of reading data in the code to the path suitable for your computer.

