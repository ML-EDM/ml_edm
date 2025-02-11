# ``ml_edm`` : A python package for Early Decision Making-based Machine Learning 

Many situations require decisions to be made quickly to avoid the costs associated with delaying the decision. A doctor who needs to choose which test to perform on their patient and an agent considering whether a certain behavior on a network is caused by a hacker are examples of individuals confronted with such situations. However, taking a decision too hastily may lead to more mistakes, resulting in additional costs that could have been avoided.  In these situations, where there is a trade-off between the *earliness* of the decision and the *accuracy* of the prediction, the **Machine Learning for Early Decision Making** (ML-EDM) framework offers ML solutions not only to make a prediction but also to decide when to trigger its associated decision timely.

The ``ml_edm`` package provides tools to facilitate dealing with the **Early Classification of Time Series** (ECTS) problem, whose goal is to determine the class associated with a time series before all measurements are available.

## Install 

```console
pip install git+https://github.com/ML-EDM/ml_edm
```

For more information about the ML-EDM research domain, including latest publications, please have a look at the ML-EDM research GitHub: https://github.com/ML-EDM/ML-EDM.

## Citation

If you use the package, please refer to it using the following the bibtex entry: 

    @misc{renault2024mledmpackagepythontoolkit,
        title={ml_edm package: a Python toolkit for Machine Learning based Early Decision Making}, 
        author={Aurélien Renault and Youssef Achenchabe and Édouard Bertrand and Alexis Bondu and Antoine Cornuéjols and Vincent Lemaire and Asma Dachraoui},
        year={2024},
        eprint={2408.12925},
        archivePrefix={arXiv},
        primaryClass={cs.LG},
        url={https://arxiv.org/abs/2408.12925}, 
    }