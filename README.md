# LEPID
# Learning evolving prototypes for imbalanced data stream classification with limited labels
This repository provides the source code, algorithms, experimental setup for LEPID. At the same time, we provide scripts to generate synthetic datasets. All synthetic datasets in this article are implemented based on MOA(https://moa.cms.waikato.ac.nz/).
# Abstract 
Real-world data streams often exhibit long-tailed distributions with heavy class imbalance, posing great challenges for data stream classification, especially in the case of label scarcity and concept drift. Several active learning methods have been proposed to address this problem by selecting the most valuable instances for labeling. However, existing methods often struggle to dynamically identify the most valuable instances that truly represent the current concept while still requiring a large label budget. In this work, we propose a new algorithm, LEPID, to combine dynamic micro-cluster concept modeling and local entropy modeling to select current important concepts and prototypes. Specifically, we give greater weight to concept drift prototypes and minority prototypes to focus more on those regions that represent current concepts. We use a local entropy strategy based on micro-clusters to select the most valuable instances for labeling and reduce the label budget. Extensive experiments on real-world and synthetic imbalanced datasets show that, compared to state-of-the-art algorithms, our method can naturally adapt to concept drift and dynamically capture the current and most valuable prototypes to achieve better results even in the case of label scarcity.

# Synthetic datasets
We use MOA to generate synthetic datasets. MOA tutorial can be found in this link: https://moa.cms.waikato.ac.nz/tutorial-5-simulating-concept-drift-in-moa/. 
1. Download MOA in https://moa.cms.waikato.ac.nz/downloads/
2. Open the lib folder under moa in a terminal in an integrated development environment such as eclipse.
3. Input the code of Generate synthetic data streams to generate synthetic data streams in arff file format.
4. The following code is used to synthesize the SEA_sudden dataset. The SEA_sudden dataset contains the following information: 50K instances | Sudden drift with imbalance ratio 5:1 → 1:5 | Sudden drift occurs at the 10,000th instances | The drift window is 1
```
java -cp moa.jar -javaagent:sizeofag-1.0.4.jar moa.DoTask WriteStreamToARFFFile -s (ConceptDriftStream -s (ImbalancedStream -s (generators.SEAGenerator -f 3) -c 0.8333;0.1667) -d (ImbalancedStream -s (generators.SEAGenerator -f 2) -c 0.1667;0.8333) -p 10000 -w 1) -f SEA_sudden.arff -m 50000
