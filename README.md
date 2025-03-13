# LEPID
**Learning evolving prototypes for imbalanced data stream classification with limited labels**

The code is coming soon. All synthetic datasets in this article are implemented based on MOA(https://moa.cms.waikato.ac.nz/).


**Abstract**

Real-world data streams often exhibit long-tailed distributions with heavy class imbalance, posing great challenges for data stream classification, especially in the case of label scarcity and concept drift. Several active learning methods have been proposed to address this problem by selecting the most valuable instances for labeling. However, existing methods often struggle to dynamically identify the most valuable instances that truly represent the current concept while still requiring a large label budget. In this work, we propose a new algorithm, LEPID, to combine dynamic micro-cluster concept modeling and local entropy modeling to select current important concepts and prototypes. Specifically, we give greater weight to concept drift prototypes and minority prototypes to focus more on those regions that represent current concepts. We use a local entropy strategy based on micro-clusters to select the most valuable instances for labeling and reduce the label budget. Extensive experiments on real-world and synthetic imbalanced datasets show that, compared to state-of-the-art algorithms, our method can naturally adapt to concept drift and dynamically capture the current and most valuable prototypes to achieve better results even in the case of label scarcity.
