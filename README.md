# PLocGAN
It's a generative-adversarial-network-based model called PLocGAN, which could generate protein fluorescence images with quantitative fraction annotation to alleviate the insufficiency of the quantitative fraction of protein expression.
# part1 Data preprocess
There are two data sources. One is the real dataset, which can be accessed by 'https://murphylab.cbd.cmu.edu/software/2010_PNAS_Unmixing/', and the other comes from the subcellular section in the Human Protein Atlas (HPA, https://proteinatlas.org). Run the codes step by step to get the single-cell images.
# part2 PLocGAN
The 'model.py' and 'network.py' are the base model without contrastive learning module, and the 'model_cl.py' and 'network_cl.py' have the contrastive learning module. Run .\train.py to train a generative model. And the model will be applied to the unmixing model (part3).
# part3 Unmixing model
The model is based on Bestfitting, can be obtained by 'https://github.com/CellProfiling/HPA-competition-solutions/tree/master/bestfitting'. Run .\run\train_gan.py to get a quantitative prediction model that introduces PLocGAN. Run.\run\train.py to get a baseline if you want to compare to the unmixing model with PLocGAN.
