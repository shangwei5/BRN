Bilateral Recursive Network for Single Image Deraining
===============================================
Introduction
We in this paper propose a bilateral recurrent network (BRN) to simultaneously exploit the rain streak layer and the clean background image. Generally, we employ dual residual networks (ResNet) that are recursively unfolded to sequentially extract the rain streak layer (Fr) and predict the clean background image (Fx). In particular, we further propose bilateral LSTMs (BLSTM), which not only can respectively propagate deep features of the rain streak layer and the background image acorss stages, but also bring reciprocal communications between Fr and Fx. The experimental results demonstrate that our BRN notably outperforms state-of-the-art deep deraining
networks on synthetic datasets quantitatively and qualitatively. On real rainy images, our BRN also performs more favorably in generating visually plausible background images. 
