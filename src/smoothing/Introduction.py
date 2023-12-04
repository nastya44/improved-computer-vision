import streamlit as st


st.set_page_config(
    page_title="Gradient-free deep learning: Project introduction",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)


st.title("Gradient-free deep learning")
st.markdown(
    """
[![Source Code](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/antonAce/improved-computer-vision)

The project, under the "Applied Modeling" course, integrates emerging trends in optimization theory with pertinent challenges in deep learning. The objective is 
to implement a modified gradient descent algorithm, as proposed in the [paper](https://arxiv.org/abs/2308.08422), within the widely-used deep learning framework, 
[PyTorch](https://pytorch.org/). This implementation will be applied to address the object detection problem in [Swin Transformers](https://browse.arxiv.org/pdf/2103.14030.pdf), 
trained on the [COCO dataset](https://www.v7labs.com/blog/coco-dataset-guide).
"""
)
st.video("./assets/yolo_demo_crop.mp4")
st.markdown(
    "*Object detection problem demo video from [Ultralytics](https://s3.amazonaws.com/media-p.slid.es/videos/1963163/Y29RCFLg/yolo_demo_crop.mp4)*"
)

st.header("A project objective")
st.markdown(
    """
 1. Examine the most widely used gradient optimization algorithms;
 2. Develop and apply a gradient-free optimization algorithm for **Backpropagation**;
 3. Train and assess the deep learning model for object segmentation: [Swin Transformers](https://browse.arxiv.org/pdf/2103.14030.pdf).
"""
)

st.header("Motivation")
st.markdown(
    """
 - The problem of image segmentation is widely recognized in the field of automation.
 - One of the most significant applications of this technology in Ukraine at present is in the area of **military technology**: specifically, automatic targeting for unmanned aerial vehicles (UAVs) or artillery systems.
"""
)

st.header("References")
st.markdown(
    """
1. Ermoliev, Y. M., & Norkin, V. I. (2003). Solution of Nonconvex Nonsmooth Stochastic Optimization Problems. *Cybernetics and Systems Analysis, 39(5),* 701–715. [https://doi.org/10.1023/B:CASA.0000012091.84864.65](https://doi.org/10.1023/B:CASA.0000012091.84864.65)
2. Newton, D., Yousefian, F., & Pasupathy, R. (2018). Stochastic Gradient Descent: Recent Trends. In *Recent Advances in Optimization and Modeling of Contemporary Problems* (pp. 193–220). INFORMS. [https://doi.org/10.1287/educ.2018.0191](https://doi.org/10.1287/educ.2018.0191)
3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Available at [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/)
4. Rosenblatt, F. (1958). The perceptron: A probabilistic model for information storage and organization in the brain. *Psychological Review, 65(6),* 386–408. [https://doi.org/10.1037/h0042519](https://doi.org/10.1037/h0042519)
5. International Workshop on Artificial Neural Networks (1995 : Torremolinos). (1995). *From natural to artificial neural computation : International Workshop on Artificial Neural Networks, Malaga-Torremolinos, Spain, June 7-9, 1995 : proceedings.* Berlin ; New York : Springer-Verlag. Accessed November 14, 2023. Available at [http://archive.org/details/fromnaturaltoart1995inte](http://archive.org/details/fromnaturaltoart1995inte)
6. (1998). *On-line learning in neural networks.* Cambridge [England] ; New York : Cambridge University Press. Accessed November 14, 2023. Available at [http://archive.org/details/onlinelearningin0000unse](http://archive.org/details/onlinelearningin0000unse)
7. Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive Subgradient Methods for Online Learning and Stochastic Optimization. *JMLR, 2121–2159.*
8. Kingma, D. P., & Ba, J. (2017). Adam: A Method for Stochastic Optimization. arXiv. Accessed November 14, 2023. Available at [http://arxiv.org/abs/1412.6980](http://arxiv.org/abs/1412.6980)
9. Bayandina, A. S., Gasnikov, A. V., & Lagunovskaya, A. A. (2018). Gradient-Free Two-Point Methods for Solving Stochastic Nonsmooth Convex Optimization Problems with Small Non-Random Noises. *Autom Remote Control, 79(8),* 1399–1408. [https://doi.org/10.1134/S0005117918080039](https://doi.org/10.1134/S0005117918080039)
10. Chagas, J. Q., Diehl, N. M. L., & Guidolin, P. L. (2017). Some properties for the Steklov averages. arXiv. [https://doi.org/10.48550/arXiv.1707.06368](https://doi.org/10.48550/arXiv.1707.06368)
11. Duchi, J. C., Jordan, M. I., Wainwright, M. J., & Wibisono, A. (2015). Optimal Rates for Zero-Order Convex Optimization: The Power of Two Function Evaluations. *IEEE Transactions on Information Theory, 61(5),* 2788–2806. [https://doi.org/10.1109/TIT.2015.2409256](https://doi.org/10.1109/TIT.2015.2409256)
"""
)

