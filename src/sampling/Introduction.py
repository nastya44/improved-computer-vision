import streamlit as st


st.set_page_config(
    page_title="Gradient sampling deep learning: Project introduction",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)


st.title("Gradient sampling deep learning")
st.markdown(
    """
[![Source Code](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/antonAce/improved-computer-vision)

The project, under the "Applied Modeling" course, integrates emerging trends in optimization theory with pertinent challenges in deep learning. The objective is 
to implement a modified gradient descent algorithm, as described in the [paper](https://epubs.siam.org/doi/10.1137/030601296), within the widely-used deep learning framework, 
[PyTorch](https://pytorch.org/). This implementation will be applied to address the image classification problem in [Convolutional Neural Networks](https://www.researchgate.net/publication/2453996_Convolutional_Networks_for_Images_Speech_and_Time-Series), 
trained on the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).
"""
)
st.video("./assets/yolo_demo_crop.mp4")
st.caption(
    "*Object detection problem demo video from [Ultralytics](https://s3.amazonaws.com/media-p.slid.es/videos/1963163/Y29RCFLg/yolo_demo_crop.mp4)*"
)

st.header("A project objective")
st.markdown(
    """
 1. Examine the most widely used gradient optimization algorithms;
 2. Develop and apply a sampling gradient optimization algorithm for **Backpropagation**;
 3. Train and assess the deep learning model for the image classification problem: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).
"""
)

st.header("Motivation")
st.markdown(
    """
 - The problem of image segmentation is widely recognized in the field of automation;
 - One of the most significant applications of this technology in Ukraine at present is in the area of **military technology**: 
specifically, automatic targeting for unmanned aerial vehicles (UAVs) or artillery systems;
 - Gradient algorithms in deep learning offer faster convergence and more efficient computation compared to zero-order optimization methods, 
particularly in high-dimensional spaces.
 - Gradient sampling method maintain these advantages, but contrast, the gradient encounters difficulties in navigating nonsmooth functions.
"""
)

st.header("References")
st.markdown(
    """
1. Bolte, J., Le, T., Pauwels, E., & Silveti-Falls, T. (2021). Nonsmooth implicit differentiation for machine-learning and optimization. Advances in neural information processing systems, 34, 13537-13549.
2. Qingchao Zhang, Xiaojing Ye, Yunmei Chen, "Nonsmooth nonconvex LDCT image reconstruction via learned descent algorithm," Proc. SPIE 11840, Developments in X-Ray Tomography XIII, 1184013 (9 September 2021); https://doi.org/10.1117/12.2597798
3. Shor N.Z. Minimization Methods for Non-Differentiable Functions. Kiev. Naukova Dumka. 1979. 200 p. (In Russian)
4. Norkin V., Kozyriev A. On Shor's r-Algorithm for Problems with Constraints. Cybernetics and Computer Technologies. 2023. 3. P. 16–22.
5. Gilmore, Paul C; Gomory, Ralph E (1961). "A linear programming approach to the cutting stock problem". Operations Research. 9 (6): 849–859. doi:10.1287/opre.9.6.849.
6. Bagirov, A., Karmitsa, N., Mäkelä, M.M. (2014). Bundle Methods. In: Introduction to Nonsmooth Optimization. Springer, Cham. https://doi.org/10.1007/978-3-319-08114-4_12
7. Ma, Tengyu and Andrew Ng. “CS229 Lecture notes.” (2007). https://api.semanticscholar.org/CorpusID:628573
8. Goodfellow, I., Bengio, Y.,, Courville, A. (2016). Deep Learning. MIT Press. ISBN: 9780262035613
9. Krizhevsky, A., Nair, V., & Hinton, G. (n.d.). CIFAR-10 (Canadian Institute for Advanced Research). http://www.cs.toronto.edu/~kriz/cifar.html
10. Poliak, B. T. (1987). The Heavy Ball Method. In Introduction to optimization (pp. 65–68). Optimization Software
11.Ermoliev, Y.: Stochastic Quasigradient Methods, pp. 3801–3807. Springer US, Boston, MA (2009)
12. Nesterov, Y. E. (1983). A method of solving a convex programming problem with convergence rate 0(1/k2). Doklady Akademii Nauk, 269, 543–547. Russian Academy of Sciences.
13. Burke, J.V., Curtis, F.E., Lewis, A.S., Overton, M.L., Simões, L.E.A. (2020). Gradient Sampling Methods for Nonsmooth Optimization. In: Bagirov, A., Gaudioso, M., Karmitsa,  Mäkelä, M., Taheri, S. (eds) Numerical Nonsmooth Optimization.
14. Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., … Chintala, S. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. In Advances in Neural Information Processing Systems 32 (pp. 8024–8035). Curran Associates, Inc. Retrieved from http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf
"""
)
