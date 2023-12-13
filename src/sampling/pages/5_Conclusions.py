import streamlit as st


st.set_page_config(
    page_title="Gradient sampling deep learning: Conclusions",
    page_icon="📌",
    layout="centered",
    initial_sidebar_state="expanded",
)


st.title("Conclusions")

st.markdown(
    """ In this project, popular gradient descent methods were examined for model optimization. Each of the methods was modified using the Gradient Sampling technique, which allows preserving the advantages of conventional methods while performing better for solving non-smooth optimization problems.

    The considered methods included Stochastic Gradient Descent (SGD), Momentum, and Nesterov Accelerated Gradient (NAG). They were modified using the Gradient Sampling method, demonstrating improved convergence results compared to the standard, unmodified methods. Models were trained on the CIFAR-10 dataset, which consists of 32x32 images categorized into 10 classes. A drawback of this method was the lengthy model training due to the computation of gradients not only for model parameters but also for all samples of these parameters at each step. The number of samples needed for the method to function properly had to be high (>100).
    """) 

st.markdown(""" 
Advantages:

    """)

lst = ['better convergence compared to non modified methods', 'good for solving non-smooth optimization problems', 'can modify different popular methods']

s = ''

for i in lst:
    s += "- " + i + "\n"

st.markdown(s)

st.markdown(""" 
Disadvantages:

    """)

lst = ['take much time to train (much more than with non modified methods)']

s = ''

for i in lst:
    s += "- " + i + "\n"

st.markdown(s)


st.markdown(
    """ 
    As a result, an investigation of the Gradient Sampling method was conducted in comparison to other methods, demonstrating its ability to improve the convergence of standard techniques. Also, increasing the number of samples could lead to even better results. However, a major drawback of the method is its extensive computational time, which could be an area for improvement in future research. This study was valuable as the Gradient Sampling method was implemented for the first time as part of a neural network optimizer.
    """) 

