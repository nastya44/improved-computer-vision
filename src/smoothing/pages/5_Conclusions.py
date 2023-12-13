import streamlit as st


st.set_page_config(
    page_title="Gradient-free deep learning: Conclusions",
    page_icon="📌",
    layout="centered",
    initial_sidebar_state="expanded",
)


st.title("Conclusions")
st.markdown(
    """
    **In this research we have:**
    * Implemented modified gradient-free versions of adaptive gradient descent algorithms.
    * Tested implemented optimizers on classical image classification dataset - MNIST.
    * Investigated new hyper-parameters and their impact on final performance.
    * Trained and deployed object detection model for vehicle detection task.

    Even though finite-difference smoothing algorithm was unable to outperform the classical gradient descent optimizers, there are still some
    conclusions we can reach:

    First of all, we can notice that adaptive learning rate modifications of FinD underperform, compared to FinD-SGD. Which makes us think, that
    because of the randomness inside the algorithm logic, acumulation of gradient direction leads to suboptimal positions of loss surface. Most likely this issue can be solved
    with increase of approximations accuracy, although this requires more computational resource and low-level code optimizations.

    Now, let's get to the comparison of regular and finite-difference versions of optimizers. We can't help but notice that results of FinD versions are worse, but
    aslo we can see, that with increase of K hyper-parameter, results slowly converge to better values. Which means that there is still room for improvent and, yet again, 
    better results might
    be achived by increasing approximation accuracy.

    At the end we would like to say, that finite-difference algorithms have potential to be at least as good as gradient descent optimizers, but they require additional work
    on parallel computing logic to be able to run as fast as backpropagation in modern deep learning toolkits.
    """
)