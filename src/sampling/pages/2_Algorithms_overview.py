import streamlit as st


st.set_page_config(
    page_title="Gradient sampling deep learning: Algorithms overview",
    page_icon="📖",
    layout="centered",
    initial_sidebar_state="expanded",
)


st.title("Algorithms overview")
st.header("Stochastic gradient descent")
st.markdown(
    """
> **Definition 1.** The random vector $ u^k $ is designated as the stochastic gradient of the function $ f(x) $ at the point $ x = x^k $ if
the condition $ \\mathbb{E} \\{ u^k | x^k \\} = \\nabla f(x^k) $ holds true, with $ \\mathbb{E} \\{ u^k | x^k \\} $ denoting the conditional mathematical expectation.

Therefore, if $ \\nabla f(x) = \mathbb{E} [ \\nabla_x F(x, \\xi) ] $, the vector $ u^k = \\nabla_x F(x^k, \\xi^k) $ is indeed the stochastic 
gradient of the function $ f(x) $ at the point $ x = x^k $.

For the definition of $ u^k = \\nabla_x F(x^k, \\xi^k) $, we denote it as the gradient along the $ x $ variable for the function $ F(\\cdot, \\xi^k) $, holding 
the parameter value $ \\xi = \\xi^k $ as constant. Here, $ \\rho_k $ signifies the non-negative step multipliers for the gradient, and $ \\{ \\xi^k \\} $ designates 
the independent observations (or statistics) of a random variable $ \\xi $. The corresponding stochastic gradient descent method is then expressed by the subsequent 
recurrent equation:

$$ x^{k+1} = \\Pi_X (x^k - \\rho_k u^k), \\> x^0 \\in X, \\> k \\in \\mathbb{N} $$

In order to ensure the convergence of the method given by equation, the step multipliers $ \\rho_k \\geq 0 $ are required to satisfy the conditions delineated as follows:

$$ \\sum_{k=0}^{\\infty} \\rho_k = + \\infty, \\> \\> \\> \\sum_{k=0}^{\\infty} \\rho_k^2 < + \\infty $$
"""
)

st.header("Adaptive gradient method (AdaGrad)")
st.info("**TODO:** Add `AdaGrad` method description with $\\LaTeX$ formulas")

st.header("Root mean square propagation (RMSProp)")
st.info("**TODO:** Add `RMSProp` method description with $\\LaTeX$ formulas")

st.header("Gradient sampling algorithm")
st.info("**TODO:** Add a proposed `Sampling` method description with $\\LaTeX$ formulas")
