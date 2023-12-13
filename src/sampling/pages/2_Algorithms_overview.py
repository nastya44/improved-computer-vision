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

Therefore, if $ \\nabla f(x) = \\mathbb{E} [ \\nabla_x F(x, \\xi) ] $, the vector $ u^k = \\nabla_x F(x^k, \\xi^k) $ is indeed the stochastic 
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

st.header("Momentum")
st.markdown(    
    """
Returning back to the analogy of gradient descent algorithm as a ball, rolling down the hill, important to mention that it's not rolling with a constant speed $ \\lambda $, but instead gaining speed with time by keeping the momentum of itself. The same approach was discovered in optimization theory: the convergence rate of an algorithm can be increased by applying a fraction $ \gamma $ of gradient from the previous iteration:

$ v_{t} = \\gamma v_{t-1} + \\lambda \\nabla_{\\theta} J(\\theta_{1}^{(i)}, ..., \\theta_{n}^{(i)}, x_{1}^{(i)}, ..., x_{n}^{(i)}, y^{(i)}) $

$ \\theta^{(i+1)} = \\theta^{(i)} - v_{t} $

The momentum term $ \\gamma $ is usually set to 0.9 or a similar value.

As mentioned in previous sections, SGD has trouble navigating ravines, i.e. areas where the surface curves much more steeply in one dimension than in another, which are common around the local minimum. Momentum helps accelerate SGD in the relevant direction and dampens oscillations.
"""
)

st.header("Nesterov accelerated gradient (NAG)")
st.markdown(    
    """
In this approach, the momentum is not only applied to the calculated gradient value on the iteration, but also to the current parameters, which gives an approximation of their next position, like a preliminary step of the algorithm:

$ v_{i} = \\gamma v_{i-1} + \\lambda \\nabla_{\\theta} J(\\theta^{(i)} - \\gamma v_{i-1}) $

$ \\theta^{(i+1)} = \\theta^{(i)} - v_{i} $

Taking the analogy from above, if a ball is made of light material, it may slope up to the other side of a hill using its momentum. But if it is made of heavy material (like steel), it will slow down near the bottom of the hill. The NAG algorithm work by the same principle. While momentum first computes the current gradient and then takes a big jump in the direction of the updated accumulated gradient, NAG first makes a big jump in the direction of the previously accumulated gradient, measures the gradient, and then makes a correction. This anticipatory update prevents us from going too fast and results in increased responsiveness.
"""
)

st.header("Gradient sampling algorithm")
st.markdown(    
    """
One of the newest approaches in general NSO is to use gradient sampling algorithms developed by Burke. The gradient sampling method (GS) is a method for minimizing an objective function that is locally Lipschitz continuous and smooth on an open dense subset $D \\in \\mathbb{R}^n$. The objective may be nonsmooth and/or nonconvex. The GS may be considered as a stabilized steepest descent algorithm. The central idea behind these techniques is to approximate the subdifferential of the objective function through random sampling of gradients near the current iteration point. The ongoing progress in the development of gradient sampling algorithms  suggests that they may have potential to rival bundle methods in the terms of theoretical might and practical performance. 


Let $f$ be a locally Lipschitz continuous function on $\\mathbb{R}^n$, and suppose that $f$ is smooth on an open dense subset $D \\in \\mathbb{R}^n$. In addition, assume that there exists a point such that the level set $lev_{ƒ(\\bar{x})} = \\{x | f(x)≤ f(\\bar{x})\\}$ is compact. At a given iterate $x_k$ the gradient of the objective function is computed on a set of randomly generated nearby points $u_{kj}$ with $j \\in \\{1, 2,\\ldots, m\\}$ and $m > n + 1$. This information is utilized to construct a search direction as a vector in the convex hull of these gradients with the shortest norm. A standard line search is then used to obtain a point with lower objective function value. The stabilization of the method is controlled by the sampling radius ε used to sample the gradients.
The algorithm of the GS is the following:


1. independently sample $\\left\\{x^{k, 1}, \\ldots, x^{k, m}\\right\\}$ uniformly from $\\bar{B}\\left(x^k ; \\epsilon_k\\right)$ 


2. compute $\\boldsymbol{g}^k$ as the solution of $\\min _{g \\in \\mathcal{G}^k} \\frac{1}{2}\\|\\boldsymbol{g}\\|^2$, where
$$
\\mathcal{G}^k:=\\left\\{\\nabla f\\left(\\boldsymbol{x}^k\\right), \\nabla f\\left(\\boldsymbol{x}^{k, 1}\\right), \\ldots, \\nabla f\\left(\\boldsymbol{x}^{k, m}\\right)\\right\\}
$$

3. use $\\boldsymbol{g}^k$ as gradient for calculating updated parameter


"""
)
