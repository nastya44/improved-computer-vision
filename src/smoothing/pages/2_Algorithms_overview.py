import streamlit as st


st.set_page_config(
    page_title="Gradient-free deep learning: Algorithms overview",
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
st.markdown(
    """
The AdaGrad algorithm adapts the learning rate to the parameters, performing larger updates for infrequent and smaller updates for frequent parameters: given learning rate $ \lambda $ value is divided on some term $ G_{i} $ that accumulates gradient values from previous iterations:

$ \\theta^{(i+1)} = \\theta^{(i)} - \\frac{\lambda}{\sqrt{G_{i} + \epsilon}} \odot g_{i} $, where $ g_{i} = \\nabla_{\\theta} J(\\theta^{(i)})$, $ \epsilon $ is a smoothing term (a small coefficient, usually equals $ 3 \cdot 10^{-6} $, that prevents the denominator from turning to zero) and  $ G_{i} $ is a combination of the previously accumulated gradient terms and element-wise product of $ g_{i} $ on itself: $ G_{i} = G_{i-1} + g_{i} \cdot g_{i} = G_{i-1} + (g_{i})^{2} $

Although the mentioned problem with tuning learning rate $ \lambda $ is solved (in practice, the value is set to default 0.01 and left like that), the new one appears: accumulation of the squared gradients in the denominator causes $ \lambda $ to shrink with every iteration (since every term is positive) and eventually is approaching zero. It can be illustrated on the given optimization problem for a small set of training examples: the learning rate $ \lambda $ should be big enough ($ \lambda $ = 0.1 for this example) to reach an area near the global minimum.
    """
)

st.header("Root mean square propagation (RMSProp)")
st.markdown(
    """
A serious drawback of the AdaGrad algorithm is and exponential decay of a learning rate $ \lambda $, due to the accumulation of the gradient values and uncontrollable growth in the denominator. On the one hand, rate decay results in reducing fluctuations in the objective function, but on the other hand, the rate becomes negligibly small after some iterations, so the algorithm may never reach the global minimum.

In the RMSProp (Root Mean Square Propagation) algorithm this issue is solved by using the mean value of gradients in the denominator from the previous iterations instead of constant accumulation:

$ E(g^{2})_{i} = \\beta E(g^{2})_{i-1} + (1 - \\beta) g_{i}^{2} $

$ \\theta^{(i+1)} = \\theta^{(i)} - \\frac{\lambda}{\sqrt{E(g^{2})_{i} + \epsilon}} \odot g_{i} $

where $ g_{i} = \\nabla_{\\theta} J(\\theta^{(i)})$, $ \epsilon $ is a smoothing term, and $ \\beta $ is a momentum term (according to the original research, the optimal value is 0.9
    """
)

st.header("Adaptive moment estimation (ADAM)")
st.markdown(
    """
The ADAM (Adaptive Moment Estimation) algorithm adapts not only the learning rate $ \lambda $ to the parameters (like AdaGrad and RMSProp) also keeps an exponentially decaying average of past gradients $ m_{i} $, similar to momentum:

$ m_{i} = \\beta_{1} m_{i-1} + (1 - \\beta_{1}) g_{i} $

$ v_{i} = \\beta_{2} v_{i-1} + (1 - \\beta_{2}) g_{i}^{2} $

where $ g_{i} = \\nabla_{\theta} J(\\theta^{(i)})$, $ m_{i} $ - mean value estimation (first moment) of the gradients respectively, $ v_{i} $ - the uncentered variance estimation (second moment) of the gradients respectively.

The original research was shown, that moment estimations are biased towards zero on the first iterations, especially when terms $ \\beta_{1} $, $ \\beta_{2} $ are small. So in the descent algorithm, bias-corrected estimates are used:

$ \hat{m_{i}} = \\frac{m_{i}}{1 - \\beta_{1}} $

$ \hat{v_{i}} = \\frac{v_{i}}{1 - \\beta_{2}} $

Now update rule for parameters can be modified using adaptive estimations:

$ \\theta^{(i+1)} = \\theta^{(i)} - \\frac{\lambda}{\sqrt{\hat{v_{i}} + \epsilon}} \hat{m_{i}} $

From the empirical point of view, default estimation terms for the algorithm are equal: $ \\beta_{1} = 0.9 $, $ \\beta_{2} = 0.999 $.

For the given optimization problem, the approach with adaptive estimations for learning rate $ \lambda $ and for momentum will result in not only efficient convergence on the sparse data but also in adaptive decay of the momentum when the algorithm is approaching the minimum point (so the algorithm will not jump over the minimum, unlike pure momentum algorithm).
    """
)

st.header("Finite-difference smoothing algorithm (FinD)")
st.markdown(
    """
Gradient methods in non-smooth optimization problems demonstrate a slow convergence rate and low robustness to the local minima. Therefore, to smooth the problem and reduce the problem's convexity, finite-difference algorithm was introduced:

***Finite-diference algorthm***

**Require**: $F$ - a nonsmooth $L$-Lipschitz continuity, initial element $x_0$, step multiplier $0 < \lambda < 1$, $K$ a smoothing multiplier.

1. **Set** $i = 0;$
2. **Repeat:**
* **Set** $x^i = x^{i-1} - \lambda \cdot S_k;$
* **Set** $i = i + 1;$
3. **Until** $|x^i - x^{i-1}| < \epsilon_1, |F_h(x^i) - F_h(x^{i-1})| < \epsilon_2$

Here $S_k = \\frac{1}{K} \sum_{i=1}^K \\frac{1}{2h_k} (F(x_k + h_k \\tilde y_{k,i}) - F(x_k - h_k \\tilde y_{k, i})) \\tilde y_{k, i}$ - finite-difference gradient approximations, $h_k$ - approximation grid size, $\\tilde y_{k, i}$ - uniform distribution on a unit sphere $B_1(0)$.

To enable usage of FinD in given above methods we can simply replace gradient $g_i$ vectors with $S_k$.
    """
)
