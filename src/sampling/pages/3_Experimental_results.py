import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Gradient sampling deep learning: Experimental results",
    page_icon="🧪",
    layout="centered",
    initial_sidebar_state="expanded",
)


st.title("Experimental results")

st.markdown(
    """<style>
        td {text-align: center !important}
        th {text-align: center !important}
    </style>
    """, unsafe_allow_html=True) 


methods_score = {"sgd_score": 0.831, "momentum_score": 0.852, "nag_score": 0.858, "gs_sgd_score": 0.838, "gs_momentum_score": 0.867, "gs_nag_score": 0.864}

# Sample data
data = {
    'Methods Name': ['Stochastic gradient descent (SGD)', 'Momentum', 'Nesterov accelerated gradient (NAG)', 'SGD + Gradient Sampling', 'Momentum + Gradient Sampling', 'NAG + Gradient Sampling'],
    'Final losses': [0.4122, 0.3166, 0.3022, 0.4013, 0.2979, 0.2998],
    #'Final accuracies': [str(round(methods_score.get("sgd_score") * 100, 1)) + '%', '81.1%', '83.5%', '82.8%', '84.1%', '85.7%']
}

data['Final accuracies'] = [f"{round(methods_score.get(key, 0) * 100, 1)}%" for key in ['sgd_score', 'momentum_score', 'nag_score', 'gs_sgd_score', 'gs_momentum_score', 'gs_nag_score']]
# Create a DataFrame
df = pd.DataFrame(data)

# Display the table
st.table(df)

# Read CSV file into a pandas DataFrame
csv_file_path = 'data/losses/gs_momentum_losses.csv'
df = pd.read_csv(csv_file_path)
gs_momentum_losses = df.to_numpy()

csv_file_path = 'data/losses/sgd_losses.csv'
df = pd.read_csv(csv_file_path)
sgd_losses = df.to_numpy()

csv_file_path = 'data/losses/gs_sgd_losses.csv'
df = pd.read_csv(csv_file_path)
gs_sgd_losses = df.to_numpy()

csv_file_path = 'data/losses/momentum_losses.csv'
df = pd.read_csv(csv_file_path)
momentum_losses = df.to_numpy()

csv_file_path = 'data/losses/nag_losses.csv'
df = pd.read_csv(csv_file_path)
nag_losses = df.to_numpy()

csv_file_path = 'data/losses/gs_nag_losses.csv'
df = pd.read_csv(csv_file_path)
gs_nag_losses = df.to_numpy()

fig = plt.figure(figsize=(10, 6))

plt.plot(gs_sgd_losses, label='Sgd Gradient Sampling')
plt.plot(gs_momentum_losses, label='Momentum Gradient Sampling')
plt.plot(sgd_losses, label='Sgd')
plt.plot(momentum_losses, label='Momentum')
plt.plot(gs_nag_losses, label='Nag Gradient Sampling')
plt.plot(nag_losses, label='Nag')

plt.title('Gradient descent algorithms comparison')
plt.legend()

st.pyplot(plt.gcf())



