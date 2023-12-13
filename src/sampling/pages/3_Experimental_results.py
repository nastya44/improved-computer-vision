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


st.header("Dataset")
st.image("./assets/cifar.png", caption="CIFAR-10 data sample", use_column_width=True)
st.markdown(
    """The CIFAR-10 dataset is a widely-used dataset in the field of machine learning and computer vision. 
It consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class.
The dataset is divided into 50,000 training images and 10,000 test images.
The classes in the dataset represent everyday objects such as airplanes, cars, birds, cats, deer, dogs, frogs,
horses, ships, and trucks. The simplicity and small size of the images in the CIFAR-10 dataset make it a popular
choice for testing and developing machine learning algorithms, especially for image recognition and classification tasks.
The dataset was created by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton from the University of Toronto.
"""
)


st.header("Final accuracy and loss")
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



