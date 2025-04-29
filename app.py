import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

# Streamlit UI
st.title("Binary Classification WebApp")
st.markdown("Are your mushrooms edible or poisonous? 🍄")

st.sidebar.title("Binary Classification")
st.sidebar.markdown("Are your mushrooms edible or poisonous?")

# Load dataset with caching
@st.cache_data
def load_data():
    data = pd.read_csv('mushrooms.csv')
    label = LabelEncoder()
    for col in data.columns:
        data[col] = label.fit_transform(data[col])
    return data

# Split dataset with caching
@st.cache_data
def split(df):
    y = df["type"]
    x = df.drop(columns=["type"])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    return x_train, x_test, y_train, y_test

# Metrics Plotting Function
def plot_metrics(metrics_list, model, x_test, y_test, y_pred):
    if 'Confusion Matrix' in metrics_list:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        st.pyplot()

    if 'ROC Curve' in metrics_list:
        st.subheader("ROC Curve")
        RocCurveDisplay.from_estimator(model, x_test, y_test)
        st.pyplot()

    if 'Precision-Recall Curve' in metrics_list:
        st.subheader("Precision-Recall Curve")
        PrecisionRecallDisplay.from_estimator(model, x_test, y_test)
        st.pyplot()

# Load and split dataset
df = load_data()
x_train, x_test, y_train, y_test = split(df)
class_names = ["edible", "poisonous"]

# Sidebar - Classifier selection
st.sidebar.subheader("Choose Classifier")
classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))

# SVM Classifier
if classifier == "Support Vector Machine (SVM)":
    st.sidebar.subheader("Model Hyperparameters")
    C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01)
    kernel = st.sidebar.radio("Kernel", ("rbf", "linear"))
    gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"))
    
    metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
    
    if st.sidebar.button("Classify"):
        st.subheader("Support Vector Machine (SVM) Results")
        model = SVC(C=C, kernel=kernel, gamma=gamma)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        st.write("Accuracy:", accuracy)
        st.write("Precision:", precision_score(y_test, y_pred))
        st.write("Recall:", recall_score(y_test, y_pred))
        plot_metrics(metrics, model, x_test, y_test, y_pred)

# Logistic Regression Classifier
if classifier == "Logistic Regression":
    st.sidebar.subheader("Model Hyperparameters")
    C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01)
    max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500)
    
    metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
    
    if st.sidebar.button("Classify"):
        st.subheader("Logistic Regression Results")
        model = LogisticRegression(C=C, max_iter=max_iter)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        st.write("Accuracy:", accuracy)
        st.write("Precision:", precision_score(y_test, y_pred))
        st.write("Recall:", recall_score(y_test, y_pred))
        plot_metrics(metrics, model, x_test, y_test, y_pred)

# Random Forest Classifier
if classifier == "Random Forest":
    st.sidebar.subheader("Model Hyperparameters")
    n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10)
    max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1)
    bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'))
    
    metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
    
    if st.sidebar.button("Classify"):
        st.subheader("Random Forest Results")
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap == 'True', n_jobs=-1)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        st.write("Accuracy:", accuracy)
        st.write("Precision:", precision_score(y_test, y_pred))
        st.write("Recall:", recall_score(y_test, y_pred))
        plot_metrics(metrics, model, x_test, y_test, y_pred)

# Show dataset
if st.sidebar.checkbox("Show raw data", False):
    st.subheader("Mushroom Data Set (Classification)")
    st.write(df)

if __name__ == "__main__":
    st.write("App Loaded Successfully!")
