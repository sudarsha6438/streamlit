# pip install scikit-learn
# pip install pandas
# pip install streamlit
# To run the code: streamlit run main.py

import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Load Iris dataset
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

def main():
    st.set_page_config(layout="wide")  # Wide layout helps with column placement
    st.title("🌸 Streamlit Demo: ML Models on Iris Dataset")

    # Sidebar 1: Model selection
    with st.sidebar:
        st.header("🧠 Select Algorithms")
        use_random_forest = st.checkbox("Random Forest", True)
        use_decision_tree = st.checkbox("Decision Tree", True)
        use_svm = st.checkbox("Support Vector Machine (SVM)", False)

    # Sidebar 2: Configuration settings (simulated sidebar using columns)
    col1, col2, col3 = st.columns([1, 3, 1])  # Layout: Sidebar 2 | Main content | Empty or spacer

    with col1:
        st.markdown("## ⚙️ Configuration")
        random_state = st.number_input("Random State", min_value=0, value=60)
        test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2, step=0.05)

    with col2:
        # Split data
        X = df[data.feature_names]
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        results = {}

        if use_random_forest:
            rf_model = RandomForestClassifier(random_state=random_state)
            rf_model.fit(X_train, y_train)
            rf_y_pred = rf_model.predict(X_test)
            results['Random Forest'] = {
                'Accuracy': accuracy_score(y_test, rf_y_pred),
                'Report': classification_report(y_test, rf_y_pred, target_names=data.target_names, output_dict=True)
            }

        if use_decision_tree:
            dt_model = DecisionTreeClassifier(random_state=random_state)
            dt_model.fit(X_train, y_train)
            dt_y_pred = dt_model.predict(X_test)
            results['Decision Tree'] = {
                'Accuracy': accuracy_score(y_test, dt_y_pred),
                'Report': classification_report(y_test, dt_y_pred, target_names=data.target_names, output_dict=True)
            }

        if use_svm:
            svm_model = SVC(random_state=random_state)
            svm_model.fit(X_train, y_train)
            svm_y_pred = svm_model.predict(X_test)
            results['SVM'] = {
                'Accuracy': accuracy_score(y_test, svm_y_pred),
                'Report': classification_report(y_test, svm_y_pred, target_names=data.target_names, output_dict=True)
            }

        # Display dataset
        st.write("### 🌼 Dataset Overview")
        st.dataframe(df.head())

        # Display results
        st.write("### 📊 Model Results")
        for model_name, result in results.items():
            st.write(f"#### {model_name}")
            st.write(f"**Accuracy:** {result['Accuracy']:.2f}")
            st.write("**Classification Report:**")
            st.dataframe(pd.DataFrame(result['Report']).transpose())

        # Prediction section
        st.write("### 🔮 Make a Prediction")
        input_data = []
        st.write("#### Set Feature Values Using Sliders")
        for feature in data.feature_names:
            min_value = float(X[feature].min())
            max_value = float(X[feature].max())
            value = st.slider(f"{feature}", min_value, max_value, float(X[feature].mean()))
            input_data.append(value)

        if st.button("Predict Individually"):
            if use_random_forest:
                rf_prediction = rf_model.predict([input_data])
                st.write(f"🌲 Random Forest Prediction: **{data.target_names[rf_prediction[0]]}**")
            if use_decision_tree:
                dt_prediction = dt_model.predict([input_data])
                st.write(f"🌳 Decision Tree Prediction: **{data.target_names[dt_prediction[0]]}**")
            if use_svm:
                svm_prediction = svm_model.predict([input_data])
                st.write(f"🧮 SVM Prediction: **{data.target_names[svm_prediction[0]]}**")

if __name__ == "__main__":
    main()
