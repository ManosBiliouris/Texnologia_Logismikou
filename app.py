import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans,  AgglomerativeClustering
from sklearn.metrics import silhouette_score


st.markdown("""
    <style>
    .main {
        background-color: #121212;
        color: #e0e0e0;
    }
    h1, h2 {
        color: #BB86FC; /* Light Purple */
        text-align: center;
    }
    .stTabs [role="tab"] {
        border: 1px solid #333;
        padding: 10px 15px; /* Reduced padding to make tabs closer */
        margin: 0 2px; /* Reduced margin to make tabs closer */
        border-radius: 5px;
        background-color: #1F1F1F;
        font-size: 18px;
        color: #BB86FC; /* Light Purple */
        transition: background-color 0.3s, color 0.3s;
        display: inline-block; /* Ensure tabs are inline */
    }
    .stTabs [role="tab"][aria-selected="true"] {
        background-color: #BB86FC; /* Light Purple */
        color: #121212;
    }
    .stTabs [role="tab"]:hover {
        background-color: #333;
    }
    .stButton button {
        background-color: #BB86FC; /* Light Purple */
        color: #121212;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        transition: background-color 0.3s, transform 0.3s;
    }
    .stButton button:hover {
        background-color: #9A67EA; /* Darker Purple */
        transform: scale(1.05);
    }
    .stNumberInput input {
        border-radius: 5px;
        background-color: #333;
        color: #e0e0e0;
        border: 1px solid #555;
        padding: 5px;
    }
    .stFileUploader label {
        background-color: #BB86FC; /* Light Purple */
        color: #121212;
        border-radius: 5px;
        padding: 10px;
        transition: background-color 0.3s, transform 0.3s;
    }
    .stFileUploader label:hover {
        background-color: #9A67EA; /* Darker Purple */
        transform: scale(1.05);
    }
    .css-1aumxhk {
        background-color: #333 !important;
        border: 1px solid #555;
        border-radius: 5px;
        padding: 10px;
        margin-top: 10px;
    }
    .css-1aumxhk:hover {
        background-color: #444 !important;
    }
    .css-145kmo2 {
        background-color: #121212 !important;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("<h1 style='color: #BB86FC;'>Πληροφορίες</h1>", unsafe_allow_html=True)
st.sidebar.markdown("""
    <h2 style='color: #BB86FC;'>Οδηγός Εφαρμογής</h2>
    <h3 style='color: #BB86FC;'>1. Ανέβασμα Αρχείου:</h3>
    <p style='color: #e0e0e0;'>Επιλέξτε την καρτέλα "Ανέβασε το αρχείο σου".</p>
    <p style='color: #e0e0e0;'>Κάντε κλικ στο κουμπί "Επιλέξτε ένα αρχείο CSV" ή στην περιοχή για να επιλέξετε το αρχείο CSV σας.</p>
    <p style='color: #e0e0e0;'>Περιμένετε να φορτωθεί το αρχείο μετά την επιλογή.</p>
    <hr>
    <h3 style='color: #BB86FC;'>2. 2D Οπτικοποίηση:</h3>
    <p style='color: #e0e0e0;'>Επιλέξτε την καρτέλα "2D Visualization".</p>
    <p style='color: #e0e0e0;'>Επιλέξτε τον επιθυμητό τύπο οπτικοποίησης (PCA, t-SNE, EDA).</p>
    <p style='color: #e0e0e0;'>Το PCA και το t-SNE θα οπτικοποιήσουν τα δεδομένα σας σε 2 διαστάσεις. Το EDA θα εμφανίσει περιγραφικά στατιστικά και γραφήματα όπως ιστογράμματα και χάρτες θερμότητας συσχέτισης.</p>
    <hr>
    <h3 style='color: #BB86FC;'>3. Αλγόριθμοι Κατηγοριοποίησης:</h3>
    <p style='color: #e0e0e0;'>Επιλέξτε την καρτέλα "Αλγόριθμοι Κατηγοριοποίησης".</p>
    <p style='color: #e0e0e0;'>Ορίστε τον αριθμό των γειτόνων για το KNeighborsClassifier και τον αριθμό των δέντρων για το RandomForestClassifier.</p>
    <p style='color: #e0e0e0;'>Δείτε την ακρίβεια και τον πίνακα σύγχυσης για κάθε αλγόριθμο. Αναλυτικά στατιστικά όπως τα True Positives (TP), True Negatives (TN), False Positives (FP), και False Negatives (FN) είναι επίσης διαθέσιμα.</p>
    <hr>
    <h3 style='color: #BB86FC;'>4. Αλγόριθμοι Ομαδοποίησης:</h3>
    <p style='color: #e0e0e0;'>Επιλέξτε την καρτέλα "Αλγόριθμοι Ομαδοποίησης".</p>
    <p style='color: #e0e0e0;'>Ορίστε τον αριθμό των ομάδων για το KMeans και το Agglomerative Clustering.</p>
    <p style='color: #e0e0e0;'>Δείτε τα αποτελέσματα της ομαδοποίησης, συμπεριλαμβανομένων των κεντροειδών, των προβλεπόμενων ομάδων και των στατιστικών όπως το inertia και το silhouette score.</p>
    <hr>
    <h3 style='color: #BB86FC;'>5. Πληροφορίες:</h3>
    <p style='color: #e0e0e0;'>Επιλέξτε την καρτέλα "Πληροφορίες" για να δείτε λεπτομέρειες σχετικά με την εφαρμογή και την ομάδα ανάπτυξης.</p>
    <hr>
    <h2 style='color: #BB86FC;'>Πληροφορίες Ομάδας</h2>
    <ul style='color: #e0e0e0;'>
        <li>Ευάγγελος Κωνσταντινίδης</li>
        <li>Εμμανουήλ Μπιλιούρης</li>
        <li>Φώτιος Σταματόπουλος</li>
    </ul>
""", unsafe_allow_html=True)





# Tab definitions
tab1, tab2, tab3, tab4 = st.tabs(["Ανέβασε το αρχείο σου", "2D Visualization", "Αλγόριθμοι Κατηγοριοποίησης", "Αλγόριθμοι Ομαδοποίησης"])

st.set_option('deprecation.showPyplotGlobalUse', False)

with tab1:
    st.write("Εδώ μπορείτε να ανεβάσετε το csv αρχείο σας")
    uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True, type="csv")

with tab2:
    if uploaded_files:
        list_of_dataframes = [pd.read_csv(file, encoding='ISO-8859-1') for file in uploaded_files]
        data = pd.concat(list_of_dataframes, ignore_index=True)

        if data is not None:
            visualization_type = st.selectbox(
                "Choose the visualization type:",
                ("PCA", "t-SNE", "EDA")
            )

            if visualization_type in ['PCA', 't-SNE']:
                if visualization_type == 'PCA':
                    pca = PCA(n_components=2)
                    components = pca.fit_transform(data.select_dtypes(include=[np.number]))
                    st.write("PCA Αποτελέσματα")
                    fig, ax = plt.subplots()
                    ax.scatter(components[:, 0], components[:, 1])
                    ax.set_xlabel('Principal Component 1')
                    ax.set_ylabel('Principal Component 2')
                    st.pyplot(fig)
                elif visualization_type == 't-SNE':
                    tsne = TSNE(n_components=2)
                    components = tsne.fit_transform(data.select_dtypes(include=[np.number]))
                    st.write("t-SNE Αποτελέσματα")
                    fig, ax = plt.subplots()
                    ax.scatter(components[:, 0], components[:, 1], alpha=0.5)
                    ax.set_xlabel('t-SNE Component 1')
                    ax.set_ylabel('t-SNE Component 2')
                    st.pyplot(fig)
            elif visualization_type == 'EDA':
                st.write("Περιγραφικά στατιστικά:", data.describe())

                st.write("Ιστογράμματα για όλες τις αριθμητικές στήλες:")
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                for col in numeric_cols:
                    st.write(f"Ιστόγραμμα από {col}")
                    fig, ax = plt.subplots()
                    sns.histplot(data[col], kde=True, ax=ax)
                    st.pyplot(fig)

                if len(numeric_cols) > 1:
                    st.write("Χάρτης θερμότητας συσχέτισης:")
                    plt.figure(figsize=(10, 7))
                    sns.heatmap(data[numeric_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm')
                    st.pyplot()

with tab3:
    st.write('<span style="font-size:24px; color:#FFD700">Αλγόριθμοι Κατηγοριοποίησης</span>', unsafe_allow_html=True)
    if uploaded_files:
        X = data.drop(columns=data.columns[-1])
        y = data.iloc[:, -1]

        # Χρήστης Δίνει τους γείτονες KNeighborsClassifier
        num_neighbors = st.number_input("Δώσε τον αριθμό των γειτόνων για το KNeighborsClassifier:", min_value=1, max_value=20, step=1, value=3)

        # Χρηστης δίνει το estimator για το Δάσος
        num_estimators = st.number_input("Δώσε τον αριθμό των estimators για το RandomForestClassifier:", min_value=1, max_value=100, step=1, value=10)

        st.markdown("---")

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        # Preprocessing
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Τραιν KNeighborsClassifier model
        knn = KNeighborsClassifier(n_neighbors=num_neighbors)
        knn.fit(X_train_scaled, y_train)

        # Τραιν the RandomForestClassifier model
        rf_classifier = RandomForestClassifier(n_estimators=num_estimators, random_state=0)
        rf_classifier.fit(X_train_scaled, y_train)

        # Προβλεψεις για τον γειτονα
        y_pred_knn = knn.predict(X_test_scaled)

        # Προβλέψεις για το Δάσος
        y_pred_rf = rf_classifier.predict(X_test_scaled)

        st.write('<span style="font-size:24px; color:#FFD700">Στατιστικά για το KNeighborsClassifier</span>', unsafe_allow_html=True)

        # Στατιστικα για τον Γείτονα
        accuracy_knn = knn.score(X_test_scaled, y_test)
        st.write(f'KNeighborsClassifier accuracy with k={num_neighbors}: {accuracy_knn:.2f}')
        st.write("Confusion Matrix for KNeighborsClassifier:")
        st.write(confusion_matrix(y_test, y_pred_knn))

        st.markdown("---")

        st.write('<span style="font-size:24px; color:#FFD700">Στατιστικά για το RandomForestClassifier</span>', unsafe_allow_html=True)
        # Στατιστικα για το Δάσος
        accuracy_rf = rf_classifier.score(X_test_scaled, y_test)
        st.write(f'RandomForestClassifier accuracy with {num_estimators} estimators: {accuracy_rf:.2f}')
        st.write("Confusion Matrix for RandomForestClassifier:")
        st.write(confusion_matrix(y_test, y_pred_rf))

        st.markdown("---")

        st.write('<span style="font-size:24px; color:#FFD700">Αποτελέσματα</span>', unsafe_allow_html=True)
        if accuracy_knn > accuracy_rf:
            st.markdown(f'<span style="font-size:16px; color:#BB86FC">Ο αλγόριθμος KNeighborsClassifier είναι καλύτερος με ακρίβεια {accuracy_knn:.2f}</span>', unsafe_allow_html=True)
        elif accuracy_rf > accuracy_knn:
            st.markdown(f'<span style="font-size:16px; color:#BB86FC">Ο αλγόριθμος RandomForestClassifier είναι καλύτερος με ακρίβεια {accuracy_rf:.2f}</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span style="font-size:16px; color:#BB86FC">Και οι δύο αλγόριθμοι έχουν την ίδια ακρίβεια: {accuracy_knn:.2f}</span>', unsafe_allow_html=True)

        

with tab4:
    st.write('<span style="font-size:24px; color:#FFD700">Αλγόριθμοι Ομαδοποίησης</span>', unsafe_allow_html=True)
    if uploaded_files:
        X = data.drop(columns=data.columns[-1])
        y = data.iloc[:, -1]

        # Ο χρήστης δίνει τα δεδομένα του εδώ
        num_clusters = st.number_input("Δώσε τον αριθμό για το Kmeans:", min_value=1, max_value=20, step=1, value=3)
        num_clusters_agglomerative = st.number_input("Δώσε τον αριθμό των ομάδων για το Agglomerative Clustering:", min_value=1, max_value=20, step=1, value=3)

        st.markdown("---")

        st.write('<span style="font-size:24px; color:#FFD700">Αλγόριθμος Kmeans</span>', unsafe_allow_html=True)

        # Preprocessing
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # KMeans clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        kmeans.fit(X_scaled)
        y_pred_kmeans = kmeans.predict(X_scaled)
        inertia_kmeans = kmeans.inertia_
        silhouette_kmeans = silhouette_score(X_scaled, y_pred_kmeans)

        st.write(f'Centroids of KMeans with {num_clusters} clusters:')
        st.write(kmeans.cluster_centers_)

        st.write("Predicted clusters for the dataset:")
        st.write(y_pred_kmeans)
        
        st.write(f"KMeans Inertia: {inertia_kmeans}")
        st.write(f"KMeans Silhouette Σκορ: {silhouette_kmeans}")

        st.markdown("---")

        st.write('<span style="font-size:24px; color:#FFD700">Αλγόριθμος Agglomerative Clustering</span>', unsafe_allow_html=True)
        

        # Agglomerative Clustering
        agglomerative = AgglomerativeClustering(n_clusters=num_clusters_agglomerative)
        y_pred_agglomerative = agglomerative.fit_predict(X_scaled)
        silhouette_agglomerative = silhouette_score(X_scaled, y_pred_agglomerative)

        st.write(f'Predicted clusters for Agglomerative Clustering with {num_clusters_agglomerative} clusters:')
        st.write(y_pred_agglomerative)
        
        st.write(f"Agglomerative Clustering Silhouette Σκορ: {silhouette_agglomerative}")

        st.markdown("---")

        st.write('<span style="font-size:24px; color:#FFD700">Αποτελέσματα</span>', unsafe_allow_html=True)
        if silhouette_kmeans > silhouette_agglomerative:
            st.markdown(f'<span style="font-size:16px; color:#BB86FC">Ο αλγόριθμος KMeans είναι καλύτερος με Silhouette Σκορ {silhouette_kmeans:.2f}</span>', unsafe_allow_html=True)
        elif silhouette_agglomerative > silhouette_kmeans:
            st.markdown(f'<span style="font-size:16px; color:#BB86FC">Ο αλγόριθμος Agglomerative Clustering είναι καλύτερος με Silhouette Σκορ {silhouette_agglomerative:.2f}</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span style="font-size:16px; color:#BB86FC">Και οι δύο αλγόριθμοι έχουν το ίδιο Silhouette Σκορ: {silhouette_kmeans:.2f}</span>', unsafe_allow_html=True)

        

