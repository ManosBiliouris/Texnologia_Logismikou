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

# Tab definitions
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Ανέβασε το αρχείο σου", "2D Visualization", "Αλγόριθμοι Κατηγοριοποίησης", "Αλγόριθμοι Ομαδοποίησης", "Information"])

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

        
with tab5:
    st.write("""
    ### Οδηγίες Χρήσης

    1. **Ανέβασε το αρχείο σου**:
        - Επιλέξτε το tab "Ανέβασε το αρχείο σου".
        - Κάντε κλικ στο κουμπί ή στην περιοχή "Choose a CSV file" για να επιλέξετε το αρχείο CSV που θέλετε να χρησιμοποιήσετε.
        - Μετά την επιλογή του αρχείου, παρακαλώ περιμένετε για τη φόρτωση των δεδομένων.
    ****
    2. **2D Visualization**:
        - Επιλέξτε το tab "2D Visualization".
        - Επιλέξτε τον τύπο οπτικοποίησης που επιθυμείτε (PCA, t-SNE, EDA).
        - Για PCA και t-SNE, τα δεδομένα σας θα οπτικοποιηθούν σε 2 διαστάσεις.
        - Για EDA, θα δείτε περιγραφικά στατιστικά και γραφήματα όπως ιστογράμματα και χάρτες θερμότητας συσχέτισης.
    ****   
    3. **Αλγόριθμοι Κατηγοριοποίησης**:
        - Επιλέξτε το tab "Αλγόριθμοι Κατηγοριοποίησης".
        - Ορίστε τον αριθμό των γειτόνων για τον αλγόριθμο KNeighborsClassifier και τον αριθμό των estimators για το RandomForestClassifier.
        - Δείτε την ακρίβεια και τον πίνακα σύγχυσης για κάθε αλγόριθμο. Επίσης, θα δείτε αναλυτικά στατιστικά όπως True Positives (TP), True Negatives (TN), False Positives (FP), και False Negatives (FN) για κάθε κατηγορία.
    ****         
    4. **Αλγόριθμοι Ομαδοποίησης**:
        - Επιλέξτε το tab "Αλγόριθμοι Ομαδοποίησης".
        - Ορίστε τον αριθμό των ομάδων για τους αλγόριθμους KMeans και Agglomerative Clustering.
        - Δείτε τα αποτελέσματα της ομαδοποίησης, συμπεριλαμβανομένων των κεντροειδών, των προβλεπόμενων ομάδων, και των στατιστικών όπως inertia και silhouette score.
    ****
    5. **Information**:
        - Επιλέξτε το tab "Information" για να δείτε πληροφορίες σχετικά με την εφαρμογή και την ομάδα που τη δημιούργησε.
    """)

    st.write("""
    ****
    ### Πληροφορίες Ομάδας

    - Ευάγγελος Κωνσταντινίδης
    - Εμμανουήλ Μπιλιούρης
    - Φώτιος Σταματόπουλος
    """)

