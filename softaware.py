import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix , classification_report

#Οτι Tab Θα χρειαστουμε , tab1 = Ανεβασε το αρχειο σου , tab2 =  2D Visualization και τα λοιπααα!!!
tab1 , tab2 , tab3 , tab4 ,tab5 = st.tabs(["Ανέβασε το αρχείο σου", "2D Visualization" , "Μηχανικής Μάθησης" ,"Αποτελέσματα και Σύγκριση" , "Information"])

st.sidebar.title("Εργασία Εξαμήνου")#ειναι το Sidebar 
st.set_option('deprecation.showPyplotGlobalUse', False)

with tab1:
    st.write("Εδώ μπορείτε να ανεβάσετε το csv αρχείο σας")
    uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True , type = "csv") # Καλυπτει κ το ερωτημα 2 
with tab2:
    if uploaded_files: # Καλυπτει κ το ερωτημα 2
        list_of_dataframes = [pd.read_csv(file, encoding='ISO-8859-1') for file in uploaded_files] #allagi utf to ISO
        data = pd.concat(list_of_dataframes, ignore_index=True) # Καλυπτει κ το ερωτημα 2

        if data is not None:
            # Bar επιλογων για τους αλγοριθμους
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
                # Περιγραφικα στατιστικα
                st.write("Περιγραφικά στατιστικά:", data.describe())

                # Ιστογραμματα για ολες τις αριθμητικες στηλες
                st.write("Ιστογράμματα για όλες τις αριθμητικές στήλες:")
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                for col in numeric_cols:
                    st.write(f"Ιστόγραμμα από {col}")
                    fig, ax = plt.subplots()
                    sns.histplot(data[col], kde=True, ax=ax)
                    st.pyplot(fig)

                # Χαρτης θερμοτητας συσχετισης
                if len(numeric_cols) > 1:
                    st.write("Χάρτης θερμότητας συσχέτισης:")
                    plt.figure(figsize=(10, 7))
                    sns.heatmap(data[numeric_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm')
                    st.pyplot()
with tab3:
    st.write("αλγόριθμοι Κατηγοριοποίησης  1) KNeighborsClassifier")
    if uploaded_files:
        X = data.drop(columns=data.columns[-1])  
        y = data.iloc[:, -1]
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)

        y_pred = knn.predict(X_test)
        st.write(f'Our model accuracy with k=3 is: {knn.score(X_test, y_test)}')

        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))
        
        st.write("Classification Report:")
        st.write(classification_report(y_test, y_pred))

with tab4:
    if uploaded_files: # Καλυπτει και το ερώτημα 2
        # Εμφάνιση αποτελεσμάτων ανάλυσης και σύγκριση
        st.write("Αποτελέσματα Ανάλυσης και Σύγκριση")

        # Περιγραφικά στατιστικά
        st.write("Περιγραφικά στατιστικά:", data.describe())

        # Διάγραμμα scatter plot για τις προβλεπόμενες τιμές
        st.write("Διάγραμμα Scatter Plot για τις προβλεπόμενες τιμές")
        fig, ax = plt.subplots()
        #ax.scatter(y, predictions)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
        ax.set_xlabel('Πραγματικές Τιμές')
        ax.set_ylabel('Προβλεπόμενες Τιμές')
        st.pyplot(fig)



with tab5:     #Το παρακατω κειμενακι θα αλλαξει λογικα οσο αναπτυσεται το app
    st.write("""
    ### Οδηγίες Χρήσης

    1. **Ανέβασε το αρχείο σου**:
        - Επιλέξτε το tab "Ανέβασε το αρχείο σου".
        - Κάντε κλικ στο κουμπί ή στην περιοχή "Choose a CSV file" για να επιλέξετε το αρχείο CSV που θέλετε να χρησιμοποιήσετε.
        - Μετά την επιλογή του αρχείου, παρακαλώ περιμένετε για τη φόρτωση των δεδομένων.
    ****
    2. **2D Visualization**:
        - Επιλέξτε το tab "2D Visualization".
        - Εδώ μπορείτε να προβάλετε οπτικοποιήσεις των δεδομένων σας σε διάφορες διαστάσεις.
    ****   
    3. **Μηχανικής Μάθησης**:
        - Επιλέξτε το tab "Μηχανικής Μάθησης".
        - Εδώ μπορείτε να εφαρμόσετε αλγόριθμους μηχανικής μάθησης στα δεδομένα σας.
    ****         
    4. **Αποτελέσματα και Σύγκριση**:
        - Επιλέξτε το tab "Αποτελέσματα και Σύγκριση".
        - Εδώ μπορείτε να δείτε τα αποτελέσματα της ανάλυσης σας και να συγκρίνετε διαφορετικές προσεγγίσεις.
    ****
    5. **Information**:
        - Επιλέξτε το tab "Information".
        - Εδώ θα βρείτε πληροφορίες σχετικά με την ομάδα που δημιούργησε την εφαρμογή ή άλλες γενικές πληροφορίες που θεωρείτε σημαντικές για τους χρήστες σας.
    """)

    st.write("""
    ****
    ### Πληροφορίες Ομάδας

    - Ευάγγελος Κωνσταντινίδης
    - Εμμανουήλ Μπιλιούρης
    - Φώτιος Σταματόπουλος
    """)
