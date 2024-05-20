# 🎉 Streamlit Data Analysis App with Docker 🌟

Welcome to our super cool Data Analysis App built with Streamlit and Docker! 🚀

## 🏫 About the Project

This project is developed by a team of enthusiastic computer science students from the Ionian University. Our goal is to create a user-friendly web application that makes data analysis fun and easy! 🥳

## 📈 Features

- **Upload Your Data**: Easily upload your CSV files and start analyzing!
- **2D Visualization**: Choose from PCA, t-SNE, and EDA to visualize your data in 2D.
- **Classification Algorithms**: Utilize KNeighborsClassifier and RandomForestClassifier for data classification.
- **Clustering Algorithms**: Experiment with KMeans and Agglomerative Clustering for data grouping.
- **Docker Integration**: Run the app in a consistent environment using Docker.

## 🎨 App Design

- **User-Friendly Interface**: Designed with a playful and intuitive interface using Streamlit.
- **Responsive Layout**: Ensures a great experience on any device.

## 🛠️ Tech Stack

- **Streamlit**: For creating the web interface.
- **Docker**: For consistent deployment and running environment.
- **Scikit-learn**: For implementing machine learning algorithms.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib & Seaborn**: For data visualization.

## 📊 Try our Website

👉 [Streamlit Data Analysis App](https://softwaretechnology-b42qedritrsylcjthcxrvx.streamlit.app) 👈

## 🚀 How to Run the App

### With Docker

1. Clone the repo:
    ```sh
    git clone https://github.com/ManosBiliouris/Texnologia_Logismikou.git
    cd Texnologia_Logismikou
    ```
2. Build the Docker image:
    ```sh
    docker build -t streamlit-data-analysis .
    ```
3. Run the Docker container:
    ```sh
    docker run -p 8501:8501 streamlit-data-analysis
    ```
4. Open your browser and navigate to `http://localhost:8501`.

### Without Docker

1. Clone the repo:
    ```sh
    git clone https://github.com/ManosBiliouris/Texnologia_Logismikou.git
    cd Texnologia_Logismikou
    ```
2. Install the requirements:
    ```sh
    pip install -r requirements.txt
    ```
3. Run the Streamlit app:
    ```sh
    streamlit run app.py
    ```

## 👨‍💻 Meet the Team

- **Evangelos Konstantinidis**: Lead Developer [GitHub](https://github.com/vag4me)
- **Emmanouil Biliouris**: Docker Expert [GitHub](https://github.com/ManosBiliouris)
- **Fotios Stamatopoulos**: UML Designer [GitHub](https://github.com/FOTAKLAS)

## 📚 Project Documentation

- Check out our detailed [Project Report](https://github.com/ManosBiliouris/Texnologia_Logismikou/blob/main/report.pdf) (in Greek) for more information about the development process, design, and results.

Have fun exploring and analyzing your data! 🎉 If you have any questions or feedback, feel free to open an issue or contact us. 😊

Happy Data Analyzing! 📊✨
