# 🌸 Iris Data Analysis Project 🌸
Welcome to the Iris Data Analysis Project! This repository includes advanced analysis and visualization techniques applied to the Iris dataset. From exploring data distribution to performing dimensionality reduction and clustering, this project is packed with interesting insights and methods. 🚀

## 🛠️ Features
#### 📊 Exploratory Data Analysis (EDA):
   - Distribution plots for features
   - Pairwise relationships with pairplots
   - Boxplots for outlier detection

#### 📉 Dimensionality Reduction:
   - PCA for visualizing data in 2D
   - Advanced t-SNE visualization

#### 📈 Statistical Analysis:
   - Skewness and kurtosis
   - Hypothesis testing with ANOVA

#### 🌟 Advanced Feature Analysis:
   - Feature importance using Random Forest
   - Mutual information scores
   - Cross-feature analysis with parallel coordinates

#### 🔍 Clustering & Outlier Detection:
   - K-Means clustering with silhouette scores
   - Isolation Forest for anomaly detection

#### 📐 Machine Learning Models:
   - Logistic Regression for classification
   - Evaluation with classification reports

#### 🎨 Interactive Visualizations:
   - Radial charts for feature distribution
   - Interaction effects with scatter plots


## 📂 Dataset
The project utilizes the Iris dataset, which consists of 150 samples of iris flowers (Setosa, Versicolor, Virginica) with the following features:

- 🌱 Sepal Length
- 🌱 Sepal Width
- 🌸 Petal Length
- 🌸 Petal Width


## 🚀 How to Run the Project

1. Clone the repository:

git clone [https://github.com/your-repo/iris-analysis.git](https://github.com/priyankaa-roy/Iris-Data-Analysis-Project-)
cd iris-analysis

2. Install the required libraries:

pip install -r requirements.txt

3. Run the analysis scripts:

python iris_analysis.py


## 🧪 Key Visualizations

#### Dimensionality Reduction 🖼️
PCA and t-SNE reveal how species are grouped in feature space.

![image](https://github.com/user-attachments/assets/667fc121-9833-4972-a70d-1e838b8e8210)


#### Outlier Detection 🔴
Isolation Forest highlights anomalies within the dataset.

![image](https://github.com/user-attachments/assets/b8f80d32-e001-4651-854d-b47a809ce863)  
![image](https://github.com/user-attachments/assets/020633f8-da5e-4694-9531-0d1b5057a3e7)



#### Feature Analysis 🌟

Insights into feature importance using Random Forest and Mutual Information.
![image](https://github.com/user-attachments/assets/d841f921-1afb-475f-809e-417426e42bcf)



## 📜 Insights
1. `Feature Distributions`: Petal dimensions show stronger separability between species compared to sepal dimensions.
2. `Cluster Validity`: K-Means clustering aligns well with actual species, validated by a high silhouette score.
3. `Predictive Features`: Petal Length and Petal Width are the most critical predictors of species.

## 🛠️ Technologies Used
- Python Libraries: `pandas`, `numpy`, `seaborn`, `matplotlib`, `scikit-learn`, `statsmodels`
- Machine Learning: PCA, t-SNE, Random Forest, Logistic Regression, K-Means
- Statistical Analysis: ANOVA, Mutual Information, Kurtosis & Skewness

## 🎯 Future Work
- Add `Deep Learning` models for classification.
- Explore `hyperparameter tuning` for clustering and machine learning models.
- Enhance `interactive visualizations` with tools like Plotly or Dash.

Feel free to fork this repository and explore the exciting world of data science with Iris! 🌸✨

Happy Coding! 💻
