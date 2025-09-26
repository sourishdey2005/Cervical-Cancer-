Cervical Cancer Prediction using Machine Learning & Deep Learning

<img width="1582" height="620" alt="image" src="https://github.com/user-attachments/assets/f160029b-c99a-44d4-99df-c8af1bca6bf4" />

1. Project Overview

Cervical cancer remains a leading cause of cancer-related mortality among women globally. According to WHO (2020), cervical cancer ranks fourth among all cancers affecting women, with 604,000 new cases and 342,000 deaths annually. Persistent infection with high-risk Human Papillomavirus (HPV) strains is the primary cause of cervical carcinogenesis. Early detection is pivotal: it dramatically increases survival rates and reduces treatment complexity.

This project develops a sophisticated predictive framework combining classical machine learning, deep neural networks, and hybrid architectures to assess cervical cancer risk based on patient demographics, lifestyle factors, sexual history, and clinical test outcomes.

Objectives:

Predict biopsy-confirmed cervical cancer risk.

Handle complex real-world healthcare data (missing values, imbalance, noisy features).

Compare classical ML, deep learning (DNN, CNN), and hybrid models.

Optimize performance through hyperparameter tuning and regularization.

Provide model interpretability to highlight key risk factors.

2. Medical & Biological Background
2.1 Human Papillomavirus (HPV) & Pathogenesis

HPV is a DNA virus responsible for nearly all cervical cancer cases. High-risk strains (HPV 16, 18) integrate into host genomes, disrupting tumor suppressor pathways (p53 and Rb). Persistent infection leads to cervical intraepithelial neoplasia (CIN), which can progress to invasive cancer.

Clinical significance: Early identification of high-risk HPV infections allows timely intervention through screening and vaccination.

2.2 Risk Factors

Demographic: Age, education, parity, socio-economic status.

Behavioral: Smoking, contraceptive use, sexual activity, number of sexual partners.

Medical history: Previous STDs, abnormal cytology, immunosuppression.

Genetic predisposition: Mutations affecting cell cycle regulation.

2.3 Diagnostic Tests

Pap smear (Cytology): Detects precancerous changes in cervical cells.

Hinselmann & Schiller tests: Colposcopic staining methods highlighting abnormal epithelium.

Biopsy: Gold standard; histopathological confirmation of cervical lesions.

2.4 Importance of Early Detection

High survival rates (>90%) for localized disease.

Reduces the need for invasive treatment.

Mitigates the progression to advanced stages with poor prognosis.

3. Dataset Deep Dive
3.1 Overview

Dataset: 858 patient records, collected from Hospital Universitario de Caracas.
Features:

Feature Category	Examples
Demographics	Age, education, number of pregnancies
Lifestyle	Smoking, contraceptive use
STD History	HPV, HIV, syphilis
Medical Tests	Cytology, Hinselmann, Schiller, Biopsy

Target variable: Biopsy result (definitive diagnosis).

3.2 Statistical Analysis

Descriptive statistics: Mean, median, standard deviation for numeric variables.

Class imbalance: Positive biopsy results <10% of samples.

Feature correlations: STD history and smoking showed high correlation with biopsy outcomes.

3.3 Missingness & Bias

Missing values: Median imputation for numeric, mode for categorical features.

Biases: Skewed class distributions addressed using SMOTE.

Clinical relevance: Careful imputation ensures biologically plausible values.

3.4 Advanced Exploratory Data Analysis (EDA)

Histograms & KDEs to study skewness.

Pair plots to visualize feature interactions.

Boxplots to detect outliers.

Correlation heatmaps to highlight multicollinearity.

Dimensionality reduction: PCA & t-SNE to visualize separability of high-risk vs low-risk patients.





Mathematical & Algorithmic Foundation
4.1 Classical Machine Learning
<img width="600" height="286" alt="image" src="https://github.com/user-attachments/assets/8e7c6f28-14a8-4ee2-9479-89e4c03f2a86" />


<img width="528" height="405" alt="image" src="https://github.com/user-attachments/assets/2dd24a4e-d1f9-4dd1-bf9c-5bbd55ccfc24" />



4.2 Deep Learning

<img width="546" height="401" alt="image" src="https://github.com/user-attachments/assets/f27d1be0-ca78-4778-aa3f-c0fb1416b487" />


Hybrid DNN + CNN

Parallel branches capture complementary information.

Concatenated representations improve predictive performance.


5. Optimization & Hyperparameter Tuning

Optuna used for automated hyperparameter search.

Parameters tuned: learning rate, dropout rate, number of layers, units, max depth, tree count.
<img width="377" height="84" alt="image" src="https://github.com/user-attachments/assets/97861195-4d22-48e3-b9d5-320cba2453ef" />


Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint.


Evaluation Metrics

<img width="318" height="118" alt="image" src="https://github.com/user-attachments/assets/765115df-f33c-4fc5-a937-857b59d18c24" />



ROC-AUC: Area under true positive rate vs false positive rate curve.

PR-AUC: Area under precision-recall curve (especially relevant for imbalanced classes).

Confusion Matrix: Detailed TP, TN, FP, FN breakdown; critical for assessing false negatives (high-risk cases).



<img width="755" height="351" alt="image" src="https://github.com/user-attachments/assets/325ac85a-2c31-47f3-b707-62f640343781" />


Key insights:

Hybrid models and XGBoost excel due to ability to model non-linear interactions.

Feature importance analysis: STD history, smoking, age, and number of pregnancies are major predictors.

Clinically, minimizing false negatives is crucial (missing a high-risk patient can be life-threatening).

8. Deployment Considerations

MLOps pipeline: Preprocessing → Model training → Evaluation → Monitoring.

Cloud deployment: AWS SageMaker, GCP AI Platform, or Azure ML.

Model monitoring: Track drift, accuracy degradation, class imbalance over time.

Explainability: SHAP, LIME for feature contribution visualization.

Integration: Can be embedded into electronic health record (EHR) systems for real-time screening.

9. Future Research Directions

Multimodal learning: Combine imaging (Pap smears) and tabular data.

Federated learning: Privacy-preserving model training across multiple hospitals.

Transfer learning: Pretrained CNNs for histopathological images.

Early-stage prediction: Predict progression from CIN1 → CIN3.

Explainable AI: Incorporate domain-specific interpretations for clinical adoption.



Installation & Usage

# Install dependencies
pip install pandas numpy seaborn matplotlib plotly scikit-learn xgboost lightgbm tensorflow tensorflow-addons optuna

# Clone repo
git clone https://github.com/yourusername/Cervical_Cancer_Prediction_with_ML.git
cd Cervical_Cancer_Prediction_with_ML

# Launch Jupyter Notebook
jupyter notebook




11. Conclusion

This project demonstrates the application of machine learning and deep learning to predict cervical cancer risk using patient demographic, behavioral, and clinical data. By combining ensemble models (XGBoost) with deep architectures (Hybrid DNN+CNN), we achieved high precision and recall, making the framework suitable for potential clinical decision support systems.

Early detection remains vital: these models can assist healthcare providers, optimize screening resources, and save lives.

12. References

WHO: Cervical Cancer Statistics, 2020.

Kaggle: Cervical Cancer Risk Classification Dataset.

Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System."

Optuna: Hyperparameter Optimization Framework.

TensorFlow/Keras Documentation.
