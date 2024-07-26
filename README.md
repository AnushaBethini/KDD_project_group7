Project Title :
NASA | Nearest Earth Objects

Team Members List :

1.Arun Kusuma

2.Anusha Bethini

3.Pavan Kovi

Project Introduction:
The dataset at hand focuses on Nearest Earth Objects (NEOs), which are celestial bodies that closely approach Earth. NASA has compiled observations of these objects from 1910 to 2024, resulting in a comprehensive dataset of 338,199 records. Some of these NEOs pose significant threats to Earth and are classified by NASA as "is_hazardous." This project aims to predict whether an NEO is hazardous or not using various attributes of the dataset.

Our project is primarily predictive, aiming to accurately forecast the "is_hazardous" status of NEOs. We will employ supervised learning techniques, given that we have a target variable ("is_hazardous"). Specifically, we will explore classification algorithms to determine the most effective model for this task.

Research Question:
The primary research question for this project is: What are the key factors associated with different diagnosis groups in Alzheimer's disease, and how do these factors interrelate?

Relevant Domain Information :
Alzheimer's Disease Overview: An article providing a comprehensive overview of Alzheimer's disease, including its symptoms, risk factors, and current research trends. Alzheimer's Association

MMSE and Its Use in Alzheimer's Diagnosis: An article detailing the Mini-Mental State Examination (MMSE) and its relevance in diagnosing cognitive impairments, particularly Alzheimer's disease. Journal of Alzheimer's Disease

Data Source and Description:
Kaggle - Alzheimer's Disease Dataset : https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset

This dataset includes various attributes related to patients with Alzheimer's disease:

Age: Age of the patient.
Gender: Gender of the patient.
BMI: Body Mass Index.
SystolicBP: Systolic Blood Pressure.
DiastolicBP: Diastolic Blood Pressure.
CholesterolTotal: Total Cholesterol levels.
MMSE: Mini-Mental State Examination score.
Diagnosis: Diagnosis group (e.g., healthy, mild cognitive impairment, Alzheimer's disease).
Data Understanding and EDA:
Here are the interpretations for the visualizations:

1. Age Distribution
Interpretation: The age distribution histogram shows how the ages of patients are distributed within the dataset. The kde (kernel density estimate) line provides a smoothed version of the histogram. Peaks in the histogram and KDE indicate the most common age ranges. If the distribution is skewed, it may suggest a bias towards a particular age group within the dataset.

2. Gender Distribution
Interpretation: The gender distribution count plot displays the number of patients of each gender. This helps to understand the gender balance in the dataset. A significant imbalance might suggest that the dataset is not fully representative of the general population.

3. MMSE Score Distribution
Interpretation: The MMSE (Mini-Mental State Examination) score distribution shows the frequency of various MMSE scores in the dataset. The KDE line provides a smoothed estimate of the distribution. This visualization helps to identify the common range of cognitive scores among patients, which can be useful for understanding the severity distribution of cognitive impairment.

4. Pairplot for Numeric Features
Interpretation: The pairplot shows pairwise relationships between selected numeric features (Age, BMI, Systolic Blood Pressure, Diastolic Blood Pressure, Total Cholesterol, and MMSE). The diagonal plots show the KDE of each individual feature, while the off-diagonal plots show scatter plots of feature pairs, providing insight into potential correlations or patterns between features.

5. Correlation Matrix
Interpretation: The correlation matrix heatmap visualizes the Pearson correlation coefficients between numeric features. Values close to 1 or -1 indicate strong positive or negative correlations, respectively. This helps to identify which features are related to each other, providing insights into potential multicollinearity or redundant features.

6. MMSE Scores by Diagnosis Group
Interpretation: The boxplot shows the distribution of MMSE scores across different diagnosis groups. The boxes represent the interquartile range (IQR), with the line inside the box indicating the median. Whiskers show the range, and any outliers are displayed as individual points. This visualization helps to compare cognitive function (as measured by MMSE) across different diagnostic categories.

7. Scatter Plot for SystolicBP vs. DiastolicBP
Interpretation: The scatter plot visualizes the relationship between systolic and diastolic blood pressure, with points colored by diagnosis group. This helps to identify any patterns or differences in blood pressure readings across different diagnosis categories. Clusters or trends can indicate how blood pressure varies with different health conditions.

Data Preparation:
To prepare the Alzheimer’s disease dataset for analysis and modeling, the following steps were taken:

1.Loading the Dataset:The dataset was loaded from a CSV file.

2.Exploratory Data Analysis (EDA):

•Displayed the first few rows, basic information, and basic statistics.

•Checked for missing values and examined the distribution of key features like Age, Gender, and MMSE scores.

3.Handling Missing Values:

•If any missing values were found, strategies such as imputation or removal were employed based on the context.

4.Data Cleaning:

•Ensured that the data types of columns were appropriate.

•Converted categorical variables to numeric codes if necessary using pd.get_dummies() or LabelEncoder.

5.Feature Selection:

•Selected relevant numeric features for pairplot analysis to reduce computational load.

•Used data.select_dtypes(include=['float64', 'int64']) to filter numeric columns for the correlation matrix.

6.Sampling:

•For the pairplot, sampled 10% of the data to make the computation more manageable.

Further steps in data preparation and modeling will be detailed in Deliverable 2.
