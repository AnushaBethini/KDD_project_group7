# Project Title :
NASA | Nearest Earth Objects

Team Members List :

1.Arun Kusuma

2.Anusha Bethini

3.Pavan Kovi

# Project Introduction:
The dataset at hand focuses on Nearest Earth Objects (NEOs), which are celestial bodies that closely approach Earth. NASA has compiled observations of these objects from 1910 to 2024, resulting in a comprehensive dataset of 338,199 records. Some of these NEOs pose significant threats to Earth and are classified by NASA as "is_hazardous." This project aims to predict whether an NEO is hazardous or not using various attributes of the dataset.

Our project is primarily predictive, aiming to accurately forecast the "is_hazardous" status of NEOs. We will employ supervised learning techniques, given that we have a target variable ("is_hazardous"). Specifically, we will explore classification algorithms to determine the most effective model for this task.

# Research Question:

How accurately can we predict whether a Nearest Earth Object (NEO) is hazardous based on its characteristics such as absolute magnitude, estimated diameter, relative velocity, and miss distance?

# Relevant Domain Information :
Nearest Earth Objects (NEOs) are asteroids and comets with orbits that bring them close to Earth's orbit .
Understanding NEOs is crucial for assessing potential collision threats and planning space missions .
This field combines astronomy, planetary science, and data analytics to study the dynamics of these celestial bodies.

# Data Source and Description:
Kaggle NASA | Nearest Earth Objects: https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset https://www.kaggle.com/datasets/ivansher/nasa-nearest-earth-objects-1910-2024 Links to an external site.

The dataset originates from NASA's observations of Nearest Earth Objects spanning from 1910 to 2024. It includes the following attributes:

1. neo_id: Unique Identifier for each Asteroid
2. name: Name given by NASA
3. absolute_magnitude: Describes intrinsic luminosity
4. estimated_diameter_min: Minimum Estimated Diameter in Kilometres
5. estimated_diameter_max: Maximum Estimated Diameter in Kilometres
6. orbiting_body: Planet that the asteroid orbits
7. relative_velocity: Velocity Relative to Earth in Kmph
8. miss_distance: Distance in Kilometres missed
9. is_hazardous: Boolean feature that indicates whether the asteroid is harmful or not
  The goal is to use these features to predict the "is_hazardous" attribute
# Data Understanding and EDA:
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

# Data Preparation:
To prepare the data for modeling, we will undertake the following steps:

1. Data Cleaning:
   .  Handle missing values, outliers, and inconsistencies.
2. Feature Engineering:
   . Create new features or modify existing ones to improve model performance.
   . For example, averaging "estimated_diameter_min" and "estimated_diameter_max" to get a more robust measure of the diameter.
3. Encoding Categorical Variables:
   .  Convert categorical features like "orbiting_body" into numerical values using techniques such as one-hot encoding.
4. Normalization/Standardization:
   .Scale numerical features to ensure that they contribute equally to the model.
5. Train-Test Split: Divide the data into training and testing sets to evaluate model performance effectively.

Further steps in data preparation and modeling will be detailed in Deliverable 2.
