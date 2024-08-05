# Project Title :
NASA | Nearest Earth Objects

# Team Members List :

1.Arun Kusuma

2.Anusha Bethini

3.Pavan Kumar Kovi

# Project Introduction:
The dataset at hand focuses on Nearest Earth Objects (NEOs), which are celestial bodies that closely approach Earth. NASA has compiled observations of these objects from 1910 to 2024, resulting in a comprehensive dataset of 338,199 records. Some of these NEOs pose significant threats to Earth and are classified by NASA as "is_hazardous." This project aims to predict whether an NEO is hazardous or not using various attributes of the dataset.

Our project is primarily predictive, aiming to accurately forecast the "is_hazardous" status of NEOs. We will employ supervised learning techniques, given that we have a target variable ("is_hazardous"). Specifically, we will explore classification algorithms to determine the most effective model for this task.

# Research Question:

How accurately can we predict whether a Nearest Earth Object (NEO) is hazardous based on its characteristics such as absolute magnitude, estimated diameter, relative velocity, and miss distance?

We aim to develop a predictive model that leverages the physical and orbital attributes of NEOs to classify them as hazardous or non-hazardous. The focus will be on identifying the key features that contribute to the prediction and evaluating the model's performance in accurately classifying the NEOs.

# Relevant Domain Information :
Nearest Earth Objects (NEOs) are asteroids and comets with orbits that bring them close to Earth's orbit .
Understanding NEOs is crucial for assessing potential collision threats and planning space missions .
This field combines astronomy, planetary science, and data analytics to study the dynamics of these celestial bodies.

[_NASA - Near Earth Object Program_](https://www.nasa.gov/mission_pages/asteroids/main/index.html) - This NASA webpage provides information about NEOs, including their monitoring, detection, and the risks they pose to Earth.

[_NASA Center for Near-Earth Object Studies_](https://cneos.jpl.nasa.gov/) - The Center for Near Earth Objects Studies(CNEOS) maintains a continuous watch for asteroids and comets that could approach Earth, providing data and predictions about NEOs.


# Data Source and Description:
[_Kaggle NASA | Nearest Earth Objects_](https://www.kaggle.com/datasets/ivansher/nasa-nearest-earth-objects-1910-2024)

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

### 1. Distribution of Absolute Maginitude
Interpretation: This histogram gives valuable details for our exploratory data analysis of astronomical objects. It compares the absolute magnitude distributions of hazardous and non-hazardous objects. The statistics show that non-hazardous objects are much more numerous and have a larger range of brightness levels. Hazardous objects, however uncommon in number, tend to cluster near brighter magnitudes. There is a large overlap in brightness between the two categories, indicating that absolute magnitude alone is insufficient to identify an object's hazard level. This image aids in understanding the relationship between brightness and potential hazard, emphasizing the necessity for more features or more complex models in our research to appropriately identify celestial objects as hazardous or non-hazardous.


### 2. Distribution of Estimated Diamter(Min)
Interpretation: The data visualization shows the distribution of predicted sizes for celestial objects that are classified as hazardous (orange) or non-hazardous (blue). The data is heavily skewed toward smaller objects, with the vast majority having diameters close to 0 km and frequencies approaching 1.9 million for the smallest size group. Non-hazardous objects appear to be more common overall, particularly in smaller size ranges. While the x-axis spans 35 km, objects larger than 5 km are uncommon, with very low frequencies detectable at this size.


### 3. Distribution of Estimated Diamter(Max)
Interpretation: This graph depicts the distribution of estimated maximum diameters for celestial objects that are classified as dangerous (orange) or non-hazardous (blue). The data is strongly skewed towards smaller things, with the highest frequency reaching over 2 million for the smallest size category, which primarily includes non-hazardous objects. While the x-axis reaches up to 80 km, objects larger than 20 km are extremely rare, with only very low frequencies discernible. Non-hazardous objects appear to be substantially more abundant overall, with hazardous objects having a little greater proportion in larger size ranges than in the preceding "min diameter" figure.


### 4. Box Plot of Relative Velocity
Interpretation: This box plot offers important information about the relationship between relative velocity and hazard classification of astronomical objects. The apparent disparity in distributions of hazardous and non-hazardous objects implies that velocity is an important component in hazard evaluation. The existence of several outliers in both categories, particularly hazardous items, suggests that the dataset contains a wide range of velocities and potential complexity. This visualization aids in the identification of patterns and differences that may be important for future investigation, such as constructing predictive models or identifying the risk factors linked with certain celestial objects.


### 5. Box Plot of Miss Distance
Interpretation: This box plot sheds light on the relationship between miss distance and hazard classification of celestial objects. It demonstrates similarities in miss distance distributions across hazardous and non-hazardous objects, calling into question long-held notions regarding the function of miss distance in hazard determination. This leads an examination into other elements that influence classification. The graphic depicts the range and variability of miss distances within each category, which can help inform potential threshold considerations in risk assessment. It implies the necessity to investigate relationships with other variables and use complicated danger categorization criteria in celestial object tracking and risk assessment.


### 6. Count Plot of bitting Body
Interpretation: This count figure depicts the distribution of orbital bodies around Earth, grouped by their hazardous status. It clearly demonstrates that non-hazardous objects outnumber hazardous ones, giving a fast overview of the relative distribution of potentially deadly objects near Earth. For exploratory data research, this graphic provides as a starting point for further investigation into what characteristics contribute to an object being labeled as dangerous. It also raises concerns about the features and behaviors of the minority of items deemed potentially dangerous, necessitating further investigation into their attributes and trajectories.


### 7. Correlation Heatmap including Hazardous Status
Interpretation: This correlation heatmap gives useful information about the links between many aspects of celestial objects, including their dangerous status. It shows substantial correlations between certain physical qualities, such as size and magnitude, but lesser links between hazard status and other variables. Interestingly, the hazardous categorization appears to have a very modest link with miss distance, implying that proximity to Earth may not be the most important component in assessing an object's threat level. For exploratory data analysis, this image provides as a platform for further exploration into the complex interplay of parameters that lead to an object's classification as hazardous, perhaps guiding future research into risk assessment models for near-Earth objects.


# **Data Preparation:**

**1. Data Cleaning:** - Manage Missing Values Missing data was imputed, maybe with the mean for numerical features.

   Outliers and Inconsistencies: Outliers were detected and rectified, potentially through capping or transformation, while data inconsistencies were corrected to assure data quality.

**2. Feature Engineering**: - 
Creating New Features: New features were included to improve model performance. For example, we averaged the "estimated_diameter_min" and "estimated_diameter_max" to get a more reliable assessment of the diameter.
   Modifying Existing Features: Existing features were modified or combined to provide more insights and increase the model's prediction potential.

**3. Encoding Categorical Variables:** 
One-Hot Encoding. Categorical features such as "orbiting_body" were translated into numerical values via one-hot encoding, which generates binary columns for each category, allowing the model to accurately comprehend categorical data.

**4. Normalization/Standardization:** 

**Scaling of Numerical Features:** 
Numerical features were scaled so that they contributed equally to the model. Min-Max scaling and standardization (z-score normalization) were used to bring all features to a similar scale.

**5. Train-Test Split:**

Data Division The dataset was separated into training and testing sets to effectively assess model performance. This aids in determining how effectively the model generalizes to new, unseen data.

# **Modeling:**

Two or more modeling approaches were investigated, including the usage of PyCaret. Here are the steps.

**1. Model Selection with PyCaret:**

   PyCaret's 'compare_models' function was used to analyze and compare several models.
   - The best model chosen was 'ExtraTreesClassifier', an ensemble method based on random forests that builds numerous decision trees and integrates their results to produce more accurate and stable predictions.

**2. Other Models Explored:**

   K-Nearest Neighbors (KNN): An instance-based learning technique that determines a sample's class based on the majority of its k-nearest neighbors.
   Logistic Regression: A linear model for binary classification tasks that predicts the likelihood of a class label depending on input characteristics.
   Random The forest: An ensemble approach that creates numerous decision trees during training and returns the mode of the classes for classification.
   
# **Evaluation:**

The evaluation metrics and outcomes for the models are as follows:

**1. K-Nearest Neighbors (KNN):** Cross-validation. The KNN model performed well, with an accuracy of 89.96% ± 0.0074. However, there was some variability as evidenced by the standard deviation.

**2. Logistic Regression:** Cross-validation accuracy: 0.8710 ± 0.0039.
   Logistic Regression had a somewhat lower accuracy of roughly 87.10%, but it was quite consistent, as evidenced by the decreased standard deviation.

**3. Random Forest's:** Cross-validation accuracy is 0.9888 ± 0.0017.
   Test Accuracy: 0.9884. The Random Forest model outperformed expectations, obtaining an accuracy of approximately 98.88% in cross-validation and 98.84% on the test set. This high performance indicates that it was the most effective model for the dataset.

# **Conclusion/Results:**

The investigation revealed that Random Forest was the best-performing model, demonstrating significant predictive ability.
PyCaret's ExtraTreesClassifier performed similarly to the Random Forest findings.

These findings show the efficacy of ensemble approaches for this specific dataset, most likely due to their ability to prevent overfitting and enhance prediction accuracy by merging numerous decision trees.

# **Known Issues:**

Several potential concerns were discovered that may impair the model's performance and interpretation:

**1. Potential Bias:** The dataset's class imbalance may result in biased model performance. For example, if one class is much more common than others, the model may become biased in forecasting the majority class.

**2. Missing Data:** The imputation approach utilized (mean imputation) may not be optimal for all attributes. More complex methods, such as predictive model-based imputation or feature relationship analysis, may produce superior results.

**3. Encoding:** If there are any ordinal relationships, label encoding may not accurately reflect them. For example, if categories have a natural order, one-hot encoding or ordinal encoding may be better suited to maintaining these relationships.
