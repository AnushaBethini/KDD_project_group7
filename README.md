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


# Data Preparation

1. **Data Cleaning**: 

**Missing Values:** The dataset was loaded into a pandas DataFrame, and any rows with missing values were eliminated with 'df.dropna()'. This guarantees that the dataset used for modeling is full and contains no missing information that could impair the model's performance.

2. **Feature Engineering:** involves creating new features. A new feature, 'estimated_diameter_avg', was produced by averaging the columns 'estimated_diameter_min' and 'estimated_diameter_max'. This new feature provides a more reliable estimation of the asteroid's diameter, which could be an essential predictor of whether an asteroid is hazardous or not.

**3. Encoding Categorical Variables: **
 **Label Encoding**: The target variable 'is_hazardous' was evaluated to determine whether it was categorical (non-numeric). If it was, the 'LabelEncoder' function from scikit-learn was used to transform it to numerical values. This phase is critical for algorithms that need numerical input.

4. **Normalization/Standardization**: - **Scaling Features** The numerical features were scaled with 'StandardScaler'. This phase guarantees that all features contribute evenly to the model and enhances the convergence of gradient-based methods. Standardization changes the data to have a mean of zero and a standard deviation of one.

5. **Train-Test Split**: 
 **Data Splitting** The dataset was divided into training and testing sets with an 80/20 split ratio. This allows you to evaluate the model's performance on previously unseen data, which gives you a better idea of its generalization capacity.

# Modeling

1) **Method 1: Scikit-learn Models**:
    **Model selection**: Several scikit-learn models were tried, including:
    **RandomForestClassifier**: An ensemble method for creating numerous decision trees and combining them to get a more accurate and stable forecast.
    **LogisticRegression**: A linear model designed for binary classification situations.
    **GradientBoostingClassifier**: An ensemble approach for building models successively, with each model attempting to repair faults produced by the prior one.
    **KNeighborsClassifier**: A straightforward, instance-based learning technique that predicts a label based on the majority label of its k-nearest neighbors.
    **Training and Evaluation**: Each model was trained using scaled training data and tested using scaled test data. The performance criteria employed for evaluation included classification reports, ROC-AUC scores, and confusion matrices.

2. **Method 2: 
PyCaret**: 
**Automated Machine Learning**: PyCaret, an open-source, low-code machine learning library in Python, was used to compare multiple machine learning methods automatically. PyCaret's 'compare_models' function ranks models according to performance criteria, making it simple to choose the best model.
**Model comparison**: PyCaret's top model was compared to scikit-learn models to find which model performed best.

# Evaluation.

1. **Scikit-learn Models**: 
 **Classification Reports**: Detailed reports on precision, recall, f1-score, and support per class.
 **ROC-AUC Scores**: The area under the ROC curve was used to assess the model's ability to differentiate across classes. A higher ROC-AUC score suggests improved model performance.
 **Confusion Matrices**: Matrixes that display the true positive, true negative, false positive, and false negative counts. These aid in understanding the types of faults that the model makes.

2. PyCaret: 
**Model Ranking**: PyCaret's 'compare_models' function ranked models according on their performance. The top-performing model was then assessed and compared to the best scikit-learn model.

# Conclusion/Results
 
**Best Model**: The model with the highest ROC-AUC value was deemed the best. This model has the best capacity to appropriately distinguish hazardous and non-hazardous asteroids.
**Insights**: Key findings include the significance of good feature engineering and data pretreatment. The 'estimated_diameter_avg' feature, as well as feature normalization, considerably improved model performance.
**PyCaret Utility**: PyCaret proved to be a useful tool for quickly comparing different models and picking the best one, saving time and effort during the modeling process.

**Scikit-learn**: The model with the highest ROC-AUC score was Random Forest, among other models including Decision Tree and Logistic Regression.

**PyCaret**: The model with the highest ROC-AUC score was ExtraTreesClassifier.

# Known Issues

1. **Data Quality**: - The dataset may have outliers or inconsistencies that were not fully handled. Future generations should involve more rigorous data cleansing and exploration to detect and address such issues.

2. **Model bias**: - There may be biases in the predictors or the target variable. Monitoring and mitigating these biases is critical to ensuring fair and impartial predictions.

3. **Reporting**: Detailed and transparent reporting on model performance and comparison metrics is essential. This ensures that the analysis is reproducible and that the results are reliable.
