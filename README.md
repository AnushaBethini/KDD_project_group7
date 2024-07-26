# Project Title :
NASA | Nearest Earth Objects

# Team Members List :

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

## 1. Distribution of Absolute Maginitude by Hazardous Status
Interpretation: This histogram gives valuable details for our exploratory data analysis of astronomical objects. It compares the absolute magnitude distributions of hazardous and non-hazardous objects. The statistics show that non-hazardous objects are much more numerous and have a larger range of brightness levels. Hazardous objects, however uncommon in number, tend to cluster near brighter magnitudes. There is a large overlap in brightness between the two categories, indicating that absolute magnitude alone is insufficient to identify an object's hazard level. This image aids in understanding the relationship between brightness and potential hazard, emphasizing the necessity for more features or more complex models in our research to appropriately identify celestial objects as hazardous or non-hazardous.


## 2. Distribution of Estimated Diamter(Min) by Hazardous Status
Interpretation: The data visualization shows the distribution of predicted sizes for celestial objects that are classified as hazardous (orange) or non-hazardous (blue). The data is heavily skewed toward smaller objects, with the vast majority having diameters close to 0 km and frequencies approaching 1.9 million for the smallest size group. Non-hazardous objects appear to be more common overall, particularly in smaller size ranges. While the x-axis spans 35 km, objects larger than 5 km are uncommon, with very low frequencies detectable at this size.


## 3. Distribution of Estimated Diamter(Max) by Hardous Status
Interpretation: This graph depicts the distribution of estimated maximum diameters for celestial objects that are classified as dangerous (orange) or non-hazardous (blue). The data is strongly skewed towards smaller things, with the highest frequency reaching over 2 million for the smallest size category, which primarily includes non-hazardous objects. While the x-axis reaches up to 80 km, objects larger than 20 km are extremely rare, with only very low frequencies discernible. Non-hazardous objects appear to be substantially more abundant overall, with hazardous objects having a little greater proportion in larger size ranges than in the preceding "min diameter" figure.


## 4. Box Plot of Relative Velocity by Hazardous Status
Interpretation: This box plot offers important information about the relationship between relative velocity and hazard classification of astronomical objects. The apparent disparity in distributions of hazardous and non-hazardous objects implies that velocity is an important component in hazard evaluation. The existence of several outliers in both categories, particularly hazardous items, suggests that the dataset contains a wide range of velocities and potential complexity. This visualization aids in the identification of patterns and differences that may be important for future investigation, such as constructing predictive models or identifying the risk factors linked with certain celestial objects.


## 5. Box Plot of Miss Distance by Hazardous Status
Interpretation: This box plot sheds light on the relationship between miss distance and hazard classification of celestial objects. It demonstrates similarities in miss distance distributions across hazardous and non-hazardous objects, calling into question long-held notions regarding the function of miss distance in hazard determination. This leads an examination into other elements that influence classification. The graphic depicts the range and variability of miss distances within each category, which can help inform potential threshold considerations in risk assessment. It implies the necessity to investigate relationships with other variables and use complicated danger categorization criteria in celestial object tracking and risk assessment.


## 6. Count Plot of bitting Body by Hazardous Status
Interpretation: This count figure depicts the distribution of orbital bodies around Earth, grouped by their hazardous status. It clearly demonstrates that non-hazardous objects outnumber hazardous ones, giving a fast overview of the relative distribution of potentially deadly objects near Earth. For exploratory data research, this graphic provides as a starting point for further investigation into what characteristics contribute to an object being labeled as dangerous. It also raises concerns about the features and behaviors of the minority of items deemed potentially dangerous, necessitating further investigation into their attributes and trajectories.


## 7. Correlation Heatmap including Hazardous Status
Interpretation: This correlation heatmap gives useful information about the links between many aspects of celestial objects, including their dangerous status. It shows substantial correlations between certain physical qualities, such as size and magnitude, but lesser links between hazard status and other variables. Interestingly, the hazardous categorization appears to have a very modest link with miss distance, implying that proximity to Earth may not be the most important component in assessing an object's threat level. For exploratory data analysis, this image provides as a platform for further exploration into the complex interplay of parameters that lead to an object's classification as hazardous, perhaps guiding future research into risk assessment models for near-Earth objects.


# Data Preparation:
To prepare the data for modeling, we will undertake the following steps:

1. Data Cleaning:
.  Handle missing values, outliers, and inconsistencies.
2. Feature Engineering:
. Create new features or modify existing ones to improve model performance.
.  For example, averaging "estimated_diameter_min" and "estimated_diameter_max" to get a more robust measure of the diameter.
3. Encoding Categorical Variables:
. Convert categorical features like "orbiting_body" into numerical values using techniques such as one-hot encoding.
4. Normalization/Standardization:
. Scale numerical features to ensure that they contribute equally to the model.
5. Train-Test Split:
.  Divide the data into training and testing sets to evaluate model performance effectively.

Further steps in data preparation and modeling will be detailed in Deliverable 2.
