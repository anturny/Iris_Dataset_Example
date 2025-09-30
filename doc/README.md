# doc
- This folder is for extensive documentation if needed

## Statistical Analysis
For our statistical analysis, we see that (from MEDIA), the Iris 2D Raw Data Scatter and the Iris 3D Raw Data Scatter 3 showcases how setosa (blue and class 0) is not touching both versicolor and virginica in any way. This is why there is perfect precision and recall within this class because it is linearly seperable. Class 3 and 2 can be determined by their center of mass in the planes, but they will never be perfectly separable and hence the reason why there may be some bad predictions with classes 2 and 3. The bad predictions will then lead to false positives and false negatives.

Our classification report further supports our precision and recall by giving an accuracy and F1 score that showcases how well precision and recall are balanced, indicating a good model performance.

The heatmap in Iris Correlation Matrix showcases the standardized measure of the strength and direction of the linear relationship between variables. We see an extremely closely related score as 1 which indicates a perfect positive correlation where as one variable increases, so does the other. The scores below 0 headed towards negative one indicates a perfectly negative correlation where as one variable increases, the other decreases. 

The heatmap in Iris Covariance Matrix showcases the relationship between a pair of random variables where a change in one variable causes a change in another variable. It is similar to a correlation but is not the same because correlation measures the relationship between two variables while covariance measures how close the random variables seem to be related.