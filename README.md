# Consumer_Complaints_Machine_Learning

## Abstract
This project introduces the Consumer Feedback Insight & Prediction Platform, a system leveraging machine learning to analyze the extensive Consumer Financial Protection Bureau (CFPB) Complaint Database, a publicly available resource exceeding 4.9 GB in size. 
The platform itself utilizes machine learning models to predict two key aspects of complaint resolution: the timeliness of company responses and the nature of those responses (e.g., closed, closed with relief etc.). 
Furthermore, the platform employs Latent Dirichlet Allocation (LDA) to delve deeper, uncovering common themes within complaints and revealing underlying trends and consumer issues. This comprehensive approach empowers both consumers and regulators. Consumers gain valuable insights into potential response wait times, while regulators can utilize the platform's findings to identify areas where companies may require further scrutiny regarding their complaint resolution practices.

## Project Workflow
To develop the Prediction Platform, we adopted a structured workflow encompassing data preparation, model training, and evaluation. </br>
- Initially, we acquired the CFPB Complaint Database and performed meticulous data preprocessing. This involved removing irrelevant columns, handling missing values, and employing frequency encoding for categorical features like company, issue, and state. This reduced data complexity and optimized it for machine learning algorithms.
- Subsequently, the preprocessed data was divided into training (70%) and test (30%) sets. The training set provided the foundation for model training.
- We utilized appropriate machine learning algorithms to predict two key aspects of complaint resolution: timely company response and the nature of the company’s complaint resolution.
- Feature importance analysis was then conducted to understand the most influential factors for each prediction.
- To assess model performance, we employed metrics tailored to the specific prediction tasks.
  - For the binary classification task of predicting timely company response, we utilized Area Under the Curve (AUC) in addition to the precision and recall.
  - For the multiclassification task of predicting the nature of the company response (e.g., closed, closed with explanation, etc.), we evaluated the models using precision and recall assessing their ability to accurately classify different outcomes.

### Data Challenges and Preprocessing Techniques
Our raw complaint data presented two key challenges:
  1. Imbalanced target variables: Target variables for both timely response prediction and company response prediction exhibited significant class imbalances. Oversampling and, for company response, under-sampling techniques were employed to create more balanced training sets.
  2. A high-cardinality feature: the "Company" feature with its 7,000 unique values required attention. Frequency encoding tackled this challenge by transforming company names into numerical values based on their frequency within the dataset, effectively reducing complexity and improving model performance.

### Models Explained
This section explores the use of machine learning to predict two key consumer complaint outcomes: 
1. Timely responses 
2. The nature of the company response.
   
#### Predicting Timely Responses: Binary Classification
Goal: Predicting whether a company will respond to a complaint within a designated timeframe.
Features: Company Name, Product Category, Complaint Issue, State of Complaint Origin and Date Sent to Company.
The target variable: "timely response" (Yes/No).
Machine learning algorithms used: 
  - Gradient Boosted Trees (GBT)
  - Support Vector Machines (SVM)
  - Logistic Regression (LR).

Results:
- Gradient Boosted Trees (GBT) achieved the best overall performance across all metrics. 
- While Logistic Regression demonstrated a high recall for "Yes" responses (indicating a good ability to identify timely responses), it missed many "No" responses. 
- Support Vector Machine (SVM) exhibited comparable performance to Logistic Regression. 

#### Predicting Compony Responses: Multiclass Classification
Goal: Predicting the nature of a company's response to a consumer complaint using a multiclass classification approach. This information can be valuable for understanding potential complaint resolution pathways and informing customer service strategies. These categories can include outcomes such as "closed with explanation," "closed with monetary relief," or "closed with relief." This classification task is considered multiclass as there are more than two possible response categories.
Features: Company Name, Product Category and Complaint Issue.
The target variable: "company_response" with 8 unique categories:
  • Closed with explanation
  • Closed with non-monetary relief
  • In progress
  • Closed with monetary relief
  • Closed without relief
  • Closed
  • Untimely response
  • Closed with relief
Machine learning algorithms used: 
  - Random Forest (RF)
  - Decision Tree (DT)

Results:
- Notably, both models exhibited high recall scores (93% for RF and 95% for DT) for identifying instances of "closed with relief," indicating a robust capability for recognizing this outcome. For "closed with monetary relief," both RF (54%) and DT (57%) demonstrated a discernible capacity for identification, albeit with lower recall scores compared to "closed with relief."
- An interesting finding is the trade-off observed in classifying "closed with explanation." While RF achieved a high recall score (88%), its precision (83%) suggests a higher rate of false positives compared to DT (83% recall, 90% precision). This implies that DT might miss some instances of "closed with explanation," but it produces more accurate classifications overall for this category.
- Both models displayed minimal computational time, ensuring efficient processing. However, considering the slight advantage in recall scores and precision for key categories like "closed with explanation," the Decision Tree model emerged as the slightly more suitable choice for this multiclass classification task.

#### Latent Dirichlet Allocation (LDA) for Topics Discovery
Goal: To uncover underlying thematic structures within consumer complaint narratives. This approach, known as topic modeling, offers valuable insights into prevalent consumer financial concerns and areas requiring potential regulatory focus. LDA acts as a machine learning technique that identifies latent topics within a vast collection of documents, in this case, the 1.7 million consumer complaint narratives. By analyzing the most frequent words and phrases associated with each topic, we can gain a deeper understanding of the challenges consumers face.

