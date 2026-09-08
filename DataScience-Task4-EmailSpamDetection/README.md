# Task 4 - Email Spam Detection

## Objective
Build a machine learning model that classifies SMS/email messages as **spam** or **ham (not spam)** based on their text content.

## Dataset
SMS Spam Collection Dataset — 5,572 SMS messages labeled as spam or ham.

## Approach
1. Loaded and cleaned the dataset (dropped unused columns, renamed for clarity)
2. Converted labels to numeric form: ham = 0, spam = 1
3. Split data into training (80%) and test (20%) sets
4. Converted message text into numerical features using **TF-IDF Vectorization**
5. Trained a **Multinomial Naive Bayes** classifier
6. Evaluated performance using accuracy, precision, recall, and a confusion matrix

## Results
- **Accuracy:** 96.2%
- **Precision:** 1.00
- **Recall:** 0.72
- Confusion Matrix:

[[965 0]
[ 42 108]]


The model achieves perfect precision — no legitimate messages were ever misclassified as spam — while correctly catching 72% of actual spam messages.

## Tools Used
- Python
- pandas
- scikit-learn (TfidfVectorizer, MultinomialNB, train_test_split, metrics)

## How to Run
1. Clone this repository
2. Open `email_spam_detection.ipynb` in Jupyter Notebook or VS Code
3. Run all cells in order

## Author
Navyasri Pulyala — Data Science Intern, Oasis Infobyte