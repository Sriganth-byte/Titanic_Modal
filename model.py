import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle

train_df = pd.read_csv('titanic train.csv')
test_df = pd.read_csv('titanic test.csv')

encoders = {}

def clean_and_transform(df, is_train=True):
    df = df.copy()

    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    if is_train:
        encoders['Sex'] = LabelEncoder().fit(df['Sex'])
    df['Sex'] = encoders['Sex'].transform(df['Sex'])

    if is_train:
        encoders['Embarked'] = LabelEncoder().fit(df['Embarked'])
    df['Embarked'] = encoders['Embarked'].transform(df['Embarked'])

    drop_cols = ['Name', 'Ticket', 'Cabin']
    if 'PassengerId' in df.columns:
        drop_cols.append('PassengerId')
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    return df

train_cleaned = clean_and_transform(train_df, is_train=True)

X = train_cleaned.drop('Survived', axis=1)
y = train_cleaned['Survived']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

val_preds = model.predict(X_val)
print("✅ Validation Accuracy: {:.2f}%".format(accuracy_score(y_val, val_preds) * 100))

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

print("🎉 Model and encoders saved successfully.")
