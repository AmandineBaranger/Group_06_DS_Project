#import pandas to manipulate the tables
import pandas as pd
#import the tool that applies different steps to different columns
from sklearn.compose import ColumnTransformer
#import the tool that fills missing values
from sklearn.impute import SimpleImputer
#import the pipeline to chain steps together
from sklearn.pipeline import Pipeline
#import the encoder for categories and the scaler for numbers
from sklearn.preprocessing import OneHotEncoder, StandardScaler

#columns that are codes and not real numbers
ID_COLS = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]
#columns that contain ICD-9 diagnosis codes
DIAG_COLS = ["diag_1", "diag_2", "diag_3"]
#columns where None means that the test was not done
TEST_COLS = ["max_glu_serum", "A1Cresult"]
#all the medication columns of the original dataset (only the ones still present are used)
MED_COLS = ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"]


#function that turns one ICD-9 code into a big diagnosis group
def icd9_to_group(code):
    #missing codes get their own group
    if pd.isna(code):
        #return the missing group
        return "Missing"
    #make sure the code is a text
    code = str(code)
    #codes starting with V or E are supplementary codes
    if code.startswith(("V", "E")):
        #return the other group
        return "Other"
    #try to convert the code to a number
    try:
        #keep the number so we can compare it with ranges
        number = float(code)
    #if the code is not a number
    except ValueError:
        #return the other group
        return "Other"
    #a text like nan is also a missing code
    if pd.isna(number):
        #return the missing group
        return "Missing"
    #diabetes codes are 250.xx
    if 250 <= number < 251:
        #return the diabetes group
        return "Diabetes"
    #circulatory system codes
    if 390 <= number <= 459 or int(number) == 785:
        #return the circulatory group
        return "Circulatory"
    #respiratory system codes
    if 460 <= number <= 519 or int(number) == 786:
        #return the respiratory group
        return "Respiratory"
    #digestive system codes
    if 520 <= number <= 579 or int(number) == 787:
        #return the digestive group
        return "Digestive"
    #injury and poisoning codes
    if 800 <= number <= 999:
        #return the injury group
        return "Injury"
    #musculoskeletal system codes
    if 710 <= number <= 739:
        #return the musculoskeletal group
        return "Musculoskeletal"
    #genitourinary system codes
    if 580 <= number <= 629 or int(number) == 788:
        #return the genitourinary group
        return "Genitourinary"
    #neoplasm codes
    if 140 <= number <= 239:
        #return the neoplasms group
        return "Neoplasms"
    #everything else
    return "Other"


#function that turns an age bin like [70-80) into its middle value
def age_to_number(age_bin):
    #take the lower bound of the bin as a number
    lower = int(age_bin[1:age_bin.index("-")])
    #return the middle of the bin
    return lower + 5


#function that prepares the features row by row (nothing is learned from the data here)
def prepare_features(df):
    #work on a copy so the original table is not changed
    df = df.copy()
    #loop over the id columns
    for col in ID_COLS:
        #convert the codes to text so they are treated as categories
        df[col] = df[col].astype(str)
    #loop over the diagnosis columns
    for col in DIAG_COLS:
        #replace each ICD-9 code by its big group
        df[col] = df[col].apply(icd9_to_group)
    #loop over the test result columns
    for col in TEST_COLS:
        #the text None (or a missing value) means that the test was not done so we give it a clear name
        df[col] = df[col].fillna("None").replace("None", "NotMeasured")
    #create a numeric age with the middle of each bin
    df["age_num"] = df["age"].apply(age_to_number)
    #drop the original text age
    df = df.drop(columns=["age"])
    #keep only the medication columns that are still in the table
    meds = [c for c in MED_COLS if c in df.columns]
    #count how many medications the patient takes
    df["n_meds_used"] = (df[meds] != "No").sum(axis=1)
    #count how many medications had their dose changed (Up or Down)
    df["n_meds_changed"] = df[meds].isin(["Up", "Down"]).sum(axis=1)
    #return the prepared table
    return df


#function that splits the columns into numeric and categorical
def get_column_types(X):
    #numeric columns
    num_cols = X.select_dtypes(include="number").columns.tolist()
    #all the other columns are categorical
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    #return both lists
    return num_cols, cat_cols


#function that builds the preprocessing (it is NOT fitted here)
def build_preprocessor(num_cols, cat_cols):
    #pipeline for the numeric columns
    numeric_pipe = Pipeline([
        #fill missing numbers with the median of the training data
        ("imputer", SimpleImputer(strategy="median")),
        #standardize to mean 0 and standard deviation 1
        ("scaler", StandardScaler()),
    ])
    #pipeline for the categorical columns
    categorical_pipe = Pipeline([
        #fill missing categories with the most frequent one
        ("imputer", SimpleImputer(strategy="most_frequent")),
        #one-hot encoding with a bucket for rare categories (less than 0.5 percent)
        ("encoder", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.005)),
    ])
    #combine both pipelines in one transformer
    preprocessor = ColumnTransformer(
        #list of (name, transformer, columns)
        [("num", numeric_pipe, num_cols), ("cat", categorical_pipe, cat_cols)],
        #always return a dense array
        sparse_threshold=0,
    )
    #return the unfitted preprocessor
    return preprocessor
