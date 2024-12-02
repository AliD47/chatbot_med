import tkinter as tk
from tkinter import scrolledtext
import re
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.svm import SVC
from sklearn import preprocessing
import csv

# Load datasets
training = pd.read_csv('Training.csv')
testing = pd.read_csv('Testing.csv')
data = pd.concat([training, testing], ignore_index=True)
cols = training.columns[:-1]

reduced_data = training.groupby(training['prognosis']).max()

# Preprocessing and training models
x = training[cols]
y = training['prognosis']
le = preprocessing.LabelEncoder()
le.fit(y)
y_encoded = le.transform(y)

clf1 = DecisionTreeClassifier().fit(x, y_encoded)
model = SVC().fit(x, y_encoded)

severityDictionary=dict()
description_list = dict()
precautionDictionary=dict()

symptoms_dict = {}

for index, symptom in enumerate(x):
       symptoms_dict[symptom] = index

def calc_condition(exp,days):
    sum=0
    for item in exp:
         sum=sum+severityDictionary[item]
    if((sum*days)/(len(exp)+1)>13):
        print("You should take the consultation from doctor. ")
    else:
        print("It might not be that bad but you should take precautions.")


def getDescription():
    global description_list
    with open('symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description={row[0]:row[1]}
            description_list.update(_description)

def getSeverityDict():
    global severityDictionary
    with open('symptom_severity.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction={row[0]:int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass

def getprecautionDict():
    global precautionDictionary
    with open('symptom_precaution.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec={row[0]:[row[1],row[2],row[3],row[4]]}
            precautionDictionary.update(_prec)


# Logic Functions
def check_pattern(dis_list, inp):
    inp = inp.replace(' ', '_')
    pattern = f"{inp}"
    regexp = re.compile(pattern)
    pred_list = [item for item in dis_list if regexp.search(item)]
    return (1, pred_list) if len(pred_list) > 0 else (0, [])

def sec_predict(symptoms_exp):
    symptoms_dict = {symptom: index for index, symptom in enumerate(cols)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[[symptoms_dict[item]]] = 1
    pred = model.predict(pd.DataFrame([input_vector], columns=cols))
    return le.inverse_transform(pred)

# Initialize data
getSeverityDict()
getDescription()
getprecautionDict()

# GUI Application
class HealthCareChatBot:
    def __init__(self, root):
        self.root = root
        self.root.title("HealthCare ChatBot")
        self.user_name = ""
        self.symptoms_exp = []
        self.current_symptom = ""
        self.symptoms_given = []
        self.num_days = 0
        self.step = 0

        # GUI Layout
        self.label = tk.Label(root, text="Welcome to HealthCare ChatBot!", font=("Arial", 14))
        self.label.pack(pady=10)

        self.text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=20, font=("Arial", 10))
        self.text_area.pack(pady=10)
        self.text_area.insert(tk.END, "Bot: Your Name?\n")
        self.text_area.config(state=tk.DISABLED)

        self.entry = tk.Entry(root, font=("Arial", 12), width=50)
        self.entry.pack(pady=10)

        self.submit_btn = tk.Button(root, text="Submit", command=self.process_input, font=("Arial", 12))
        self.submit_btn.pack(pady=10)
    
    def tree_to_code(self, tree, feature_names):
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        symptoms_present = []
        def recurse(node, depth):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                self.bot_reply(f"Are you experiencing {name}? (yes/no)")
                user_input = self.get_user_response()
                if user_input.lower() == "yes":
                    symptoms_present.append(name)
                    recurse(tree_.children_right[node], depth + 1)
                else:
                    recurse(tree_.children_left[node], depth + 1)
            else:
                disease = self.print_disease(tree_.value[node])
                self.bot_reply(f"You may have {disease}.")
                self.finalize_diagnosis(disease, symptoms_present)
        recurse(0, 1)


    def bot_reply(self, message):
        self.text_area.config(state=tk.NORMAL)
        self.text_area.insert(tk.END, f"Bot: {message}\n")
        self.text_area.config(state=tk.DISABLED)
        self.text_area.yview(tk.END)

    def process_input(self):
        user_input = self.entry.get().strip().lower()
        self.entry.delete(0, tk.END)

        if self.step == 0:  # Asking for name
            self.user_name = user_input
            self.bot_reply(f"Hello, {self.user_name}! What symptom are you experiencing?")
            self.step += 1

        elif self.step == 1:  # After the first symptom is input
            self.tree_to_code(clf1, cols)
            self.step += 1


        elif self.step == 2:  # Confirming symptom selection
            try:
                index = int(user_input)
                self.current_symptom = self.current_symptom[index]
                self.bot_reply(f"Okay. From how many days have you been experiencing {self.current_symptom}?")
                self.step += 1
            except (ValueError, IndexError):
                self.bot_reply("Please enter a valid number corresponding to the symptom.")

        elif self.step == 3:  # Asking for duration
            try:
                self.num_days = int(user_input)  # Ensure valid number
                self.bot_reply(f"Noted. Let me ask a few more questions related to {self.current_symptom}.")
        
                # Append the initial symptom
                if self.current_symptom not in self.symptoms_exp:
                    self.symptoms_exp.append(self.current_symptom)
        
                # Use symptoms_dict to get the index of the symptom
                transformed_symptom = symptoms_dict.get(self.current_symptom, None)
                if transformed_symptom is None:
                    self.bot_reply(f"Error: {self.current_symptom} not recognized. Please try again.")
                    return
        
                # Ensure the symptom index is valid in reduced_data
                if transformed_symptom not in reduced_data.index:
                    self.bot_reply("No data found for the given symptom. Proceeding with diagnosis.")
                    self.final_prediction()
                    return
        
                # Fetch related symptoms from reduced_data
                symptom_row = reduced_data.loc[transformed_symptom]
                self.symptoms_given = symptom_row.index[symptom_row.nonzero()].tolist()
        
                # Transition to follow-up symptom questions
                if self.symptoms_given:
                    self.ask_follow_up()
                    self.step = 4
                else:
                    self.bot_reply("No further related symptoms found. Proceeding with diagnosis.")
                    self.final_prediction()
        
            except ValueError:
                self.bot_reply("Please enter a valid number.")
        


        elif self.step == 4:  # Follow-up symptoms
            if user_input in ["yes", "no"]:
                if user_input == "yes" and self.symptoms_given:
                    self.symptoms_exp.append(self.symptoms_given[0])
                self.symptoms_given = self.symptoms_given[1:]
                self.ask_follow_up()
            else:
                self.bot_reply("Please answer with 'yes' or 'no'.")

    def ask_follow_up(self):
        if self.symptoms_given:
            next_symptom = self.symptoms_given.pop(0)  # Remove the next symptom from the list
            self.bot_reply(f"Are you experiencing {next_symptom}? (yes/no)")
        else:
            self.final_prediction()

    def final_prediction(self):
        second_prediction = sec_predict(self.symptoms_exp)
        present_disease = second_prediction[0]

        self.bot_reply(f"You may have {present_disease}.")
        self.bot_reply(description_list[present_disease])
        self.bot_reply("Take the following precautions:")
        for i, precaution in enumerate(precautionDictionary[present_disease], 1):
            self.bot_reply(f"{i}. {precaution}")


# Start the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = HealthCareChatBot(root)
    root.mainloop()
