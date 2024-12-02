import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.scrolled import ScrolledText
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

class HealthCareChatBot:
    def __init__(self, root):
        self.root = root
        self.root.title("HealthCare ChatBot")

        # Set the theme to 'superhero'
        self.style = ttk.Style(theme="superhero")

        self.user_name = ""
        self.symptoms_exp = []
        self.current_symptom = ""
        self.symptoms_given = []
        self.num_days = 0
        self.step = 0

        # GUI Layout with ttkbootstrap widgets
        self.label = ttk.Label(root, text="Welcome to HealthCare ChatBot!", font=("Arial", 14), bootstyle="primary")
        self.label.pack(pady=10)

        self.text_area = ScrolledText(root, wrap=tk.WORD, width=60, height=20, font=("Arial", 10), bootstyle="info")
        self.text_area.pack(pady=10)
        self.text_area.text.insert(tk.END, "Bot: Your Name?\n")
        self.text_area.text.config(state=tk.DISABLED)


        self.entry = ttk.Entry(root, font=("Arial", 12), width=50)
        self.entry.pack(pady=10)

        self.submit_btn = ttk.Button(root, text="Submit", command=self.process_input, bootstyle="success")
        self.submit_btn.pack(pady=10)

    def get_user_input(self):
        """Helper function to get input from user dynamically."""
        self.text_area.text.config(state=tk.NORMAL)
        self.text_area.insert(tk.END, "Waiting for your response...\n")
        self.text_area.text.config(state=tk.DISABLED)

        self.entry.focus()  # Bring focus to the input field
        self.root.wait_variable(self.response_var)  # Wait for user input
        return self.entry.get().strip()

    def tree_to_code(self, tree, feature_names):
        """Interactive function to traverse the decision tree."""
        tree_code = []
        current_node = 0  # Start at the root of the tree

        def recurse(node, depth=0):
            nonlocal current_node

            # Check if this is a leaf node
            if tree.tree_.feature[node] == -2:
                value = tree.tree_.value[node]
                return value.argmax()

            # Not a leaf node, get feature and threshold
            feature_index = tree.tree_.feature[node]
            threshold = tree.tree_.threshold[node]
            feature_name = feature_names[feature_index]

            # Ask user the question and decide direction
            self.bot_reply(f"Do you have {feature_name}? (yes/no)")
            user_input = self.get_user_input()  # Function to wait for and capture input

            if user_input == "yes":
                current_node = tree.tree_.children_left[node]
            else:
                current_node = tree.tree_.children_right[node]

            return recurse(current_node, depth + 1)

        prediction = recurse(current_node)
        return prediction


    def traverse_tree(question_map, user_input):
    # Example traversal logic based on user input
        if question_map["Symptom1"] <= 0.5:
            if question_map["Symptom2"] <= 1.5:
                return "Diagnosis A"
            else:
                return "Diagnosis B"
        else:
            return "Diagnosis C"


    def bot_reply(self, message):
        self.text_area.text.config(state=tk.NORMAL)
        self.text_area.insert(tk.END, f"Bot: {message}\n")
        self.text_area.text.config(state=tk.DISABLED)
        self.text_area.text.yview(tk.END)

    def process_input(self):
        """Main function to process input at various steps."""
        user_input = self.entry.get().strip()
        if not user_input:
            return  # Ignore empty inputs
        self.entry.delete(0, tk.END)

        # Display the user's input in the text area
        self.text_area.text.config(state=tk.NORMAL)
        self.text_area.insert(tk.END, f"You: {user_input}\n")
        self.text_area.text.config(state=tk.DISABLED)
        self.text_area.text.yview(tk.END)

        if self.step == 0:  # Asking for name
            self.user_name = user_input
            self.bot_reply(f"Hello, {self.user_name}! What symptom are you experiencing?")
            self.step += 1

        elif self.step == 1:  # Asking for symptom
            conf, cnf_dis = check_pattern(cols, user_input)
            if conf == 1:
                self.bot_reply(f"Did you mean: {', '.join(cnf_dis)}? Please select (e.g., 0).")
                self.current_symptom = cnf_dis
                self.step += 1
            else:
                self.bot_reply("Enter a valid symptom.")

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
                self.num_days = int(user_input)
                self.bot_reply(f"Noted. Let's ask some follow-up questions based on our analysis.")
                self.tree_to_code(clf1, cols)
                self.bot_reply("Based on your symptoms, let me give you a final diagnosis.")
                self.final_prediction()  # Proceed to final prediction
            except ValueError:
                self.bot_reply("Please enter a valid number.")


    def ask_follow_up(self):
        if self.symptoms_given:
            next_symptom = self.symptoms_given.pop(0)  # Remove the next symptom from the list
            self.bot_reply(f"Are you experiencing {next_symptom}? (yes/no)")
        else:
            self.final_prediction()

    def final_prediction(self):
        """Make the final diagnosis based on symptoms collected."""
        second_prediction = sec_predict(self.symptoms_exp)
        present_disease = second_prediction[0]

        self.bot_reply(f"You may have {present_disease}.")
        self.bot_reply(description_list[present_disease])
        self.bot_reply("Take the following precautions:")
        for i, precaution in enumerate(precautionDictionary[present_disease], 1):
            self.bot_reply(f"{i}. {precaution}")


# Start the GUI
if __name__ == "__main__":
    root = ttk.Window(themename="darkly")
    app = HealthCareChatBot(root)
    root.mainloop()
