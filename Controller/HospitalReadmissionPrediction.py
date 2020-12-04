from PyQt5 import uic,QtWidgets
from PyQt5.QtWidgets import QApplication,QMainWindow
from PyQt5.QtWidgets import QFileDialog
import sys
from PyQt5 import QtGui
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import accuracy_score
class mainForm(QMainWindow):

    def __init__(self):
        super(mainForm,self).__init__()
        uic.loadUi('../View/HosptialReadmissionPrediction.ui',self)
        self.setWindowIcon(QtGui.QIcon('../img/redcrossimage.jpg'))
        self.pushButton_predictNN.clicked.connect(self.predictResultWithNN)
        self.pushButton_predictRF.clicked.connect(self.predictResultWithRF)
        self.loadTrainData()
        self.pushButton_Browse.clicked.connect(self.browseFile)
        self.pushButton_EvaluateRF.clicked.connect(self.evaluateRF)
        self.pushButton_AnalysisResult.clicked.connect(self.analysisResult)
        self.pushButton_EvaluateNN.clicked.connect(self.evaluateNN)
        self.setMyStyle()


    def setMyStyle(self):

        self.pushButton_predictRF.setStyleSheet("QPushButton { background-color:#ffdf46;"
                                           "border-width: 2px;border-radius: 10px;padding:15px;min-width:10px;}"
                                           "QPushButton:pressed { background-color: #185189226}"
                                           "QPushButton:hover {background-color: #FDEDEC;}")

        self.pushButton_predictNN.setStyleSheet("QPushButton { background-color:#ffdf46;"
                                                "border-width: 2px;border-radius: 10px;padding:15px;min-width:10px;}"
                                                "QPushButton:pressed { background-color: #185189226}"
                                                "QPushButton:hover {background-color: #FDEDEC;}")

        self.pushButton_EvaluateRF.setStyleSheet("QPushButton { background-color:#ffdf46;"
                                                "border-width: 2px;border-radius: 10px;padding:10px;min-width:10px;}"
                                                "QPushButton:pressed { background-color: #185189226}"
                                                "QPushButton:hover {background-color: #FDEDEC;}")

        self.pushButton_EvaluateNN.setStyleSheet("QPushButton { background-color:#ffdf46;"
                                                "border-width: 2px;border-radius: 10px;padding:10px;min-width:10px;}"
                                                "QPushButton:pressed { background-color: #185189226}"
                                                "QPushButton:hover {background-color: #FDEDEC;}")

        self.pushButton_AnalysisResult.setStyleSheet("QPushButton { background-color:#ffdf46;"
                                                "border-width: 2px;border-radius: 10px;padding:10px;min-width:10px;}"
                                                "QPushButton:pressed { background-color: #185189226}"
                                                "QPushButton:hover {background-color: #FDEDEC;}")


    def testInput(self):
        zeroFun = lambda x:int(0) if(x=="") else int(x)
        race = self.lineEdit_race.text()
        gender = self.comboBox_gender.currentText()
        admission_type = self.comboBox_admissionType.currentText()
        discharge = self.comboBox_discharge.currentText()
        age = self.spinBox_age.value()
        admission_source = self.comboBox_admissionSource.currentText()
        num_Lab = zeroFun(self.lineEdit_numLab.text())
        num_Med = zeroFun(self.lineEdit_numMed.text())
        num_outpatient = zeroFun(self.lineEdit_numOut.text())
        time_in_hospital = zeroFun(self.lineEdit_timeHospital.text())
        num_pro = zeroFun(self.lineEdit_numPro.text())
        num_inpatient = zeroFun(self.lineEdit_numIn.text())
        num_emergency = zeroFun(self.lineEdit_numEmer.text())
        diag1 = self.lineEdit_diag1.text()
        insulin = self.comboBox_insulin.currentText()
        maxGlu = self.comboBox_maxGlu.currentText()
        a1cresult = self.comboBox_a1c.currentText()
        num_diagnose = zeroFun(self.lineEdit_numDiag.text())
        change = self.comboBox_change.currentText()
        diabetes_Med = self.comboBox_diabetesMed.currentText()
        '''print(race, gender, admission_type, discharge, age, admission_source, num_Lab, num_Med, num_outpatient,
              time_in_hospital, num_pro, num_inpatient, num_emergency, diag1, insulin, maxGlu, a1cresult, num_diagnose,
              change, diabetes_Med)'''
        result = self.preprocessingResult(race, gender, admission_type, discharge, age, admission_source, num_Lab,
                                          num_Med, num_outpatient, time_in_hospital, num_pro, num_inpatient,
                                          num_emergency, diag1, insulin, maxGlu, a1cresult, num_diagnose, change,
                                          diabetes_Med)
        print(result)
        return result

    #Preprocessing Before Predict
    def preprocessingResult(self, race, gender, admission_type, discharge, age, admission_source, num_Lab, num_Med,
                            num_outpatient, time_in_hospital, num_Pro, num_inpatient, num_emergency, diag1, insulin,
                            max_glu, a1cresult, num_diagnose, change, diabetesMed):
        # age preprocessing
        if age < 50:
            dfAge = {'[0-50)': [1], '[50-60)': [0], '[60-70)': [0], '[70-80)': [0], '[80-100)': [0]}
        elif age < 60:
            dfAge = {'[0-50)': [0], '[50-60)': [1], '[60-70)': [0], '[70-80)': [0], '[80-100)': [0]}
        elif age < 70:
            dfAge = {'[0-50)': [0], '[50-60)': [0], '[60-70)': [1], '[70-80)': [0], '[80-100)': [0]}
        elif age < 80:
            dfAge = {'[0-50)': [0], '[50-60)': [0], '[60-70)': [0], '[70-80)': [1], '[80-100)': [0]}
        else:
            dfAge = {'[0-50)': [0], '[50-60)': [0], '[60-70)': [0], '[70-80)': [0], '[80-100)': [1]}
        data_age = pd.DataFrame(data=dfAge)

        # gender preprocessing
        if gender == "Female":
            dfGender = {'Female': [1], 'Male': [0]}
        else:
            dfGender = {'Female': [0], 'Male': [1]}
        data_gender = pd.DataFrame(data=dfGender)

        # race preprocessing
        if race == 'AfricanAmerican':
            dfRace = {'AfricanAmerican': [1], 'Asian': [0], 'Caucasian': [0], 'Hispanic': [0], 'Other': [0]}
        elif race == 'Asian':
            dfRace = {'AfricanAmerican': [0], 'Asian': [1], 'Caucasian': [0], 'Hispanic': [0], 'Other': [0]}
        elif race == 'Caucasian':
            dfRace = {'AfricanAmerican': [0], 'Asian': [0], 'Caucasian': [1], 'Hispanic': [0], 'Other': [0]}
        elif race == 'Hispanic':
            dfRace = {'AfricanAmerican': [0], 'Asian': [0], 'Caucasian': [0], 'Hispanic': [1], 'Other': [0]}
        else:
            dfRace = {'AfricanAmerican': [0], 'Asian': [0], 'Caucasian': [0], 'Hispanic': [0], 'Other': [1]}
        data_race = pd.DataFrame(data=dfRace)

        # max_glucose_serum
        if max_glu == '>200':
            dfMaxGlu = {'>200': [1], '>300': [0], 'None': [0], 'Norm': [0]}
        elif max_glu == '>300':
            dfMaxGlu = {'>200': [0], '>300': [1], 'None': [0], 'Norm': [0]}
        elif max_glu == 'None':
            dfMaxGlu = {'>200': [0], '>300': [0], 'None': [1], 'Norm': [0]}
        else:
            dfMaxGlu = {'>200': [0], '>300': [0], 'None': [0], 'Norm': [1]}
        data_max_glu = pd.DataFrame(data=dfMaxGlu)

        # a1cResult
        if a1cresult == '>7':
            dfA1cresult = {'>7': [1], '>8': [0], 'None': [0], 'Norm': [0]}
        elif a1cresult == '>8':
            dfA1cresult = {'>7': [0], '>8': [1], 'None': [0], 'Norm': [0]}
        elif a1cresult == 'None':
            dfA1cresult = {'>7': [0], '>8': [0], 'None': [1], 'Norm': [0]}
        else:
            dfA1cresult = {'>7': [0], '>8': [0], 'None': [0], 'Norm': [1]}
        data_a1cresult = pd.DataFrame(data=dfA1cresult)

        # Insulin
        if insulin == 'Down':
            dfInsulin = {'Down': [1], 'No': [0], 'Steady': [0], 'Up': [0]}
        elif insulin == 'No':
            dfInsulin = {'Down': [0], 'No': [1], 'Steady': [0], 'Up': [0]}
        elif insulin == 'Steady':
            dfInsulin = {'Down': [0], 'No': [0], 'Steady': [1], 'Up': [0]}
        else:
            dfInsulin = {'Down': [0], 'No': [0], 'Steady': [0], 'Up': [1]}
        data_insulin = pd.DataFrame(data=dfInsulin)

        # change
        if change == 'Yes':
            dfChange = {'Ch': [1], 'No': [0]}
        else:
            dfChange = {'Ch': [1], 'No': [0]}
        data_change = pd.DataFrame(data=dfChange)

        # Diabetes_Medication
        if diabetesMed == 'No':
            dfDiabetesMed = {'No': [1], 'Yes': [0]}
        else:
            dfDiabetesMed = {'No': [0], 'Yes': [1]}
        data_diabetesMed = pd.DataFrame(data=dfDiabetesMed)

        # Discharge
        if discharge == 'Discharged to Home':
            dfDischarge = {'Home': [1], 'Other discharge': [0]}
        else:
            dfDischarge = {'Home': [0], 'Other discharge': [1]}
        data_discharge = pd.DataFrame(data=dfDischarge)

        # Admission_Type
        if admission_type == 'Emergency':
            dfAdmissionType = {'Emergency': [1], 'Other type': [0]}
        else:
            dfAdmissionType = {'Emergency': [0], 'Other type': [1]}
        data_admission_type = pd.DataFrame(data=dfAdmissionType)

        # Admission_Source
        if admission_source == 'Emergency':
            dfAdmissionSource = {'Emergency Room': [1], 'Other source': [0], 'Referral': [0]}
        elif admission_source == 'Other Source':
            dfAdmissionSource = {'Emergency Room': [0], 'Other source': [1], 'Referral': [0]}
        else:
            dfAdmissionSource = {'Emergency Room': [0], 'Other source': [0], 'Referral': [1]}
        data_admission_source = pd.DataFrame(data=dfAdmissionSource)

        # Diag1
        diag1 = [1 if val.startswith('250') else 0 for val in [diag1]]
        dfDiag1 = {'diag1': diag1}
        data_diag1 = pd.DataFrame(data=dfDiag1)

        input_data = {}
        input_data = {'time_in_hospital': time_in_hospital, 'num_lab_procedures': num_Lab, 'num_procedures': num_Pro,
                      'num_medications': num_Med,
                      'number_diagnoses': num_diagnose, 'number_inpatient': [np.sqrt(num_inpatient + 0.5)],
                      'number_emergency': [np.sqrt(num_emergency + 0.5)],
                      'number_outpatient': [np.sqrt(num_outpatient + 0.5)]}
        input_data = pd.DataFrame(data=input_data)
        input_data = pd.concat([input_data, data_diag1, data_age, data_race, data_gender, data_max_glu, data_a1cresult,
                                data_insulin, data_change, data_diabetesMed, data_discharge,
                                data_admission_source, data_admission_type], axis=1)
        #input_data.info()
        return input_data


    # Predict with Random Forest Model
    def predictResultWithRF(self):
        try:
            x_test = self.testInput()
            print('This is RF')
            # load the model from disk
            with open("../Model/randomforest_model.sav", 'rb') as file:
                loaded_model = pickle.load(file)

            print('Afrer load model')
            result = loaded_model.predict(x_test)
            print("predicted Result is :", result)
            message = ""
            if (result == 0):
                print(result)
                message = "The patient cannot be readmitted."
                self.callrfdialog(message)
                print(message)
            else:
                print(result)
                message = "The patient can be readmitted"
                self.callrfdialog(message)
                print(message)
        except Exception as e:
            print('RF Error:',str(e))

    #Predict with Neural Network Model
    def predictResultWithNN(self):
        x_test = self.testInput()
        print('This is NN')
        print(type(x_test))
        with open("../Model/neuralnetwork_model.sav", 'rb') as file:
            loaded_model = pickle.load(file)

        print('Afrer load model')
        result = loaded_model.predict(x_test)
        print("predicted Result is :", result)

        message = ""
        if (result == 0):
            message = "The patient cannot be readmitted."
            self.callnndialog(message)
        else:
            message = "The patient can be readmitted"
            self.callnndialog(message)



    def callnndialog(self,message):
        # show dialog box
        from Controller.dialogNN import Ui_Dialog
        Dialog = QtWidgets.QDialog()
        ui = Ui_Dialog()
        ui.setupUi(Dialog, message)
        Dialog.show()
        Dialog.exec_()



    def callrfdialog(self,message):
        # show dialog box
        from Controller.dialogRF import Ui_Dialog
        Dialog = QtWidgets.QDialog()
        ui = Ui_Dialog()
        ui.setupUi(Dialog, message)
        Dialog.show()
        Dialog.exec_()

    #load training data to tableView
    def loadTrainData(self):
        print('loadTrainData')
        df = pd.read_csv('../Dataset/TrainData.csv')
        df.drop(['Unnamed: 0'], axis=1, inplace=True)
        from Model.TableModel import pandasModel
        model = pandasModel(df)
        print('hi')
        self.tableView_train.setModel(model)

    #browseFile for test data and load to tableView
    def browseFile(self):
        filename = QFileDialog.getOpenFileName()
        path = filename[0]
        print(path)
        df = pd.read_csv(path)
        global test_df
        test_df= df
        df.drop(['Unnamed: 0'], axis=1, inplace=True)
        from Model.TableModel import pandasModel
        model = pandasModel(df)
        self.tableView_test.setModel(model)

    def evaluateRF(self):
        try:
            print(test_df)
            #df = self.preprocessingTestData(test_df)

            df = test_df
            #df.drop(['Unnamed: 0'], axis=1, inplace=True)
            # create X (features) and y (response)
            x_test = df.drop(['readmitted'], axis=1)
            y_test = df['readmitted']
            # load the model from disk
            with open("../Model/randomforest_model.sav", 'rb') as file:
                loaded_model = pickle.load(file)
            y_pred = loaded_model.predict(x_test)
            cm = confusion_matrix(y_test, y_pred)
            print("...Random Forest with features...")
            print(cm)
            rf = accuracy_score(y_test, y_pred)
            print(" Accuracy score:{:.2f}".format(rf))
            print(classification_report(y_test, y_pred))
            self.evaluateResultShow(rf,classification_report(y_test, y_pred))
        except Exception as e:
            print("Evaluate RF Error: ",str(e))

    def evaluateNN(self):
        try:
            print(test_df)
            #df = self.preprocessingTestData(test_df)
            df = test_df
            #df.drop(['Unnamed: 0'], axis=1, inplace=True)
            # create X (features) and y (response)
            x_test = df.drop(['readmitted'], axis=1)
            y_test = df['readmitted']
            # load the model from disk
            with open("../Model/neuralnetwork_model.sav", 'rb') as file:
                loaded_model = pickle.load(file)
            y_pred = loaded_model.predict(x_test)
            cm = confusion_matrix(y_test, y_pred)
            print("...Neural Network with features...")
            print(cm)
            nn = accuracy_score(y_test, y_pred)
            print(" Accuracy score:{:.2f}".format(nn))
            print(classification_report(y_test, y_pred))
            self.evaluateResultShow(nn, classification_report(y_test, y_pred))
        except Exception as e:
            print("Evaluate RF Error: ",str(e))

    def analysisResult(self):
        try:
            #df = self.preprocessingTestData(test_df)
            df = test_df
            #df.drop(['Unnamed: 0'], axis=1, inplace=True)

            # create X (features) and y (response)
            x_test = df.drop(['readmitted'], axis=1)
            y_test = df['readmitted']
            # load randomForest model from disk
            with open("../Model/randomforest_model.sav", 'rb') as file:
                rf_model = pickle.load(file)
            y_pred_rf = rf_model.predict(x_test)
            rf = accuracy_score(y_test, y_pred_rf)
            # load NeuralNetwork model from disk
            with open("../Model/neuralnetwork_model.sav", 'rb') as file:
                nn_model = pickle.load(file)
            y_pred_nn = nn_model.predict(x_test)
            nn = accuracy_score(y_test, y_pred_nn)
            self.create_bar(rf, nn)

        except Exception as e:
            print('AnalysisResult Error:',str(e))

    def create_bar(self,rf,nn):
        from Model.CanvasModel import Canvas
        Canvas(self,rf,nn)

    def evaluateResultShow(self,score,classificationResult):
        self.label_accuracy_score.setText(str(score*100))
        self.plainTextEdit_classificationResult.setPlainText(classificationResult)


    '''def preprocessingTestData(self,data):
        # Recategorize 'age' so that the population is more evenly distributed
        data['age'] = pd.Series(['[0-50)' if val in ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)'] else val
                                 for val in data['age']], index=data.index)
        data['age'] = pd.Series(['[80-100)' if val in ['[80-90)', '[90-100)'] else val
                                 for val in data['age']], index=data.index)
        # original 'discharge_disposition_id' contains 28 levels
        # reduce 'discharge_disposition_id' levels into 2 categories
        # discharge_disposition_id = 1 corresponds to 'Discharge Home'
        data['discharge_disposition_id'] = pd.Series(['Home' if val == 1 else 'Other discharge'
                                                      for val in data['discharge_disposition_id']], index=data.index)
        # original 'admission_source_id' contains 25 levels
        # reduce 'admission_source_id' into 3 categories
        data['admission_source_id'] = pd.Series(
            ['Emergency Room' if val == 7 else 'Referral' if val == 1 else 'Other source'
             for val in data['admission_source_id']], index=data.index)
        # original 'admission_type_id' contains 8 levels
        # reduce 'admission_type_id' into 2 categories
        data['admission_type_id'] = pd.Series(['Emergency' if val == 1 else 'Other type'
                                               for val in data['admission_type_id']], index=data.index)
        # denote 'diag_1' as '1' if it relates to diabetes and '0' if it's not
        data['diag_1'] = pd.Series([1 if str(val).startswith('250') else 0 for val in data['diag_1']], index=data.index)
        # one-hot-encoding on categorical features
        # convert nominal values to dummy values
        df_age = pd.get_dummies(data['age'])
        df_race = pd.get_dummies(data['race'])
        df_gender = pd.get_dummies(data['gender'])
        df_max_glu_serum = pd.get_dummies(data['max_glu_serum'])
        df_A1Cresult = pd.get_dummies(data['A1Cresult'])
        df_insulin = pd.get_dummies(data['insulin'])
        df_change = pd.get_dummies(data['change'])
        df_diabetesMed = pd.get_dummies(data['diabetesMed'])
        df_discharge_disposition_id = pd.get_dummies(data['discharge_disposition_id'])
        df_admission_source_id = pd.get_dummies(data['admission_source_id'])
        df_admission_type_id = pd.get_dummies(data['admission_type_id'])

        data = pd.concat([data, df_age, df_race, df_gender, df_max_glu_serum, df_A1Cresult,
                          df_insulin, df_change, df_diabetesMed, df_discharge_disposition_id,
                          df_admission_source_id, df_admission_type_id], axis=1)
        data.drop(['Unnamed: 0','age', 'race', 'gender', 'max_glu_serum', 'A1Cresult', 'insulin', 'change',
                   'diabetesMed', 'discharge_disposition_id', 'admission_source_id',
                   'admission_type_id'], axis=1, inplace=True)
        # apply square root transformation on right skewed count data to reduce the effects of extreme values.
        # here log transformation is not appropriate because the data is Poisson distributed and contains many zero values.
        data['number_outpatient'] = data['number_outpatient'].apply(lambda x: np.sqrt(x + 0.5))
        data['number_emergency'] = data['number_emergency'].apply(lambda x: np.sqrt(x + 0.5))
        data['number_inpatient'] = data['number_inpatient'].apply(lambda x: np.sqrt(x + 0.5))
        return data'''


app=QApplication(sys.argv)
window=mainForm()
window.show()
app.exec_()