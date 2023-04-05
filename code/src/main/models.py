class Medication:
    def __init__(self, patientID, date, medicine):
        self.patientID = patientID
        self.date = date
        self.medicine = medicine

class LabResult:
    def __init__(self, patientID, date, resultName, value):
        self.patientID = patientID
        self.date = date
        self.resultName = resultName
        self.value = value

class Diagnostic:
    def __init__(self, patientID, code, date):
        self.patientID = patientID
        self.date = date
        self.code = code