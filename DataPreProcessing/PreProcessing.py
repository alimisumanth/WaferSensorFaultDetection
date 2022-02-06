import re
class PreProcessing:
    def __int__(self):
        pass

    def regexMatching(self):
        regex = "['wafer'|'Wafer']+[\_]+(\d{8}\_)+(\d{6})+\.csv"
        return regex

