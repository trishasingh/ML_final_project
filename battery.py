import csv
import datetime
import dateutil.parser
import numpy as np

def parse_csv(file):
    """
    parse data from CSV files into an array
    :param file: name of file to parse
    :return: array of data
    """
    with open(file, newline='\n') as csv_file:
    #with open(file, newline='\n') as csv_file:
        reader = csv.reader(csv_file)
        data = []
        i = 0
        # Get the data from the files, ignore the first row
        for row in reader:
            try:
                data.append(row)
                i += 1
            except ValueError:
                pass
    data_final = []
    # Skip the label row.
    for row in data[1:]:
        # Parse the data.
        date = dateutil.parser.parse(row[1]+" "+row[2])
        power_kw = float(row[3])
        #power_solar = float(row[4])
        data_final.append([date, power_kw ])#-power_solar])

    return data_final

def monthSeparate(data):
    """

    :param data:
    :return:
    """
    data_final = []
    for yr in [2014,2015]:
        for mon in range(1,13,1):
            data_final.append([])
            ind = (yr - 2014)*12 + mon - 1
            for i in range(len(data)):
                if data[i][0].year == yr and data[i][0].month == mon:
                    data_final[ind].append(data[i])
    return data_final

def checkThreshold(data,threshold):
    """

    :param data:
    :param threshold:
    :return:
    """



if __name__ == '__main__':

    site1 = parse_csv("site_1.csv")
    ms = monthSeparate(site1)
