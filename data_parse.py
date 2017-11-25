import csv
import datetime
import dateutil.parser
import numpy as np
import matplotlib.pyplot



def parse_csv(file):
    """
    parse data from CSV files into an array
    :param file: name of file to parse
    :return: array of data
    """
    with open(file, newline='\n') as csv_file:
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
    for row in data[1:50]:  ## developing with a subset for speed
        # parse the data
        date = dateutil.parser.parse(row[1]+" "+row[2])
        power_kw = row[3]
        power_solar = row[4]
        data_final.append([date, power_kw, power_solar])

    return data_final

def parse_holidays(file):
    """
    create array of holiday dates as data time objects from USBankHolidays.txt
    :param data:
    :return:
    """
    with open(file, newline='\n') as csv_file:
        reader = csv.reader(csv_file)
        holidays = []
        for row in reader:
            #holidays.append(dateutil.parser.parse(row[1]).date())
            holidays.append(str(row[1]))
        return holidays


def generate_NN_features(data, holidays): # based off features used in Gajowniczek paper
    """
    generate features for the dataset
    :param data:
    :param holidays:
    :return:
    """

    for i in range(len(data)):
        hour = data[i][0].hour
        for h in range(24):
            data[i].append(hour == h)
        wd = data[i][0].weekday()
        for k in range(7):
            data[i].append(wd == k)
        md = data[i][0].day
        for j in range(31):
            data[i].append(md == j)
        month = data[i][0].month
        for l in range(12):
            data[i].append(month == l)
        print(holidays[20])
        print(data[i][0].date())
        data.append(data[i][0].date().isoformat() in holidays)  #I DO NOT KNOW WHY THIS IS NOT WORKING. THEY ARE BOTH STRINGS. ALSO TRIED DATETIME OBJECTS
    return data


if __name__ == '__main__':
    site1 = parse_csv("site_1.csv")
    print(site1[:10])
    print("t:")
    t = generate_NN_features(site1[:10], parse_holidays("USBankholidays.txt"))
    print(t)