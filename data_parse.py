import csv
import datetime
import dateutil.parser
import numpy as np
import matplotlib.pyplot
import argparse

import nnet


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
    for row in data[1:1000]:  ## developing with a subset for speed
        # parse the data
        date = dateutil.parser.parse(row[1]+" "+row[2])
        power_kw = row[3]
        power_solar = row[4]
        data_final.append([[date, power_solar], power_kw])
        # make it a tuple to make it suitable for nnet application
    return data_final

def parse_holidays(file):
    """
    create array of holiday dates as data time objects from USBankHolidays.txt
    :param fule:
    :return:array of holiday dates
    """
    with open(file, newline='\n') as csv_file:
        reader = csv.reader(csv_file)
        holidays = []
        for row in reader:
            holidays.append(dateutil.parser.parse(row[1]).date())
        return holidays


def generate_NN_features(data, holidays): # based off features used in Gajowniczek paper
    """
    generate features for the dataset
    :param data:
    :param holidays:
    :return:
    """

    for i in range(len(data)):
        hour = data[i][0][0].hour
        # booleans for hour of the day
        for h in range(24):
            data[i][0].append(hour == h)
        # booleans for day of week
        wd = data[i][0][0].weekday()
        for k in range(7):
            data[i][0].append(wd == k)
        # booleans for day of the month
        md = data[i][0][0].day
        for j in range(31):
            data[i][0].append(md == j)
        # booleans for month of the year
        month = data[i][0][0].month
        for l in range(12):
            data[i][0].append(month == l)
        data[i][0].append(data[i][0][0].date() in holidays)
        # past 24 hours of demand
        d1 = []
        # energy usage for each of the last 96 periods
        # if it is one of the first 96 periods, fill in zeros
        for p1 in range(96):
            d1.append(0)
        for pa in range(96):
            if i > pa:
                d1[pa] += float(data[i -pa-1][0][1])
        for p2 in d1:
            data[i][0].append(p2)
        # minimum load of last 12, 24, 48, 96 periods (3,6,12,24 hours)
        for pb in [12, 24, 48, 96]:
            d2 = [621] #620.8 is the maximum value of all usages
            for pb1 in range(pb):
                if i > pb1:
                    d2.append(float(data[i-pb1 - 1][0][1]))
            data[i][0].append(min(d2))
        # maximum load of last 12, 24, 48, 96 periods (3,6,12,24 hours)
        for pb in [12, 24, 48, 96]:
            d2 = [0]
            for pb1 in range(pb):
                if i > pb1:
                    d2.append(float(data[i-pb1 - 1][0][1]))
            data[i][0].append(max(d2))
        # load of the same hour in all days of the previous week
        pc = []
        for pc1 in range(6):
            pc.append(0)
            if i > 96 * (pc1 + 1):
                pc[pc1] = float(data[i - 96 * (pc1 + 1)][0][1])
        for pc2 in pc:
            data[i][0].append(pc2)
        # load of the same hour on the same weekday in previous 4 weeks
        pd = []
        for pd1 in range(4):
            pd.append(0)
            if i > 96 * 7 * (pc1 + 1):
                pd[pd1] = float(data[i - 96 * 7 * (pd1 + 1)][0][1])
        for pd2 in pd:
            data[i][0].append(pd2)

    return data
def write_data(data):
    """
    wites the unlabeled data to a csv file
    :param data: data to write
    :return:
    """

    with open("data.csv", "w+") as data_file:
        writer = csv.writer(data_file)
        for row in data:
            writer.writerow(row[1:])


def read_data(file):
    """
    reads the parsed data back as a list
    :param file: file to read the data
    :return:
    """
    with open(file) as data_file:
        reader = csv.reader(data_file)
        data = []
        for row in reader:
            data.append(row)
    return data




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip', "-s", dest='skip', action='store_false', help="use to skip creation of data file")
    args = parser.parse_args()
    if args.skip:
        site1 = parse_csv("site_1.csv")
        #print(site1[:10])
        print("t:")
        t = generate_NN_features(site1[:1000], parse_holidays("USBankholidays.txt"))
        #print(t)
        print("100th example:")
        print(t[100])
        print(len(t[100]))
        write_data(t)
    d = read_data("data.csv")
    print(d[100])

    # split into training and validation:
    train_data = d[:500]
    valid_data = d[500:]

    net = nnet.Network([191, 20, 1])

    print("training")
    net.train(train_data, valid_data, epochs=10, mini_batch_size=10, alpha=0.0)


    ncorrect = net.evaluate(valid_data)
    print("Validation accuracy: %.3f%%" % (100 * ncorrect / len(valid_data)))