import csv
import dateutil.parser
import numpy as np
import matplotlib.pyplot
import argparse

import machine_learn

seed = 7 #fix random seed for reproducibility
np.random.seed(seed)

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
    for row in data[1:]:  ## developing with a subset for speed
        # parse the data
        date = dateutil.parser.parse(row[1]+" "+row[2])
        power_kw = row[3]
        power_solar = row[4]
        data_final.append([date, power_kw, power_solar])

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
        hour = data[i][0].hour
        # booleans for hour of the day
        for h in range(24):
            data[i].append(hour == h)
        # booleans for day of week
        wd = data[i][0].weekday()
        for k in range(7):
            data[i].append(wd == k)
        # booleans for day of the month
        md = data[i][0].day
        for j in range(31):
            data[i].append(md == j)
        # booleans for month of the year
        month = data[i][0].month
        for l in range(12):
            data[i].append(month == l)
        data[i].append(data[i][0].date() in holidays)
        # past 24 hours of demand
        d1 = []
        # energy usage for each of the last 96 periods
        # if it is one of the first 96 periods, fill in zeros
        for p1 in range(96):
            d1.append(0)
        for pa in range(96):
            if i > pa:
                d1[pa] += float(data[i -pa-1][1])
        for p2 in d1:
            data[i].append(p2)
        # minimum load of last 12, 24, 48, 96 periods (3,6,12,24 hours)
        for pb in [12, 24, 48, 96]:
            d2 = [621] #620.8 is the maximum value of all usages
            for pb1 in range(pb):
                if i > pb1:
                    d2.append(float(data[i-pb1 - 1][1]))
            data[i].append(min(d2))
        # maximum load of last 12, 24, 48, 96 periods (3,6,12,24 hours)
        for pb in [12, 24, 48, 96]:
            d2 = [0]
            for pb1 in range(pb):
                if i > pb1:
                    d2.append(float(data[i-pb1 - 1][1]))
            data[i].append(max(d2))
        # load of the same hour in all days of the previous week
        pc = []
        for pc1 in range(6):
            pc.append(0)
            if i > 96 * (pc1 + 1):
                pc[pc1] = float(data[i - 96 * (pc1 + 1)][1])
        for pc2 in pc:
            data[i].append(pc2)
        # load of the same hour on the same weekday in previous 4 weeks
        pd = []
        for pd1 in range(4):
            pd.append(0)
            if i > 96 * 7 * (pc1 + 1):
                pd[pd1] = float(data[i - 96 * 7 * (pd1 + 1)][1])
        for pd2 in pd:
            data[i].append(pd2)

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
            new_row = []
            for item in row:
                if item == "True":
                    new_row.append(True)
                elif item =="False":
                    new_row.append(False)
                else:
                    new_row.append(float(item))
            #new_row = (np.array(new_row[1:]), np.array(new_row[0]))
            data.append(new_row)

    return data




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip', "-s", dest='skip', action='store_false', help="use to skip creation of data file")
    parser.add_argument('--custom', "-c", dest='custom', action='store_true', help="use if custom gpu settings apply")
    parser.add_argument('--gpu', "-g", dest='gpu', action='store_true', help="use if machine has gpu")

    args = parser.parse_args()
    if args.skip:
        site1 = parse_csv("site_1.csv")
        t = generate_NN_features(site1, parse_holidays("USBankholidays.txt"))
        write_data(t)
    d = read_data("data.csv")[10100:50100]
    m = len(d)
    n = len(d[0]) - 1
    x = np.zeros((m, n))
    for i in range(m):
        x[i] = d[i][1:]

    model = machine_learn.run_nnet(d, args.gpu, args.custom)
    #calculate predictions
    predictions = model.predict(x)
    # round predictions
    rounded = [round(x[0]) for x in predictions]
    print(rounded)
