# Group Members:
# Graham, Trisha, Jonah
import csv
import datetime
import dateutil.parser
import numpy as np
import argparse
import machine_learn

seed = 7 #fix random seed for reproducibility
np.random.seed(seed)

MAX_LOAD = 621

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


def parse_holidays(file):
    """
    Create array of holiday dates as data time objects from USBankHolidays.txt.
    :param file: Holiday file
    :return: array of holiday dates
    """
    with open(file, newline='\n') as csv_file:
        reader = csv.reader(csv_file)
        holidays = []
        for row in reader:
            holidays.append(dateutil.parser.parse(row[1]).date())
        return holidays


def generate_NN_features(data, holidays): # based off features used in Gajowniczek paper
    """
    Generate features for the data-set.
    :param data: parsed raw data
    :param holidays: parsed holiday info
    :return: features
    """
    for i in range(len(data)):
        minute = data[i][0].minute
        # Booleans for minute of the hour, only have data for 0, 15, 30, 45 minute markers
        for m in [0,15,30,45]:
            data[i].append(minute == m)
        hour = data[i][0].hour
        # Booleans for hour of the day.
        for h in range(24):
            data[i].append(hour == h)
        # Booleans for day of week.
        wd = data[i][0].weekday()
        for k in range(7):
            data[i].append(wd == k)
        # Booleans for day of the month.
        md = data[i][0].day
        for j in range(31):
            data[i].append(md == j)
        # Booleans for month of the year.
        month = data[i][0].month
        for l in range(12):
            data[i].append(month == l)
        data[i].append(data[i][0].date() in holidays)
        # Past 24 hours of demand.
        d1 = []
        # Energy usage for each of the last 96 periods.
        # If it is one of the first 96 periods, fill in zeros.
        for p1 in range(96):
            d1.append(0)
        for pa in range(96):
            if i > pa:
                d1[pa] += float(data[i -pa-1][1])
        for p2 in d1:
            data[i].append(p2)
        # Minimum load of last 12, 24, 48, 96 periods (3,6,12,24 hours).
        for pb in [12, 24, 48, 96]:
            d2 = [MAX_LOAD]
            for pb1 in range(pb):
                if i > pb1:
                    d2.append(float(data[i-pb1 - 1][1]))
            data[i].append(min(d2))
        # Maximum load of last 12, 24, 48, 96 periods (3,6,12,24 hours).
        for pb in [12, 24, 48, 96]:
            d2 = [0]
            for pb1 in range(pb):
                if i > pb1:
                    d2.append(float(data[i-pb1 - 1][1]))
            data[i].append(max(d2))
        # Load of the same hour in all days of the previous week.
        pc = []
        for pc1 in range(6):
            pc.append(0)
            if i > 96 * (pc1 + 1):
                pc[pc1] = float(data[i - 96 * (pc1 + 1)][1])
        for pc2 in pc:
            data[i].append(pc2)
        # Load of the same hour on the same weekday in previous 4 weeks.
        pd = []
        for pd1 in range(6):
            pd.append(0)
            if i > 96 * 7 * (pd1 + 1):
                pd[pd1] = float(data[i - 96 * 7 * (pd1 + 1)][1])
        for pd2 in pd:
            data[i].append(pd2)
        # add max, min, avg for that day of the week in the 4 previous weeks
        for wk in range(4):
            prevwkd = []
            if i > 96*8*(wk+1):
                pOfDay = ((data[i][0].minute/15)+1)*(data[i][0].hour + 1)
                for pd3 in range(96*6*(wk+1)+pOfDay, 96*7*(wk+1)+pOfDay, 1):
                    prevwkd.append(data[i-pd3][1])
                    pwkdMax = max(prevwkd)
                    pwkdMin = min(prevwkd)
                    pwkdAvg = sum(prevwkd)/len(prevwkd)
                    data[i].append(pwkdMax)
                    data[i].append(pwkdMin)
                    data[i].append(pwkdAvg)
    return data


def write_data(data):
    """
    Writes the unlabeled data to a csv file.
    :param data: data to write
    """
    with open("data.csv", "w+", newline='') as data_file:
        writer = csv.writer(data_file)
        for row in data:
            writer.writerow(row)


def read_data(file):
    """
    Reads the parsed data back as a list.
    :param file: file to read the data
    :return: features
    """
    with open(file) as data_file:
        reader = csv.reader(data_file)
        data = []
        for row in reader:
            new_row = []
            for item in range(len(row)):
                if item == 0:
                    new_row.append(dateutil.parser.parse(row[item]))
                # Convert boolean strings to booleans.
                elif row[item] == "True":
                    new_row.append(True)
                elif row[item] == "False":
                    new_row.append(False)
                else:
                    new_row.append(float(row[item]))
            data.append(new_row)

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip', "-s", dest='skip', action='store_true', help="use to skip creation of data file")
    parser.add_argument('--gpu', "-g", dest='gpu', action='store_true', help="use gpu optimization")
    parser.add_argument('--model', "-m", dest='model', action='store', default='', help="path to model being retrained")
    parser.add_argument('--no_train', "-n", dest='no', action='store_true', help="use to skip training")
    args = parser.parse_args()
    # Do we want to skip?
    if not args.skip:
        site1 = parse_csv("site_1.csv")
        t = generate_NN_features(site1, set(parse_holidays("USBankholidays.txt")))
        write_data(t)
    # Do we train?
    if not args.no:
        # Read in data.
        d = read_data("data.csv")[5000:] #edit dataset size here
        x, y = machine_learn.format_data(d)
        model = machine_learn.run_nnet(x, y, args.gpu, args.model)
        # Save the model.
        model.save("models/model_"+datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S") +".h5")