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

    :param data: parsed data
    :return: array of arrays separated by month
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

def testDeltas(month, y):
    """
    :param month: one month's energy usage data from monthSeparate
    :param y: kwH of battery
    :return: nothing for now
    """
    for i in range(0, 105, 5):
        m = peakShaved(month, y, i)
        print("Shaved by: " + str(i) + " kw")
        print(str(m[0]) + " in period " + str(m[1]))

def testDeltaMonth(month, y):
    """

    :param month: one month's energy usage data from monthSeparate
    :param y: kwH of battery
    :return: max amount able to peak shave for that month
    """
    maxShaved = 0
    for i in range(0,205,5):
        m = peakShaved(month,y,i)
        if m[0]:
            maxShaved = i
    return maxShaved

def testSize(data,y):
    """
    :param data: array of arrays from monthSeparate
    :param y: battery size in kwh
    :return: array of max amount shaved per month
    """
    shavedByMonth = []
    for i in range(len(data)):
        x = testDeltaMonth(data[i],y)
        shavedByMonth.append(x)
    return shavedByMonth

def peakShaved(data, y, delta):
    """
    :param data: one month's energy usage data from monthSeparate
    :param y: battery size in kWh
    :param delta: difference between month's peak and the threshold in kw
    :return:
    """
    worked = True
    period = "all"
    z = y/2  # kw peak, this is an assumption that that is a constant ratio
    a = y
    eU = []  # an array that just has the energy usage numbers, instead of an array of arrays
    for i in range(len(data)):
        eU.append(data[i][1])
    thresh = max(eU) - delta  # set the threshold for the month
    for p in range(len(eU)):
        b = z
        x = eU[p]
        if x > thresh:  # if need to drain
            b -= x - thresh  # kw above thresh
            a -= (x - thresh)/4  # drained kWh from battery
        if thresh - x > 20:  # assuming that our prediction will be this good and we are safe to charge
            da = thresh - x - 20
            if da > b:
                da = b
            a += da/4
            if a > y:
                a = y
        if a < 0:
            worked = False
            period = p
            break
        if b < 0:
            worked = False
            period = p
            break
    return worked, period



if __name__ == '__main__':

    site1 = parse_csv("site_1.csv")
    ms = monthSeparate(site1)
    y = 200
    a = testSize(ms, y)
    print(a[:12])
    print(a[12:])
    print()
    z = 400
    b = testSize(ms, z)
    print(b[:12])
    print(b[12:])
