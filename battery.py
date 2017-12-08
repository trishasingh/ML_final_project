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


def costsSaved(shaved, year, duration, peakRate = 18, discount = .05, rateInflation = .02):
    """
    Assumption: the operator of the battery has a really good prediction and can perfectly set peak threshold
    :param shaved: data for amount possible to shave per month for one building for two years
    :param year: 2014 or 2015, need to choose one year's data to replicate
    :param duration: length of time in years that will project cost savings for
    :param peakRate: rate per peak kW in dollars
    :param discount: annual discount rate for the building owner to invest in battery
    :param rateInflation: expected peak rate increases annually
    :return: present value dollar amount for costs saved from peak shaving
    """
    if year == 2014:
        shaved = shaved[:12]
    if year == 2015:
        shaved = shaved[12:]
    mlyDiscount = (1 + discount) ** (1/12)
    mlyInflation = (1 + rateInflation) ** (1/12)
    for i in range(duration-1):
        for i2 in range(12):
            shaved.append(shaved[i2])
    save = []
    for s in range(len(shaved)):
        ir = (peakRate*(mlyInflation ** s))
        dr = mlyDiscount ** s
        s2 = shaved[s]
        d1 = ir * s2 / dr
        save.append(d1)
    pv = sum(save)
    return pv


def netGain(saved, size, price = 500):
    """
    Assumption : peak kW of battery = 1/2 kWh
    :param saved: costs saved from peak shaving
    :param size: size of battery in kWh
    :param price: price per kWh of battery
    :return: net profit in present value
    """
    batteryCost = size * price
    net = saved - batteryCost
    return net


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


def testDeltaMonth(month, y):
    """

    :param month: one month's energy usage data from monthSeparate
    :param y: kwH of battery
    :return: max amount able to peak shave for that month
    """
    maxShaved = 0
    for i in range(0, y+1, 1):
        m = peakShaved(month, y, i)
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
        x = testDeltaMonth(data[i], y)
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


def loopSizes(data, largest, year, duration, smallest = 200, increment = 50, price = 500, peakRate = 18, discount = .05, rateInflation = .02):
    """
    :param data: monthlySeparated data for site
    :param largest: largest battery size feasible
    :param smallest: smallest battery size available, standard is 200 kWh for tesla powerpack
    :param increment: size intervals that you can buy a pack in, let's say 50 is standard
    :param price: price of battery per kWh
    :param peakRate:
    :param discount:
    :param rateInflation:
    :return: array of net profits, array of costs saved
    """
    saved = []
    nets = []
    for size in range(smallest, largest + increment, increment):
        a = testSize(data, size)
        c = costsSaved(a, year, duration, peakRate, discount, rateInflation)
        n = netGain(c,size,price)
        saved.append(c)
        nets.append(n)
    return nets, saved


if __name__ == '__main__':

    site1 = parse_csv("site_1.csv")
    ms1 = monthSeparate(site1)
    inc = 25
    small = 25
    dur = 10
    a = loopSizes(ms1, 500, 2014, dur, smallest=small, increment=inc)
    print("site 1 using 2014 data:")
    for i1 in range(len(a[0])):
        print("size = " + str(small + i1*inc))
        print("net profit = $" + str(a[0][i1]))
    b = loopSizes(ms1, 500, 2015, dur, smallest=small, increment=inc)
    print()
    print()
    print("site 1 using 2015 data:")
    for i2 in range(len(b[0])):
        print("size = " + str(small + i2*inc))
        print("net profit = $" + str(b[0][i2]))
    print()
    print()
    site2 = parse_csv("site_2.csv")
    ms2 = monthSeparate(site2)
    e = loopSizes(ms2, 500, 2014, dur, smallest=small, increment=inc)
    print("site 2 using 2014 data:")
    for i3 in range(len(e[0])):
        print("size = " + str(small + i3 * inc))
        print("net profit = $" + str(e[0][i3]))
    f = loopSizes(ms2, 500, 2015, dur, smallest=small, increment=inc)
    print()
    print()
    print("site 2 using 2015 data:")
    for i4 in range(len(f[0])):
        print("size = " + str(small + i4 * inc))
        print("net profit = $" + str(f[0][i4]))
    # a = testSize(ms, y)
    # print(a[:12])
    # print(a[12:])
    # print()
    # z = 400
    # b = testSize(ms, z)
    # print(b[:12])
    # print(b[12:])
    # c = costsSaved(a, 2014, 1)
    # n = netGain(c, y)
    # print(n)
    # print()
    #
    # d = costsSaved(a, 2014, 10)
    # n1 = netGain(d, y)
    # print(n1)
    # print(c)
