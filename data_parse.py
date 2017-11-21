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
    for row in data[1:]:
        # parse the data
        date = dateutil.parser.parse(row[1]+" "+row[2])
        power_kw = row[3]
        power_solar = row[4]
        data_final.append([date, power_kw, power_solar])

    return data_final


if __name__ == '__main__':
    site1 = parse_csv("site_1.csv")
    print(site1[:10])