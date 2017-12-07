library(tidyverse)
site_1 <- read_csv("~/PycharmProjects/ML_final_project/site_1.csv")

site_1$date = lubridate::mdy_hm(site_1$local_date)

jan14 <- filter(site_1,month(site_1$date) == 1 & year(site_1$date) == 2014)

jan14$wd <- lubridate::wday(b$d)
# jan14$m = jan14$wd == 1
# jan14$t = jan14$wd == 2
# jan14$w = jan14$wd == 3
# jan14$h = jan14$wd == 4
# jan14$f = jan14$wd == 5
# jan14$s = jan14$wd == 6
# jan14$u = jan14$wd == 7

cbPalette <- c("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

# DayOfWeek = factor(jan14$wd)
p1<- ggplot(data=jan14,aes(x=jan14$date,y=jan14$`customer_load_no_pv (kW)`))#, colour = DayOfWeek))

p1 +geom_point() + scale_colour_manual(values=cbPalette) + labs(x = "date", y = "kW", title = "January 2014 Power Usage", caption = "Sunday = 1")

for (i in c(1:12)){
  m<- filter(site_1,month(site_1$date) == i & year(site_1$date) == 2014)
  p<- ggplot(data=m,aes(x=m$date,y=m$`customer_load_no_pv (kW)`)) + geom_point() + labs(x = "date", y = "kW", title = paste("Power Usage 2014-",i))
  print(p)
}

