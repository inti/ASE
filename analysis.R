library(VGAM)
library(plyr)
library(ggplot2)

info = read.delim("SraRunTable.txt")

data = read.delim("test_all_10K",header=T)
data$x = as.integer(data$alpha_post)
data$n = as.integer(data$beta_post)

head(data,3)


s = subset(data, name =="LOC100650155")
s2 = merge(s[,c("bam","pASE","x","n")],info[,c("Run","treatment","colony_of_origin")],by.x="bam",by.y="Run")

r = lmer(pASE ~ 0 + (1|colony_of_origin) + (1|treatment),data=s2)
