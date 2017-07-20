#!/usr/bin/env Rscript
library(readr)
require(MESS)
require(Bolstad2)
# con <- file("energyPlot.txt")
# sink(con, append=TRUE)
# sink(con, append=TRUE, type="message")

# source("integrateEnergy.r", echo=TRUE, max.deparse.length=10000)


# sink(type="message")

# cat(readLines("energyPlot.txt"), sep="\n")
# testRstudioAUC <- read_delim('1.txt', " ", escape_double = FALSE, col_names = FALSE, trim_ws = TRUE)
# x <- (testRstudioAUC$X1)
# y <- (testRstudioAUC$X2)
# data <- sintegral(x,y)$int
# data <- auc(x, y, type='spline')
# write.table(data,quote=FALSE,append=TRUE,row.names=FALSE,col.names=FALSE,sep=" ", "energyPlot.txt")
for(i in seq(from=2, to=16, by=2)){
    infile <- paste(i,".txt",sep="")
    # outfile <- paste(i,"-edit.txt",sep="")
    testRstudioAUC <- read_delim(infile, " ", escape_double = FALSE, col_names = FALSE, trim_ws = TRUE)
    x <- (testRstudioAUC$X1)
    y <- (testRstudioAUC$X2)
    data <- paste(" ", sintegral(x,y)$int, sep="")
    # data <- auc(x, y, type='spline')
    # data <- read.table(infile,header=TRUE,sep=",",row.names=NULL)
    # colnames(data)[1] = "time"
    write.table(data,quote=FALSE,append=TRUE,row.names=FALSE,col.names=FALSE,sep=" ", "energyPlot.txt")
}
# sink()
