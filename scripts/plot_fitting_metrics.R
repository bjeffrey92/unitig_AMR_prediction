#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly=TRUE)
file <- args[1]

df <- read.csv(file, sep = '\t')

training_data_loss <- df$training_data_loss
training_data_accuracy <- df$training_data_acc
testing_data_loss <- df$testing_data_loss
testing_data_accuracy <- df$testing_data_acc

png('progress_plots.png')
par(mfrow=c(2,2))
plot(training_data_loss)
plot(training_data_accuracy)
# plot(testing_data_loss)
# plot(testing_data_accuracy)
dev.off()
