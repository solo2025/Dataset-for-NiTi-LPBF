## calculate pearson correlation coefficient matrix and plot heatmap

# set tl.font as Times new roman and size 12
par(family = "Times New Roman", cex = 1.0)

# Load libraries
install.packages("corrplot")
install.packages("showtext")
library(corrplot)
library(showtext)

font_add("times","times.ttf")
showtext_auto()

# Load data
data <- read.csv("D:/Projects/muti_obj/model_features/data/feature.csv", header = TRUE, sep = ",")

# Calculate pearson correlation coefficient matrix
correlation_matrix <- cor(data, method = "pearson")

# Plot heatmap
# set the outline width of the pie chart
# set the colorbar font size
# set the colorbar font color


corrplot(corr = cor(correlation_matrix),
        method = "pie",
        type = "upper",
        order = "hclust",
        addrect = 2,
        rect.col = "#ff0000c0",
        tl.cex = 1,
        family = "times",
        tl.col = "black",
        outline = "black", 
        tl.pos = "td",
        tl.srt = 45)

