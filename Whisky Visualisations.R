#A series of visualisations of data on whiskies using R and ggplot2
#Whisky dataset found at https://datashare.is.ed.ac.uk/handle/10283/2592 

library(ggplot2)
library(ggradar)

#produces a scatter plot with the names of whiskies scalling based on their smoky and body score

plot2 <- ggplot(data = whisky2, aes(Smoky, Body))
plot2 + geom_jitter(aes(colour=factor), show.legend = FALSE
    , width = 0.3, height = 0.3) + geom_text(data = subset(
    whisky2, factor>5) ,aes(label = Distillery, colour=
    factor, size=factor, label.size=0.5), angle=45, nudge_x
     = -0.3, nudge_y = -0.1, check_overlap = TRUE, show.
    legend = FALSE) + scale_radius(range = c(3,6)) +
    scale_colour_gradient(low = "#fff7bc", high = "#d95f0e
    ", guide = FALSE)

#produces a stamen map

mapscot <- ggmap(get_stamenmap(mbox, maptype = "watercolor
    ", zoom = 8), extent = "device")
mapscot + geom_point(aes(x = longs, y = lats, colour = type
    , shape = type, size = cost), data = geos) + geom_text(
    data=subset(geos, type == "Flights"), aes(x = longs, y=
    lats, label=faves, size =65), nudge_y = -0.15) +
    scale_colour_brewer(palette = "Set1") + guides(colour=
    guide_legend("type")) + theme(legend.position="bottom")

#produces a radar plot showing the differences between three whiskies

ggradar(whiskyorder,values.radar = c("0","2","4"), grid.max
     = 4,grid.mid = 2, gridline.mid.colour = "grey", group.point.size = 0)
