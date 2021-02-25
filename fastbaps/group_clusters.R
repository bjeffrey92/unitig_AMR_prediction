library(ggtree)
library(phytools)
library(ggplot2)
library(dplyr)
library(memoise)
library(cachem)
library(parallel)
library(foreach)
library(doParallel)

num.cores <- detectCores() - 1
registerDoParallel(num.cores) 


load.inputs <- function(newick.file, fastbaps.clusters.file) {
    tree <- phytools::read.newick(newick.file)

    df <- read.csv(fastbaps.clusters.file, stringsAsFactors = FALSE)
    names(df) <- c('id', 'fastbaps') 

    return(list(tree, df))
}


#wrapper for the memoised function to ensure nodes are submitted in the same order
cache.dir <<- tempdir()
cache.size <<- 2048 * 1024^2
m.fast.dist <- memoise::memoise(phytools::fastDist, 
                                cache = cachem::cache_disk(
                                                dir = cache.dir,
                                                max_size = cache.size))
node.distance <- function(nodes, tree) {
    nodes <- sort(nodes)
    return(m.fast.dist(tree, nodes[[1]], nodes[[2]]))
}


cache.percent.full <- function() {
    used.mem <- file.size(dir(cache.dir, full.names = TRUE)) %>% 
                    sum() 
    return(used.mem/cache.size * 100)
}


#loops over nodes in two clusters and returns all distances between each
all.node.distances <- function(tree, nodes.grid) {
    foreach (i = 1:nrow(nodes.grid), .combine = c) %dopar% {
        node.distance(as.character(nodes.grid[i,]), tree)
    }
}


get.neighbouring.cluster <- function(tree, cluster, df) {
    cluster.nodes <- subset(df, fastbaps == cluster)$id
    other.clusters <- subset(df, fastbaps != cluster)$fastbaps %>%
                        unique()
    
    #get the min distance between two nodes in object cluster and every other cluster
    distances <- vector('list', length(other.clusters))
    names(distances) <- other.clusters
    for (other.cluster in other.clusters) {
        target.nodes <- subset(df, fastbaps == other.cluster)$id
        nodes.grid <- expand.grid(cluster.nodes, target.nodes, 
                                stringsAsFactors = FALSE)
        min.dist <- all.node.distances(tree, nodes.grid) %>% min()
        distances[[other.cluster]] <- min.dist
    }

    closest.cluster <- distances[order(unlist(distances))] %>% 
                        head(1) %>% 
                        names()
    
    return(closest.cluster)
}


compact.clusters <- function(tree, df, min.clusters = 10, 
                            save_intermediates = TRUE) {
    n.clusters <- length(unique(df$fastbaps))
    cluster.sizes <- table(df$fastbaps) %>% 
                            as.data.frame()

    while (n.clusters > min.clusters) {
        print(sprintf('Smallest Cluster Size = %s, %s clusters in total', 
                    min(cluster.sizes$Freq),
                    n.clusters))
        
        smallest.cluster <- subset(cluster.sizes, 
                                Freq == min(cluster.sizes$Freq))$Var1

        #finds cluster most likely to be the neighbour of smallest cluster and adds smallest cluster nodes to this    
        neighbouring.cluster <- get.neighbouring.cluster(tree, 
                                                        smallest.cluster, df)
        df[df$fastbaps == smallest.cluster,]$fastbaps <- neighbouring.cluster

        print(sprintf('Cache %s%% full', cache.percent.full()))
        n.clusters <- length(unique(df$fastbaps))
        cluster.sizes <- table(df$fastbaps) %>% 
                            as.data.frame()
        
        df$fastbaps <- as.numeric(as.factor(df$fastbaps))
        if (save_intermediates){
            write.csv(df, sprintf('compacted_clusters_%s.csv', n.clusters), 
                    quote = FALSE, row.names = FALSE)
        }
    }
    return(df)
}


plot.clusters <- function(tree, plot.df) {
    gg <- ggtree(tree)
    f2 <- facet_plot(gg, panel = "fastbaps", data = plot.df, geom = geom_tile, 
                    aes(x = fastbaps), color = "blue")
    return(f2)
}


main <- function() {
    newick.file <- 'fasttree.final_tree.tre'
    fastbaps.clusters.file <- 'phylogeny_constrained_fastbaps_clusters.csv' #output of fastbaps with phylogeny command line arg

    inputs <- load.inputs(newick.file, fastbaps.clusters.file)
    tree <- inputs[[1]]
    df <- inputs[[2]]
    
    compacted.cluster.df <- compact.clusters(tree, df)
    plot.clusters(tree, plot.df)
}