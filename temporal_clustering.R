# Perform temporal clustering on multivariate time series data for behavioural anlysis
library(reticulate) # loading pickles
library(dtwclust)

loadData <- function(pickleReaderPath, filePath){
  use_python("/home/timothysit/anaconda3/bin/python", required = TRUE)
  py_config()
  source_python(pickleReaderPath)
  pickle_data <- read_pickle_file(filePath)
 return(pickle_data) 
}

testDTW <- function(){
  # tests if the dwtclust library works 
  data("uciCT")
  # perform partitional clustering
  pc <- tsclust(CharTraj, type = "partitional", k = 20L, 
                distance = "dtw_basic", centroid = "pam", 
                seed = 3247L, trace = TRUE,
                args = tsclust_args(dist = list(window.size = 20L)))
  plot(pc)
}

multivarDTW <- function(multivarTimeSeries){
  # performs multivariate dynamic time warping (DTW) clustering
  mvc <- tsclust(multivarTimeSeries, k = 4L, distance = "gak", preproc = zscore, seed = 390L)
}


main <- function(){
  testDTW()
  distanceDf <- loadData(pickleReaderPath = "/media/timothysit/180C-2DDD/swc/swc_behavioural_analysis/pickle_reader.py",
                         filePath = "/media/timothysit/180C-2DDD/swc/swc_behavioural_analysis/data/body_part_loc_uninterpolated_df.pkl")
  }


if (!interactive()) {
  main()
}