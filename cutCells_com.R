library(terra)
setwd("/home/dingwenn/BGB/")
#??? user parameters ------------------------------------------------------------

parent_folder <- "/home/dingwenn/BGB/SDM/"  # contains ???current???, ???landuse???, ??? 

output_root   <- "Extracted_Squares"

win_size      <- 128              # cells per side
sample_counts <- list("25m"=2, "100m"=6, "250m"=2)
agg_factors   <- list("25m"=1, "100m"=4, "250m"=10)
maxtries      <- 10000
set.seed(42)

#??? discover the ???current??? folder & its first .tif ----------------------------

cur_folder <- file.path(parent_folder, "current")
tifs_curr  <- list.files(cur_folder, "\\.tif$", full.names=TRUE)
if(length(tifs_curr)<1) stop("No .tif in current/")

#??? derive the three lat-bands in the 25 m grid -------------------------------

ref25   <- rast(tifs_curr[1])
xmin_o  <- xmin(ref25); xmax_o <- xmax(ref25)
ymin_o  <- ymin(ref25); ymax_o <- ymax(ref25)
y_third <- (ymax_o - ymin_o) / 3
bands   <- list(
  "25m"  = c(ymin_o,           ymin_o + y_third),
  "100m" = c(ymin_o + y_third, ymin_o + 2*y_third),
  "250m" = c(ymin_o + 2*y_third, ymax_o)
)

#??? STEP 1: sample extents only once from the first ???current??? raster ---------

ref_r    <- rast(tifs_curr[1])
extents  <- list()

for(res in names(agg_factors)) {
  fac    <- agg_factors[[res]]
  r      <- if(fac>1) aggregate(ref_r, fact=fac, fun=mean) else ref_r
  
  rx     <- res(r)[1]; ry <- res(r)[2]
  yrng   <- bands[[res]]
  xmin_r <- xmin(r); xmax_r <- xmax(r)
  ymin_r <- yrng[1]; ymax_r <- yrng[2]
  
  n      <- sample_counts[[res]]
  chosen <- vector("list", n)
  done   <- 0L; tries <- 0L
  
  while(done < n && tries < maxtries) {
    tries <- tries + 1L
    x0    <- runif(1, xmin_r, xmax_r - win_size*rx)
    y0    <- runif(1, ymin_r, ymax_r - win_size*ry)
    e     <- ext(x0, x0 + win_size*rx, y0, y0 + win_size*ry)
    sq    <- crop(r, e)
    if(!anyNA(values(sq))) {
      done <- done + 1L
      chosen[[done]] <- e
    }
  }
  if(done < n) {
    warning(sprintf(
      "[reference|%s] wanted %d but got %d extents after %d tries",
      res, n, done, tries
    ))
  }
  extents[[res]] <- chosen
}

#??? STEP 2: apply these 10 extents to every .tif in every scenario ------------

# ensure output root
dir.create(output_root, showWarnings=FALSE)

scenarios <- list.dirs(parent_folder, full.names=TRUE, recursive=FALSE)
for(scen in scenarios) {
  scen_nm <- basename(scen)
  message("??? Scenario: ", scen_nm)
  scen_out <- file.path(output_root, scen_nm)
  dir.create(scen_out, recursive=TRUE, showWarnings=FALSE)
  
  tifs <- list.files(scen, "\\.tif$", full.names=TRUE)
  if(length(tifs)==0) next
  
  for(f in tifs) {
    bnm <- tools::file_path_sans_ext(basename(f))
    r0  <- rast(f)
    
    for(res in names(agg_factors)) {
      fac   <- agg_factors[[res]]
      r     <- if(fac>1) aggregate(r0, fact=fac, fun=mean) else r0
      
      out_sub <- file.path(scen_out, res, bnm)
      dir.create(out_sub, recursive=TRUE, showWarnings=FALSE)
      
      for(i in seq_along(extents[[res]])) {
        e  <- extents[[res]][[i]]
        sq <- crop(r, e)
        outp <- file.path(out_sub,
                          sprintf("%s_%s_square_%02d.tif",
                                  bnm, res, i))
        writeRaster(sq, outp, overwrite=TRUE)
      }
    }
    message("  ??? done: ", bnm)
  }
}

message("??? All scenarios processed with identical 10-window positions!") 

