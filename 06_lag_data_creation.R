#!/usr/bin/env Rscript
# ==============================================================================
# Step 6: Create Peer Averages by Company Type
#
# This script creates peer benchmarking features by calculating average
# triangle values for companies of the same type. This allows the neural
# network to learn from industry patterns and identify when a company
# deviates from its peer group.
#
# The key insight: Companies of similar types (Stock, Mutual, etc.) tend
# to have similar loss development patterns. Deviations may indicate
# company-specific issues or opportunities.
#
# Author: Caleb
# Date: 2025
# ==============================================================================

library(data.table)

# Helper function to read CSV as data.table
dtr = function(x) as.data.table(read.csv(x))

# ==============================================================================
# LOAD DATA
# ==============================================================================

print("Loading data with full triangle features...")
dt <- dtr('lag_data_scaled_0407.csv')

# Remove any index column if present
if('X' %in% names(dt)) dt[,X:= NULL]

# ==============================================================================
# RESHAPE DATA FOR PEER CALCULATION
# ==============================================================================

print("Reshaping data to calculate peer averages...")

# Define company type columns (one-hot encoded)
melters <- c('cotype_desc_OTHER',
             'cotype_desc_SYNDICATE',
             'cotype_desc_MUTUAL',
             'cotype_desc_RECIPROCAL',
             'cotype_desc_RISK.RETENTION.GROUP',
             'cotype_desc_STOCK',
             'cotype_desc_US.BRANCH.OF.ALIEN.INSURER')

# Melt the data to have one row per company-type combination
# This converts from wide format (one column per type) to long format
dt <- melt(dt,
           id.vars = setdiff(names(dt), melters),  # Keep all non-type columns
           measure.vars = melters,                  # Melt the type columns
           variable.name = 'type')                  # Name for the type variable

# Keep only rows where the company is of that type (value==1)
dt <- dt[value==1]

print(paste("Processing", nrow(dt), "company-type-year combinations"))

# ==============================================================================
# CALCULATE PEER AVERAGES
# ==============================================================================

print("Calculating peer averages for triangle features...")

# Calculate mean of triangle features (ft_0 to ft_54) by year and type
# Using year-1 to create lagged peer averages (avoiding look-ahead bias)
means <- dt[,lapply(.SD, mean, na.rm=T),
            by=.(year-1, type),
            .SDcols=patterns("^ft_")]  # Select only ft_ columns

# Rename columns to indicate these are peer averages
setnames(means, c('year', 'type', paste0('avg_ft_', 0:54)))

print("Peer averages calculated for each company type and year")

# ==============================================================================
# MERGE PEER AVERAGES WITH ORIGINAL DATA
# ==============================================================================

print("Merging peer averages with company data...")

# Merge the peer averages back to the main dataset
dtm <- merge(dt, means, all.x=T, by=c('year', 'type'))

# Remove any unnamed columns
if('Unnamed..0' %in% names(dtm)) dtm[,Unnamed..0:=NULL]

# ==============================================================================
# RESHAPE BACK TO WIDE FORMAT
# ==============================================================================

print("Reshaping back to wide format...")

# Cast back to wide format with company types as columns
dt_final <- dcast(dtm, ...~type, value.var = 'value')

# Fill NAs with 0 (companies are not of that type)
dt_final[is.na(dt_final)] <- 0

# ==============================================================================
# SAVE OUTPUT
# ==============================================================================

print("Saving final dataset with peer averages...")
fwrite(dt_final, file = 'lag_data_0416.csv')

print(paste("Complete! Saved", nrow(dt_final), "records with peer averages"))
print("Added 55 peer average features (avg_ft_0 to avg_ft_54)")
print("These features enable the model to:")
print("  - Benchmark each company against its peer group")
print("  - Identify unusual development patterns")
print("  - Learn from industry-wide trends")




# ==============================================================================
# ARCHIVED CODE (Alternative peer calculations)
# ==============================================================================
# The code below shows an alternative approach using sums of chain ladder
# and development values. Kept for reference but not used in current pipeline.
#
# dt[,cl_sum := cl_dev_9+cl_dev_8+cl_dev_7+cl_dev_6+cl_dev_5+cl_dev_4+cl_dev_3+cl_dev_2+cl_dev_1+cl_dev_0]
# dt[,lag_cl_avg := mean(cl_sum),by=year-1]
# dt[,lag_cl_type := mean(cl_sum),by = .(type,year-1)]
# dt[,lag_cl_rating := mean(cl_sum), by = .(rating,year-1)]
# 
# dt[,sum_dev := dev_0+dev_1+dev_2+dev_3+dev_4+dev_5+dev_6+dev_7+dev_8]
# dt[,lag_dev_avg := mean(sum_dev),by=year-1]
# dt[,lag_dev_type := mean(sum_dev),by = .(type,year-1)]
# dt[,lag_dev_rating := mean(sum_dev), by = .(rating,year-1)]














