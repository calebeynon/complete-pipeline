#!/usr/bin/env Rscript
# ==============================================================================
# Step 2: Create Firm-Specific Variables
#
# This script creates company-specific features from various data sources:
# 1. Financial ratings
# 2. Asset composition
# 3. Line of business proportions
# 4. Geographic distribution
# 5. Economic indicators
#
# Author: Caleb
# Date: 2025
# ==============================================================================

library(data.table)
library(dplyr)

# Helper function to read CSV as data.table
dtr = function(x) as.data.table(read.csv(x))


# ==============================================================================
# LOAD PREPROCESSED DATA
# ==============================================================================

print("Loading preprocessed insurance company data...")

# AMB ratings - Financial strength ratings from A.M. Best
ratings = readRDS('Rprocessed AMB')

# Company assets - Balance sheet information
assets = readRDS('Rprocessed_assets')

# BEA data - Bureau of Economic Analysis regional economic data
BEA = readRDS('Rprocessed_BEA')

# Company demographics - Type, structure, etc.
demo = readRDS('Rprocessed_demo')

# Page 14 data - Losses by line of business and state
pg14 = readRDS('Rprocessed_Page14')

# Underwriting income and expenses
uie = readRDS('Rprocessed_uie_pw')

# Filter to companies with cocode >= 10000
# (Excludes small or special purpose entities)
ratings <- ratings[cocode>=10000]
demo <- demo[cocode>=10000]
assets <- assets[cocode>=10000]
uie <- uie[cocode>=10000]
pg14 <- pg14[cocode>=10000]

print(paste("Filtered to", length(unique(demo$cocode)), "companies"))

# ==============================================================================
# CREATE BASE DATASET
# ==============================================================================

print("Creating base dataset with all company-year combinations...")

# Collect all unique company-year pairs from all data sources
dts <- list(ratings, assets, demo, pg14, uie)
final <- data.table()

for(dt in dts){
  # Union all unique company-year combinations
  final <- unique(rbind(final, unique(dt[,.(cocode, year)])))
}

print(paste("Found", nrow(final), "company-year combinations"))

# ==============================================================================
# ADD COMPANY CHARACTERISTICS
# ==============================================================================

print("Adding company characteristics...")

# Add insurer type (Stock, Mutual, Reciprocal, etc.)
final <- merge(final, unique(demo[,.(cocode, cotype_desc, year)]), 
               by=c('cocode', 'year'))

# Add financial strength ratings
final <- merge(final, unique(ratings[,.(cocode, rating, year)]), 
               by=c('cocode', 'year'))

# Add asset information and calculate financial leverage
final <- merge(final, unique(assets), by=c('cocode', 'year'))
final[, debttoequity := bonds/stocks]  # Financial leverage metric

# ==============================================================================
# ADD UNDERWRITING METRICS
# ==============================================================================

print("Adding underwriting metrics...")

# Aggregate losses and premiums across all lines and states
# dpw = Direct Premiums Written
# dpe = Direct Premiums Earned
# loss = Incurred losses
agglosses <- pg14[,.(dpw = sum(dpw), dpe = sum(dpe), loss = sum(loss)), 
                  by = .(cocode, year)]
final <- merge(final, agglosses, by = c('cocode', 'year'))

# ==============================================================================
# CALCULATE LINE OF BUSINESS PROPORTIONS
# ==============================================================================

print("Calculating line of business proportions...")

# Calculate proportion of business in each line
# This shows company specialization/diversification
dpwlstate <- pg14[,.(dpw = sum(dpw)), by = .(year, cocode, lob)]  # By line
dpwllob <- pg14[,.(dpw = sum(dpw)), by = .(year, cocode)]         # Total
props <- merge(dpwlstate, dpwllob, by = c('cocode', 'year'))
props[, proportions := dpw.x/dpw.y]  # Calculate proportion

# Add each line of business proportion as a separate column
for(lb in unique(props[,lob])){
  temp <- props[lob==lb]
  temp[,lob := NULL]
  temp[,dpw.x := NULL]
  temp[,dpw.y := NULL]
  setnames(temp, 'proportions', paste0('prop_', lb))
  final <- merge(final, temp, by = c('cocode', 'year'), all.x=T)
}

# ==============================================================================
# CALCULATE GEOGRAPHIC DISTRIBUTION
# ==============================================================================

print("Calculating geographic distribution...")

# Calculate proportion of business in each state
# This shows geographic concentration/diversification
dpwllob <- pg14[,.(dpw=sum(dpw)), by = .(year, cocode, state)]  # By state
dpwlstate <- pg14[,.(dpw=sum(dpw)), by = .(year, cocode)]       # Total
props <- merge(dpwllob, dpwlstate, by= c('cocode', 'year'))
props[, proportions := dpw.x/dpw.y]  # Calculate proportion

# Add each state proportion as a separate column
for(st in unique(props[,state])){
  temp <- props[state==st]
  temp[,state := NULL]
  temp[,dpw.x := NULL]
  temp[,dpw.y := NULL]
  setnames(temp, 'proportions', paste0('prop_', st))
  final <- merge(final, temp, by = c('cocode', 'year'), all.x=T)
}

# ==============================================================================
# SAVE INTERMEDIATE DATA (BEFORE ECONOMIC VARIABLES)
# ==============================================================================

print("Saving intermediate data (data1023.csv)...")
fwrite(final, 'data1023.csv')

# ==============================================================================
# ADD WEIGHTED ECONOMIC VARIABLES
# ==============================================================================

print("Adding weighted economic variables...")

# Merge state proportions with BEA economic data
dpwwstatedemos <- merge(props, BEA, by=c('year', 'state'))

# Calculate weighted averages based on geographic distribution
# These represent the economic conditions where the company operates
demosprops <- dpwwstatedemos[,.
  (wPOP = sum(POP*proportions),        # Weighted population
   wEMP = sum(EMP_total*proportions),  # Weighted employment
   wINC = sum(state_incomeT*proportions)), # Weighted income
  by = .(year, cocode)]

# Merge economic variables with company data
data <- merge(final, demosprops, by = c('year', 'cocode'))

# Replace NA values with 0
data[is.na(data)] <- 0

# ==============================================================================
# SAVE INTERMEDIATE DATA (WITH ECONOMIC VARIABLES)
# ==============================================================================

print("Saving intermediate data (data0302.csv)...")
fwrite(data, 'data0302.csv')

# ==============================================================================
# ADD LOSS DEVELOPMENT FACTORS
# ==============================================================================

print("Adding loss development factors from Step 1...")

# Load loss development factors from previous step
ldfdt <- dtr('developed_and_ldfs_3-6.csv')

# Remove specific LDF columns that will be kept in a different format
# These are age-to-age factors (e.g., 12-24 months, 24-36 months, etc.)
ldfdt[,c('X12.24','X24.36','X36.48','X48.60','X60.72',
         'X72.84','X84.96','X96.108','X108.120'):= NULL]

# Merge loss development data with company variables
data_merged <- merge(data, ldfdt, by = c('year', 'cocode'))

# Remove any index column if present
if('X' %in% names(data_merged)) data_merged[,X:=NULL]

# ==============================================================================
# SAVE FINAL OUTPUT
# ==============================================================================

print("Saving final dataset (data0306.csv)...")
fwrite(data_merged, 'data0306.csv')

print(paste("Complete! Created dataset with", nrow(data_merged), "records and", 
            ncol(data_merged), "features"))

# Script complete - data ready for next step (adding chain ladder estimates)















