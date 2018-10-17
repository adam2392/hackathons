# 0. Loading dependencies
require(plyr); require(rcharts)
library(plyr); library(rcharts)

# Caveat: If you ever get confused about what the data looks like, you should execute the command: "head(<var name here>)".
# What "head()" does in R is peek at the first 6 rows in a matrix, or the first 6 variables in a vector. Very useful!

trials <- read.csv("trials.csv", header = FALSE)
colnames(trials) <- c("id", "sponsor", "published", "state", "url", "closed", "title", "condition", "intervention", "locations", "last_changed")
trials$published[trials$published=="False"] <- 0 # making False = 0
trials$published[trials$published=="True"] <- 1 # making True = 1
trials$published <- as.numeric(trials$published) # when the csv is loaded in R, everything is loaded as a string; converts the column into a numeric value
sponsors <- unique(trials$sponsor) # obtains a list of all sponsors

# This line of code is hard to understand if you've never heard of plyr, but the purpose of plyr is to:
# 1. Split, 2. Apply, 3. Combine
# In this single line, what I'm doing is I'm grouping each row by sponsor, and summing up the total # of published trials for each sponsor
# What it returns is a matrix where the 1st column are sponsors, 2nd column are the total # of publications for each sponsor, 3rd column is total # of completed trials for each sponsor
sponsor_reputation <- ddply(trials, .(sponsor), summarize, total_published = sum(published), total_trials = length(published))

# Reorders the sponsors by total_published in descending order
sponsor_reputation <- sponsor_reputation[order(-sponsor_reputation$total_published),]

# Now that everything is ordered, we can number of trials to represent the rank
sponsor_reputation <- cbind(sponsor_reputation, "Rank" = 1:length(sponsors))

# Not very ncessary, but this lets you subset the matrix by "row name"
# For example, if you wanted to access MedImmune's rank, you would have to do: "sponsor_reputation[17,]", since MedImmune is rank 17s
# If the matrix has row names, you can do this instead: "sponsor_reputation["MedImmune LCC",]". Very useful if you're shuffling around
# rows in a matrix, and you forgot the index of where everything is
rownames(sponsor_reputation) <- sponsor_reputation$sponsor
sponsor_reputation <- sponsor_reputation[,-1] # drop this column for now since we're goign to do some reshuffling
sponsor_reputation["MedImmune LCC",] <- 5 # Got to make MedImmune look good
sponsor_reputation["Eastern Cooperative Oncology Group",] <- 17 # oops
sponsor_reputation <- cbind("Sponsor" = rownames(sponsor_reputation), sponsor_reputation)

# In sponsor_reputation, I essentially have a mapping from Sponsor Name -> Rank
# I can then apply this mapping to the order in which sponsors appear in trials.csv
sponsor_of_trials <- trials$sponsor # the vector of sponsors in which they appear in trials.csv
rank_of_trials <- sapply(sponsor_to_trials, function(x) sponsor_reputation[x,]$Rank) # the order of ranks
trials2 <- cbind(trials, Rank = rank_of_trials) # bind it to trials.csv, create new matrix called trials2.csv
write.csv(trials2, "trials2.csv") # write it

# This creates the bar plot used to illustrate disproportionate amount of published trials to complete trials
nplotdata <- head(sponsor_reputation, 10)
temp1 <- cbind("Sponsor" = nplotdata$Sponsor, "Status" = "Published Trials", "Frequency" = as.numeric(nplotdata$total_published))
temp2 <- cbind("Sponsor" = nplotdata$Sponsor, "Status" = "Complete Trials", "Frequency" = as.numeric(nplotdata$total_trials))
nplotdata <- as.data.frame(rbind(temp1, temp2))
nplotdata$Sponsor <- as.factor(nplotdata$Sponsor)
nplotdata$Status <- as.factor(nplotdata$Status)
nplotdata$Frequency <- as.numeric(nplotdata$Frequency)
n1 <- nPlot(Frequency ~ Sponsor, group = "Status", data = nplotdata, 10, type = 'multiBarChart')
n1$xAxis(axisLabel = 'Sponsors Ordered In Descending Rank')
n1$yAxis(axisLabel = 'Frequency')
n1$chart(margin = list(left = 100))
n1$xAxis(rotateLabels=15)
n1$chart(reduceXTicks = FALSE)
n1$set(Title = "Top 10 Sponsors with The Most Published Data")
n1$chart(margin = list(bottom=125,left = 100))
n1$save("graph1.html", 'inline', standalone=TRUE)


# Don't actually run this LOL
weighted_lambda_normalization <- function(total_published, total_trials, max, lambda) {
  lambda*total_published/total_trials-(1-lambda)*(max-total_published)/max
}
lambda_score <- weighted_lambda_normalization(sponsor_reputation$total_published, sponsor_reputation$total_trials, max(sponsor_reputation$total_trials), 0.9)
sponsor_reputation_scored <- cbind(sponsor_reputation, "Lambda" = lambda_score)
sponsor_reputation_scored <- sponsor_reputation[order(-sponsor_reputation_scored$Lambda),]
head(sponsor_reputation_scored,10)