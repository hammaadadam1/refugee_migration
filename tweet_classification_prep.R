setwd("~/Desktop/Columbia/Courses/PML/Final Project/Data Collection/Data_for_Analysis/")
library(stm)
library(tm)
library(stringr)
library(dplyr)

full_data <- readRDS("tweets_for_modeling.rds")

# Dumb stuff to ass the additional tweet. Don't run
additional_neg <- readRDS("../Data_for_Analysis/Archive/additional_neg_tweets.rds")
additional_neg$X <- c(1:nrow(additional_neg))
additional_neg_labelled <- read.csv("../Data/additional_negative.csv", stringsAsFactors = FALSE)
additional_neg_labelled$status_id <- as.character(additional_neg_labelled$status_id)
add_neg <- additional_neg %>% inner_join(additional_neg_labelled, c("X"))
add_neg <- add_neg %>% select(screen_name.x, status_id.x, text.x, Label) %>%
            rename(screen_name = screen_name.x,
                   status_id = status_id.x,
                   text = text.x, 
                   label = Label)
add_neg$location <- NA
add_neg$is_retweet <- NA
add_neg$retweet_location <- NA
add_neg$city <- NA

add_neg_labels <- add_neg %>% select(screen_name, status_id, text, label)
add_neg_text   <- add_neg %>% select(-label)
full_data_text <- rbind(full_data, add_neg_text)
#saveRDS(full_data_text, "full_data_text.rds")

# Read in training labels
training_labels <- read.csv("training_w_id_labelled_v2.csv", stringsAsFactors = FALSE, na.strings = c("NA", ""))
training_set_tweets <- readRDS("../Data_for_Analysis/Archive/training_set_text.rds")
training_set_tweets$label <- training_labels$label
training_set_tweets <- rbind(training_set_tweets, add_neg_labels)

# Begin actual work

data <- readRDS("../Data_for_Analysis/Archive/full_data_text.rds")
tweets <- data %>% select(status_id, text) %>% 
            left_join(training_set_tweets %>% select(status_id, label), c("status_id"="status_id"))
tweets <- tweets %>% distinct(status_id, .keep_all = TRUE)
saveRDS(tweets, "tweets_for_classification.rds")

View(tweets %>% filter(!(is.na(label))))

tweets %>% group_by(label) %>% summarise(n=n())
