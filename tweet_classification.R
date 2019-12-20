setwd("~/Desktop/Columbia/Courses/PML/Final Project/Data Collection/Data_for_Analysis/")
library(stm)
library(tm)
library(stringr)
library(tidytext)
library(ranger)
library(Matrix)
library(glmnet)
library(caret)
library(DMwR)
library(naivebayes)

data <- readRDS("tweets_for_classification.rds")

# Text processing: convert to lowercase, remove punctuation, stopwords, infrequently appearing words, and HTML

data_processed = textProcessor(data$text, metadata = data %>% dplyr::select(-text), lowercase = TRUE,
                                  removestopwords = TRUE, removenumbers = TRUE,
                                  removepunctuation = TRUE, stem = TRUE,
                                  wordLengths = c(3, Inf), sparselevel = 1, language = "en",
                                  verbose = TRUE, onlycharacter = TRUE, striphtml = TRUE)

plotRemoved(data_processed$documents, lower.thresh = seq(1, 1000, by = 10))

data_processed <- prepDocuments(data_processed$documents, data_processed$vocab,  data_processed$meta, 
                     lower.thresh = 200)

doc_term <- convertCorpus(data_processed$documents, data_processed$vocab, 
                          type = "Matrix")

labelled <- doc_term[!is.na(data_processed$meta$label),]
labels <- data_processed$meta$label[!is.na(data_processed$meta$label)]
labels <- as.numeric(labels)

# Model Training and Prediction

# Train and test split

train_split <- 0.85

train_idx <- sample(length(labels), train_split*length(labels))
X_train <- labelled[train_idx, ]
X_test <- labelled[-train_idx, ]
y_train <- labels[train_idx]
y_test <- labels[-train_idx]

# Upsample to balance classes

upSampled = upSample(as.matrix(X_train), factor(y_train), list=TRUE)
y_train_up = upSampled$y
X_train_up = upSampled$x
X_train_up = Matrix(X_train_up, sparse=TRUE)

# Bayesian multinomial logistic regression (no upsampling)

cvfit = cv.glmnet(X_train, y_train, family = "multinomial", alpha=0.5, type.measure = "class")
pred <- predict(cvfit, X_test, "lambda.min", type="class")
pred_prob <- predict(cvfit, X_test, "lambda.min")
pred_prob <- exp(pred_prob) / rowSums(exp(pred_prob))
confusionMatrix(factor(pred), factor(y_test))

plot_probs <- cbind(pred[,1], y_test, pred_prob[,,1])
plot_probs <- apply(plot_probs,2,as.numeric)
plot_probs <- data.frame(plot_probs)
names(plot_probs) <- c("pred", "y", "neg", "neu", "pos")
plot_probs$score <- plot_probs$pos + 0.5*plot_probs$neu
ggplot(plot_probs, aes(x=factor(y), y=score)) + geom_boxplot()

# Bayesian multinomial logistic regression (w/ upsampling)

cvfit = cv.glmnet(X_train_up, y_train_up, family = "multinomial", alpha=0.5, type.measure = "class")
plot(cvfit, label = TRUE)
cvfit$lambda.min

pred <- predict(cvfit, X_test, "lambda.min", type="class")
pred_prob <- predict(cvfit, X_test, "lambda.min")
pred_prob <- exp(pred_prob) / rowSums(exp(pred_prob))
pred_prob <- pred_prob[,,1]
confusionMatrix(factor(pred), factor(y_test))

plot_probs <- cbind(pred, y_test, pred_prob)
plot_probs <- apply(plot_probs,2,as.numeric)
plot_probs <- data.frame(plot_probs)
names(plot_probs) <- c("pred", "y", "neg", "neu", "pos")
plot_probs$score <- plot_probs$pos + 0.5*plot_probs$neu
ggplot(plot_probs, aes(x=factor(y), y=score)) + geom_boxplot()

# Random forest

X_y_train_up <- cbind(as.numeric(y_train_up), X_train_up)
dimnames(X_y_train_up)[[2]][1] <- "y"

r = ranger(data = X_y_train_up, dependent.variable.name = "y", 
            classification = TRUE, num.trees = 1000, oob.error = TRUE, 
              probability = TRUE)  
pred_ranger_probs <- predict(r, X_test)$predictions
plot_probs_ranger <- cbind(y_test, pred_ranger_probs)
plot_probs_ranger <- data.frame(plot_probs_ranger)
names(plot_probs_ranger) <- c("y", "neg", "neu", "pos")
plot_probs_ranger$score <- plot_probs_ranger$pos + 0.5*plot_probs_ranger$neu
ggplot(plot_probs_ranger, aes(x=factor(y), y=score)) + geom_boxplot()
pred_ranger <- apply(pred_ranger_probs,1, which.max)-2
confusionMatrix(factor(pred_ranger), factor(y_test))$table

# Poisson Naive Bayes

train.control <- trainControl(method = "cv", number = 10, classProbs = TRUE, 
                              summaryFunction=mnLogLoss)
pnbGrid <- expand.grid(laplace = seq(1, 1, by=1),
                       usekernel = FALSE,
                       adjust=0)

naive_bayes <- train(apply(as.matrix(X_train_up),2, as.integer), 
                     plyr::revalue(y_train_up, c("-1"="negative", "0"="neutral", "1"="positive")), 
                     method = 'naive_bayes', trControl = train.control, usepoisson=TRUE,
                     metric='logLoss', tuneGrid=pnbGrid)

sum(predict(naive_bayes, apply(as.matrix(X_train_up),2, as.integer), type="raw") == plyr::revalue(y_train_up, c("-1"="negative", "0"="neutral", "1"="positive"))) / length(y_train_up)

pred_pnb_probs <- predict(naive_bayes, apply(as.matrix(X_test),2, as.integer), type="prob")
pred_pnb <- predict(naive_bayes, apply(as.matrix(X_test),2, as.integer), type="raw")

plot_probs_pnb <- cbind(y_test, pred_pnb_probs)
plot_probs_pnb <- data.frame(plot_probs_pnb)
names(plot_probs_pnb) <- c("y", "neg", "neu", "pos")
plot_probs_pnb$score <- plot_probs_pnb$pos + 0.5*plot_probs_pnb$neu
ggplot(plot_probs_pnb, aes(x=plyr::revalue(factor(y), c("-1"="Hostile", "0"="Neutral", "1"="Sympathetic")), 
                           y=score)) + 
  geom_boxplot() + 
    theme_classic() + xlab("Actual Label") + 
      ylab("Predicted Sentiment Score") + 
        theme(text= element_text(size=14), 
              axis.text.x = element_text(size=12), 
              axis.text.y = element_text(size=10)) 


confusionMatrix(pred_pnb, plyr::revalue(factor(y_test), c("-1"="negative", "0"="neutral", "1"="positive")))$table
nb_coef <- tables(naive_bayes$finalModel)

nb_coef_clean <- data.frame(matrix(NA, nrow=length(nb_coef), ncol=4))
names(nb_coef_clean) <- c("word", "negative", "neutral", "positive")
nb_coef_clean$word <- names(nb_coef)

for(i in 1:length(nb_coef)){
  nb_coef_clean[i,2:4] <- c(nb_coef[i][[1]])
}

head(cbind(nb_coef_clean %>% dplyr::arrange(-negative) %>% dplyr::select(word),
      nb_coef_clean %>% dplyr::arrange(-neutral) %>% dplyr::select(word),
      nb_coef_clean %>% dplyr::arrange(-positive) %>% dplyr::select(word)), 12)
      
# Decide to use Naive Bayes. Upsamples and train over full dataset

X <- labelled
y <- labels
upSampled = upSample(as.matrix(X), factor(y), list=TRUE)
y_up = upSampled$y
X_up = upSampled$x
X_up = Matrix(X_up, sparse=TRUE)

train.control <- trainControl(method = "cv", number = 10, classProbs = TRUE, 
                              summaryFunction=mnLogLoss)

pnbGrid <- expand.grid(laplace = seq(1, 5, by=1),
                       usekernel = FALSE,
                       adjust=0)

naive_bayes <- train(apply(as.matrix(X_up),2, as.integer), 
                     plyr::revalue(y_up, c("-1"="negative", "0"="neutral", "1"="positive")), 
                     method = 'naive_bayes', trControl = train.control, usepoisson=TRUE,
                     metric='logLoss', tuneGrid=pnbGrid)

pred_pnb_probs <- predict(naive_bayes, apply(as.matrix(X_up),2, as.integer), type="prob")
pred_pnb <- predict(naive_bayes, apply(as.matrix(X_up),2, as.integer), type="raw")

plot_probs_pnb <- cbind(y_up, pred_pnb_probs)
plot_probs_pnb <- data.frame(plot_probs_pnb)
names(plot_probs_pnb) <- c("y", "neg", "neu", "pos")
plot_probs_pnb$score <- plot_probs_pnb$pos + 0.5*plot_probs_pnb$neu
ggplot(plot_probs_pnb, aes(x=factor(y), y=score)) + geom_boxplot()
confusionMatrix(pred_pnb, plyr::revalue(factor(y_up), c("-1"="negative", "0"="neutral", "1"="positive")))

nb_coef <- tables(naive_bayes$finalModel)

nb_coef_clean <- data.frame(matrix(NA, nrow=length(nb_coef), ncol=4))
names(nb_coef_clean) <- c("word", "negative", "neutral", "positive")
nb_coef_clean$word <- names(nb_coef)

for(i in 1:length(nb_coef)){
  nb_coef_clean[i,2:4] <- c(nb_coef[i][[1]])
}

head(cbind(nb_coef_clean %>% dplyr::arrange(-negative) %>% dplyr::select(word),
           nb_coef_clean %>% dplyr::arrange(-neutral) %>% dplyr::select(word),
           nb_coef_clean %>% dplyr::arrange(-positive) %>% dplyr::select(word)), 20)

# Full dataset prediction

final_pred_pnb <- predict(naive_bayes, apply(as.matrix(doc_term),2, as.integer), type="prob")
dim(final_pred_pnb)
sanity_check_plot <- final_pred_pnb[!is.na(data_processed$meta$label),]
sanity_check_plot <- cbind(y, sanity_check_plot)
sanity_check_plot <- data.frame(sanity_check_plot)
names(sanity_check_plot) <- c("y", "neg", "neu", "pos")
sanity_check_plot$score <- sanity_check_plot$pos + 0.5*sanity_check_plot$neu
ggplot(sanity_check_plot, aes(x=factor(y), y=score)) + geom_boxplot()

final_pred_pnb_df <- data.frame(final_pred_pnb)
final_pred_pnb_df$score <- final_pred_pnb_df$pos + 0.5*final_pred_pnb_df$neu
final_pred_pnb_df$status_id <- data_processed$meta$status_id

ggplot(final_pred_pnb_df, aes(score)) + geom_histogram() + theme_classic()
saveRDS(final_pred_pnb_df, "tweet_sentiments.rds")

data$score <- NA
data$score[-data_processed$docs.removed] <- final_pred_pnb_df$score
hist(data$score)

print((data[is.na(data_processed$meta$label[-data_processed$docs.removed]),] %>% dplyr::arrange(score) %>% 
         dplyr::select(text) %>% unique() %>% head(20))$text)
