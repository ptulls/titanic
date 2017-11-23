# titanic

#Libraries
require(ggplot2)
require(dplyr)
require(rpart)
install.packages("mice")
require(mice)
require(e1071)
require(randomForest)


#Get data
getwd()
setwd("C:/Users/LONPT12/Desktop/Training/Data Science Bootcamp 2/Kaggle/Titanic")
train_set <- read.csv("train.csv")
test_set <- read.csv("test.csv")

str(test_set)
str(train_set)

head(test_set)
head(train_set)

#Combine data
test_set$Survived <- NA
all_data <- rbind(train_set,test_set)
summary(all_data)

#explore and transform
#Pclass
all_data$Pclass <- as.factor(all_data$Pclass)
str(all_data)
#Title
head(all_data$Name,20)

all_data$Title <- gsub("(.*, )|(\\..*)","",all_data$Name)
head(all_data$Title)
table(all_data$Sex,all_data$Title)
rare_titles <- c('Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')
all_data$Title[all_data$Title == "Mlle"] <- "Miss"
all_data$Title[all_data$Title == "Mme"] <- "Mrs"
all_data$Title[all_data$Title == "Ms"] <- "Miss"
all_data$Title[all_data$Title %in% rare_titles] <- "Rare Title"

summary(all_data$Title)
all_data$Title <- as.factor(all_data$Title)

#Age
hist(all_data$Age)
all_data$Age_old <- all_data$Age
head(all_data[,c("Age_old","Age")])

age_guess <- rpart(Age~Pclass+Title+Sex+SibSp+Parch+Fare+Embarked+Fare,
      data = all_data[!is.na(all_data$Age),],
      method = "anova")
all_data$Age[is.na(all_data$Age)] <- predict(age_guess,all_data[is.na(all_data$Age),])

hist(all_data$Age_old, col=rgb(1,0,0,0.5))
hist(all_data$Age, col=rgb(0,0,1,0.5), add=T)
box()

old_age <- data.frame(all_data$Age_old,)
new_age <- data.frame(all_data$Age)
old_age$type <- "old"
new_age$type <- "new"
colnames(old_age) <- c("age", "type")
colnames(new_age) <- c("age", "type")
age_data <- rbind(old_age,new_age)
ggplot(age_data, aes(age, fill = type)) + 
  geom_histogram(alpha = 0.5, position = 'identity')

all_data <- all_data[,-ncol(all_data)]

#Family
head(all_data)
all_data$Family <- all_data$SibSp + all_data$Parch + 1
head(all_data[,c("Family","SibSp","Parch")])
hist(all_data$Family)

ggplot(all_data[!is.na(all_data$Survived),],aes(x = Family,fill=factor(Survived))) +
  geom_bar(position = "dodge")+
  xlab("Family size") +
  ylab("Number of records") +
  ggtitle("Family survival chance")

all_data$FamilyD[all_data$Family == 1] <- "Single"
all_data$FamilyD[all_data$Family > 1 & all_data$Family < 5] <- "Small"
all_data$FamilyD[all_data$Family > 4] <- "Large"
all_data$FamilyD <- as.factor(all_data$FamilyD)

#Embarked
table(all_data$Embarked)
all_data[all_data$Embarked=="",]

ggplot(all_data[all_data$Embarked %in% c("C","Q","S"),],aes(x = Embarked,y = Fare, fill = Pclass)) +
  geom_boxplot() +
  ylim(0,300) +
  geom_hline(yintercept = 80)

all_data$Embarked[all_data$Embarked==""] <- "C"
table(all_data$Embarked)

#Fare
summary(all_data)
all_data[is.na(all_data$Fare),]
all_data$Fare[1044] <- median(all_data$Fare[all_data$Embarked=="S" & all_data$Pclass == "3"],na.rm = TRUE)

#test for missing values
md.pattern(all_data)
head(all_data)
all_data <- all_data[,-ncol(all_data)]
all_data$Age <- round(all_data$Age,0)

#model building
train_set <- all_data[!is.na(all_data$Survived),]
test_set <- all_data[is.na(all_data$Survived),]
head(test_set)

set.seed(123)

train_set_1 <- train_set[1:floor(3*nrow(train_set)/4),]
train_set_2 <- train_set[nrow(train_set_1):nrow(train_set),]

#feature scaling


#logistic regression
log_model <- glm(factor(Survived)~.,family = binomial,data = train_set_1)
log_pred <- predict(log_model,type = "response", newdata = train_set_2[,-1])
plot(log_model)
#not good because data not independant or linear

#SVM kernel
svm_k_model <- svm(factor(Survived)~.,
                   data = train_set_1,
                   type = "C-classification",
                   kernel = "radial")

svm_k_pred <- predict(svm_k_model,newdata = train_set_2[,-1])
svm_k_cm <- table(train_set_2[,1],svm_k_pred)
svm_k_cm
summary(svm_k_model)

#Random Forest
rf_model <- randomForest(x = train_set_1[,-1],
                         y=factor(train_set_1$Survived),
                         ntree = 1000)
rf_pred <- predict(rf_model,newdata = train_set_2[,-1])
rf_cm <- table(train_set_2[,1],rf_pred)
rf_cm
plot(rf_model)

importance <- importance(rf_model)
varImportance <- data.frame(Variables = row.names(importance),
                            Importance = round(importance[ ,'MeanDecreaseGini'],2))
# Create a rank variable based on importance
rankImportance <- varImportance %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))

# Use ggplot2 to visualize the relative importance of variables
ggplot(rankImportance, aes(x = reorder(Variables, Importance), 
                           y = Importance, fill = Importance)) +
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
            hjust=0, vjust=0.55, size = 4, colour = 'red') +
  labs(x = 'Variables') +
  coord_flip() + 
  theme_few()
