# Capstone-Project

#### title: Predict customer churn #####

library(dplyr)
library(gains)
library(irr)
library(ggplot2)
library(caret)
library(e1071)
library(dataQualityR)
library(rpart)
library(tidyr)
library(randomForest)
library(pROC)
library(gridExtra)
library(car)


setwd("C:\\Jig18081")
getwd()
churn<-read.csv("telecomfinal.csv")
names(churn)
str(churn)
summary(churn)


#1. ### Create data quality report ###
num.file<-paste("C:\\Jig18081\\dqr_num.csv", sep="")
cat.file<-paste("C:\\Jig18081\\dqr_cat.csv", sep="")
checkDataQuality(data = churn,out.file.num = num.file,out.file.cat = cat.file)

## After creating data quality report, there are 60 continuos var and 21 categorical var. Based on max missing observation decided to drop; 
#Continuous_Var: (retdays, numbcars, income) Categorical_Var: (solflag, wrkwoman, div_type, occu1, proptype, cartype, children, mailordr, mailresp, dwllsize, dwlltype)
#Create new data set with 67 variables (Dropping 14)
churn_1 <- select (churn, -c(retdays, numbcars, income, solflag, wrkwoman, div_type, occu1, proptype, cartype, children, mailordr, mailresp, dwllsize, dwlltype))

#2. ### Variable Profiling ###
## Missing Values treatment: Profiling variables by decile binning method.
## Other methods includes; Generalized Imputation (Mean,Mode,Median), Predictive Model Method, KNN imputation, etc
## There are other packages to impute missing value from a variables like, MICE(Multivariate Imputation with Chained Equation),Amelia,missForest,Hmisc,mi

### Continous Variable profiling with decile binning method
churn_1%>%mutate(dec=ntile(age2,n=2))%>%count(churn,dec)%>%filter(churn==1)->dat1
dat1$N<-unclass(churn_1%>%mutate(dec=ntile(age2,n=2))%>%count(dec)%>%unname())[[2]]
dat1$churn_perc<-round(dat1$n/dat1$N,3)
dat1$GreaterThan<-unclass(churn_1%>%mutate(dec=ntile(age2,n=2))%>%group_by(dec)%>%summarise(min(age2)))[[2]]
dat1$LessThan<-unclass(churn_1%>%mutate(dec=ntile(age2,n=2))%>%group_by(dec)%>%summarise(max(age2)))[[2]]
dat1$varname<-rep("age2",nrow(dat1))

### Imputation to Continous variables
churn_1$age1[is.na(churn_1$age1)]<-40
churn_1$hnd_price[is.na(churn_1$hnd_price)]<-89.989
churn_1$avg6mou[is.na(churn_1$avg6mou)]<-4196
churn_1$avg6qty[is.na(churn_1$avg6qty)]<-1607
churn_1$change_mou[is.na(churn_1$change_mou)]<- -4.5 #p50
churn_1$mou_Mean[is.na(churn_1$mou_Mean)]<- 54.5
churn_1$totmrc_Mean[is.na(churn_1$totmrc_Mean)]<- 42.49
churn_1$rev_Range[is.na(churn_1$rev_Range)]<- 58.25
churn_1$mou_Range[is.na(churn_1$mou_Range)]<- 25
churn_1$ovrrev_Mean[is.na(churn_1$ovrrev_Mean)]<- 0
churn_1$rev_Mean[is.na(churn_1$rev_Mean)]<- 10.0345
churn_1$ovrmou_Mean[is.na(churn_1$ovrmou_Mean)]<- 0
churn_1$roam_Mean[is.na(churn_1$roam_Mean)]<- 0
churn_1$da_Mean[is.na(churn_1$da_Mean)]<- 0
churn_1$da_Range[is.na(churn_1$da_Range)]<- 0
churn_1$datovr_Mean[is.na(churn_1$datovr_Mean)]<- 0
churn_1$datovr_Range[is.na(churn_1$datovr_Range)]<- 0
tele$avg6qty[is.na(tele$avg6qty)]<-1607
tele$ovrmou_Mean[is.na(tele$ovrmou_Mean)]<- 0

###Individual profiles are stored separately as .csv file after imputation.
write.csv(dat1,"C:\\Jig18081\\Var_profiles\\hnd_price.csv")

### Categorical Variable profiling with event rate for each level
churn_1%>%count(churn,levels=agecat)%>%filter(churn==1)->datC1
datC1$N<-unclass(churn_1%>%filter(agecat%in%datC1$levels)%>%count(agecat))[[2]]
datC1$ChurnPerc<-round(datC1$n/datC1$N,5)
datC1$Var.Name<-rep("agecat",nrow(datC1))

### Imputation to Categorical variables
churn_1$hnd_webcap[is.na(churn_1$hnd_webcap)]<-"WC"
churn_1$prizm_social_one[is.na(churn_1$prizm_social_one)]<-"T"
churn_1$marital[is.na(churn_1$marital)]<-"S"
churn_1$ethnic[is.na(churn_1$ethnic)]<-"M"
churn_1$car_buy[is.na(churn_1$car_buy)]<-"New"
churn_1$area[is.na(churn_1$area)]<-"OHIO AREA"
tele$agecat[is.na(tele$agecat)]<- 0

###Individual profiles are stored separately as .csv file after imputation.
write.csv(datC1,"C:\\Jig18081\\Var_profiles\\crclscod.csv")

# After Variable profiling for both Continous and categorical variable decided to drop few more Var
tele<-select(churn_1, -c(forgntvl, mtrcycle, truck, csa, car_buy, age2, crclscod,ethnic)) 
#Note: Var "csa,crclscod has a too many levels
tele<-na.omit(tele) 


# merge variables removed from from corr matrix and with unknown error
#tele1 <- tele
#tele2 <- select(churn_1, c(Customer_ID, adjqty, ovrmou_Mean, plcd_vce_Mean, avg3mou, avg6qty, adjrev, plcd_dat_Mean, asl_flag, refurb_new ))
#merge (x=tele1,y=tele2,by="Customer_ID",all.x = TRUE) -> tele_new
#tele <- tele_new
#write.csv(tele,"C:\\Jig18081\\tele.csv")
#tele<-read.csv("C:\\Jig18081\\tele.csv")

#3. ### Dummy Variable Creation ###

## Check number of uniques value for each of the column to find out columns which we can convert to factors
sapply(tele, function(x) length(unique(x)))

### Create Dummy variables for prizm_social_one, marital, asl_flag, refurb_new, hnd_webcap
unique(tele$hnd_webcap)
tele$prizm_social_one_C<-ifelse(tele$prizm_social_one=="C",1,0)
tele$prizm_social_one_R<-ifelse(tele$prizm_social_one=="R",1,0)
tele$prizm_social_one_S<-ifelse(tele$prizm_social_one=="S",1,0)
tele$prizm_social_one_T<-ifelse(tele$prizm_social_one=="T",1,0)
tele$prizm_social_one_U<-ifelse(tele$prizm_social_one=="U",1,0)
tele$marital_A<-ifelse(tele$marital=="A",1,0)
tele$marital_B<-ifelse(tele$marital=="B",1,0)
tele$marital_M<-ifelse(tele$marital=="M",1,0)
tele$marital_S<-ifelse(tele$marital=="S",1,0)
tele$marital_U<-ifelse(tele$marital=="U",1,0)
tele$asl_flag_N<-ifelse(tele$asl_flag=="N",1,0)
tele$asl_flag_Y<-ifelse(tele$asl_flag=="Y",1,0)
tele$refurb_new_N<-ifelse(tele$refurb_new=="N",1,0)
tele$refurb_new_R<-ifelse(tele$refurb_new=="R",1,0)
tele$hnd_webcap_WCMB<-ifelse(tele$hnd_webcap=="WCMB",1,0)
tele$hnd_webcap_WC<-ifelse(tele$hnd_webcap=="WC",1,0)
tele$hnd_webcap_UNKW<-ifelse(tele$hnd_webcap=="UNKW",1,0)
tele<-select(tele, -c(prizm_social_one,marital, asl_flag, refurb_new, hnd_webcap))
tele<-select(tele, -c(asl_flag, refurb_new))

for (i in c("prizm_social_one_C", "prizm_social_one_R", "prizm_social_one_S", "prizm_social_one_T", "prizm_social_one_U", 
            "marital_A","marital_B" , "marital_M" ,"marital_S", "marital_U","asl_flag_N" , "asl_flag_Y","refurb_new_N",
            "refurb_new_R", "hnd_webcap_WCMB", "hnd_webcap_WC", "hnd_webcap_UNKW", "churn"))
{
  tele[,i]=as.factor(tele[,i])
}

#4. ### Converting Continuous Var to Categorical Var ####
summary(tele$age1)
str(tele$age1)
tele$age1 <-as.numeric(tele$age1)
tele$age_cat<- cut(tele$age1,breaks = c(-Inf,10,20,30,40,50,60,70,80,90,Inf) )
summary(tele$age_cat)
str(tele$age_cat)

#remove converted variable
names(tele)
#tele<-tele[,-25]
summary(tele)

#Since the minimum tenure is 6 months and maximum tenure is 61 months, group them into Upto6, 6-12,12-24,24-48, >48 months
summary(tele$months) 
tele$months_Cat<- cut(tele$months, breaks = c(-Inf,6,12,24,48,Inf))
summary((tele$months_Cat))
str(tele$months_Cat)

#remove converted variable
names(tele)
#tele<-tele[,-10]

5. ### Create Derived Variables#####
tele<-mutate(tele,comp_per_vce=comp_vce_Mean/plcd_vce_Mean, comp_per_dat=comp_dat_Mean/plcd_dat_Mean)

# Replace NaN with 0
is.nan.data.frame <- function(x)
  do.call(cbind,lapply(x,is.nan))

tele[is.nan(tele)] <- 0

#6. #####Check for collinearity######
numeric.var <- sapply(tele, is.numeric)
corr.matrix <- cor(tele[,numeric.var])
print(corr.matrix)
write.csv(corr.matrix,"C:\\Jig18081\\corr_matrix.csv")
corrplot(corr.matrix,main="\n\nCorrelation Plot for Numeric Variables", method="number")

#highly correlated Variables adjqty, ovrmou_Mean, plcd_vce_Mean, avg3mou, avg6qty, adjrev, plcd_dat_Mean are removed
tele <- select(tele, -c(Customer_ID,adjqty, ovrmou_Mean, plcd_vce_Mean, avg3mou, avg6qty, adjrev, plcd_dat_Mean))

#### Outlier Treatement####
par("mar")
par(mar=c(1,1,1,1))
par(mfrow=c(5,10))
# Outliers in data can distort predictions and affect the accuracy.

# factor variables - area, churn, prizm_social_one_C,prizm_social_one_R,prizm_social_one_S,prizm_social_one_T,prizm_social_one_U
#            marital_A,marital_B,marital_M,marital_S,marital_U,hnd_webcap_WCMB,hnd_webcap_WC,hnd_webcap_UNKW,age_cat,months_Cat
#              asl_flag_N,asl_flag_Y,refurb_new_N,refurb_new_R

#Outlier treatment using capping method
list<-names(tele)
list

#removing categorical variables
list <- list[-c(1,25,34,45:59,69:72)]

for(i in 1:length(list))
{
  boxplot(tele[,list[i]],main=list[i])
}

## Outlier treatment
for (i in 1:length(list))
{
  qnt <- quantile(tele[,list[i]], probs = c(.25, .75), na.rm = T)
  caps <- quantile(tele[,list[i]], probs = c(.05, .95), na.rm = T)
                   H <- 1.5*IQR(tele[,list[i]], na.rm = T)
                   tele[,list[i]][tele[,list[i]] < (qnt[1] - H)] <- caps[1]
                   tele[,list[i]][tele[,list[i]] > (qnt[2] + H)] <- caps[2]
}

# checking after treatment
for (i in 1:length(list))
{
  boxplot(tele[,list[i]],main=list[i])
}

### Exploratory Data Analysis (EDA)

#1. Churn Ration by Categorical Predictors
exp1 <- ggplot(train, aes(area, fill = churn)) + geom_bar(position = "fill") + labs(x = "Location", y = "ChurnRate") + theme(legend.position = "none") + coord_flip()
exp2 <- ggplot(train, aes(age_cat, fill = churn)) + geom_bar(position = "fill") + labs(x = "Age", y = "ChurnRate") + theme(legend.position = "none") + coord_flip()
exp3 <- ggplot(train, aes(months_Cat, fill = churn)) + geom_bar(position = "fill") + labs(x = "Tenure", y = "ChurnRate") + theme(legend.position = "none") + coord_flip()
grid.arrange(exp1,exp2,exp3, ncol=1,nrow=3, top="Churn/Non-churn Proportion")

#3. Explore distributions by continuous predictors
exp4 <- ggplot(train, aes(churn, mou_Mean, fill = churn )) + geom_boxplot(aplha = 0.8) + theme(legend.position = "null")
exp5 <- ggplot(train, aes(churn, eqpdays, fill = churn )) + geom_boxplot(aplha = 0.8) + theme(legend.position = "null")
exp6 <- ggplot(train, aes(churn, avgmou, fill = churn )) + geom_boxplot(aplha = 0.8) + theme(legend.position = "null")
exp7 <- ggplot(train, aes(churn, ovrrev_Mean, fill = churn )) + geom_boxplot(aplha = 0.8) + theme(legend.position = "null")
exp8 <- ggplot(train, aes(churn, comp_per_vce, fill = churn )) + geom_boxplot(aplha = 0.8) + theme(legend.position = "null")
exp9 <- ggplot(train, aes(churn, mou_Range, fill = churn )) + geom_boxplot(aplha = 0.8) + theme(legend.position = "null")
exp10 <- ggplot(train, aes(churn, uniqsubs, fill = churn )) + geom_boxplot(aplha = 0.8) + theme(legend.position = "null")
exp11 <- ggplot(train, aes(churn, actvsubs, fill = churn )) + geom_boxplot(aplha = 0.8) + theme(legend.position = "null")
grid.arrange(exp4, exp5, exp6, exp7, exp8, exp9, exp10, exp11, ncol=4, nrow=2, top="Distribution of Continuous Variable")

### Logistic regression Model Fitting
### The exploratory analysis already revealed which area, age_cat and Month_cat have higher than average churn,
### The area, age_cat and month_cat predictors will not be considered in the model. 
##Instead model will be used to interpret remaining contributors of churn

telecom <- select(tele, -c(area, age_cat, months_Cat))
set.seed(2019)
index <- sample(nrow(telecom),nrow(telecom)*0.7, replace = F)

# Let's create train and validation data sets
train <- telecom[index,]
test <- telecom[-index,]

# Lets confirm the datasets
dim(train);dim(test)

# Lets build the model taking "churn" as target var and all other as predictors
model <- glm(train$churn~., data=train, family = "binomial")
summary(model)
var_imp <- varImp(modelfinal)

#keeping only significant variables (***)
model2 <- glm(formula = churn ~ mou_Mean + totmrc_Mean + mou_Range + change_mou + eqpdays + ovrrev_Mean + rev_Mean + avgmou + avg6mou + hnd_price + actvsubs +
                uniqsubs + totrev + marital_M + marital_S + comp_per_vce + asl_flag_N + refurb_new_N, family = "binomial",data = train)
summary(model2)
step(model2, direction = "both")

modelfinal <- glm(formula = churn ~ mou_Mean + totmrc_Mean + mou_Range + change_mou + 
                eqpdays + ovrrev_Mean + rev_Mean + avgmou + avg6mou + hnd_price + 
                actvsubs + uniqsubs + totrev + marital_M + marital_S + comp_per_vce + 
                asl_flag_N + refurb_new_N, family = "binomial", data = train)
summary(modelfinal)

#Validation of Model/ Model Performance
pred <- predict(modelfinal,type = "response", newdata = test)

#Let's check the rate of 1, according to that we will set a cutoff value
table(train$churn)/nrow(train)

#Assume anything with probability of 1 greater than (0.2379598 ~ 0.25) will be 1 else 0. where, 1 is churn and 0 is no churn
glm.pred <- ifelse(pred>0.25,1,0)
kappa2(data.frame(test$churn,glm.pred))
anova(modelfinal, test = "Chisq")

#Confusion matrix
test$result <- ifelse(pred>0.25,1,0)
as.factor(test$churn) -> test$churn
as.factor(test$result) -> test$result
confusionMatrix(test$result,test$churn,positive = "1")

#Check for VIF
vif(modelfinal)

#Extract coeff
coeff <- as.data.frame(round(coef(modelfinal),5))

#Odds Ratio
#One of the interesting performance measurements in logistic regression is odds ratio. 
#Odds ration is what the odds of an event is happening

exp(cbind(OR=coef(modelfinal), confint(modelfinal)))

#Outlier Analysis using Cook's Distance method
cooksd <- cooks.distance(modelfinal)
plot(cooksd, pch="*",cex=2, main ="Influential Obs by cooks distance") #plotting
abline(h = 13*mean(cooksd, na.rm = T), col="red") # Create cutoff value
text (x=1:length(cooksd)+1,y=cooksd, labels = ifelse(cooksd>13*mean(cooksd, na.rm =T), names(cooksd),""),col="red",cex=1 ) #label

#Find out influential rows
influential <- as.numeric(names(cooksd)[(cooksd > 13*mean(cooksd, na.rm = T))]) 
head(train[influential,])

#Outliers Test
car::outlierTest(modelfinal)
#the output suggests that observation in row 37663 is most extreme. 
#Likewise, all the outliers can be identified and removed from the data.

#Creating customer segments based on revenue of predicted probability of churn
pred <- predict(modelfinal, type="response", newdata = test)
test$prob <- predict(modelfinal, type="response", newdata = test)
quantile(test$prob, prob = c(.1,.2,.3,.4,.5,.6,.7,.8,.9,1))
pred_seg <- ifelse(pred<.25,"low_score",ifelse(pred>=.25 & pred<.5,"med_score","high_score"))
table(pred_seg,test$churn)

quantile(test$totrev,prob = c(.1,.2,.3,.4,.5,.6,.7,.8,.9,1))
rev_levels <- ifelse(test$totrev<670.728,"low_rev",ifelse(test$totrev>=670.728 & test$totrev<1134.148,"med_rev","high_rev"))
table(rev_levels)
table(pred_seg,rev_levels)

#gain Chart
test$churn <- as.numeric(test$churn)
gains(test$churn, predict(modelfinal,type = "response",newdata = test),groups = 10)

#The confusion matrix is used to review the prediction vs actual. It is important to note that while the accuracy of the model appears to be high, 
#a lot that accuracy is driven by disproportionate number of non churn customers predicted correctly.
# To benchmark the logistic model, compare its performance against another model, a single decision tree model (CART).

#<<<<<< Not able to identify the error/required packages for below predictive models, code is only for reference >>>>>>>>>>


#All models will be built using 5-fold validation
modelCtrl<- trainControl(method = "repeatedcv",number = 3)
#CART
set.seed(2019)
cart.model <- train(churn ~., method = "rpart2", data = train, trControl = modelCtrl)
cart.pred <- predict(cart.model, newdata = test)
#Model accuracy
cart.accuracy <- 1-mean(cart.pred != test$churn)
table(actual=test$churn,predicted=cart.pred)

#Random Forest
set.seed(2019)
rf.model <- train(train$churn ~., method = "rf", data = train)
rf.pred <- predict(rf.model, newdata = test)
#Model accuracy
rf.accuracy <- 1-mean(rf.pred != test$churn)
table(actual=test$churn,predicted=rf.pred)

#Models Performance
results <- resamples(list(CART=cart.model, RF=rf.model))
dotplots(results)

#Pick the Winner from RandomForest, CART and Logistic
glm.roc <- roc(response = test$churn, predictor =as.numeric((glm.pred)))
cart.roc <- roc(response = test$churn, predictor =as.numeric((cart.pred)))
rf.roc <- roc(response = test$churn, predictor =as.numeric((rf.pred)))

plot(glm.roc, legacy.axes = TRUE, print.auc.y = 0.4, print.auc =TRUE)
plot(cart.roc, legacy.axes = TRUE, print.auc.y = 0.5, print.auc =TRUE)
plot(rf.roc, legacy.axes = TRUE, print.auc.y = 0.6, print.auc =TRUE)

legend(0.0,0.2, c("Random Forest", "CART", "Logistic"), lty = c(1,1), lwd = c(2,2), col = c("red", "blue", "black"), cex = 0.75)

### Conclusion ###

#Please refer the power point presentation.
