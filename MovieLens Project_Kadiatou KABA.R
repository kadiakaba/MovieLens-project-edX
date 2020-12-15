###### Name : Kadiatou KABA ######
########## MOVIELENS PROJECT ###########



##### Loading of Libraries #####

library(tidyverse)
library(caret)
library(lubridate)
library(data.table)
library(ggplot2)
library(psych)
library(rmarkdown)


##### Downloading of the Data from MovieLens Website ####

dl <- tempfile()  
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl) 
ratings <- str_split_fixed(readLines(unzip(dl,"ml-10M100K/ratings.dat")), "\\::", 4) 
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")  
ratings <- as.data.frame(ratings) %>%
  mutate(userId = as.numeric(userId), 
         movieId = as.numeric(levels(movieId))[movieId], 
         rating = as.numeric(levels(rating))[rating], 
         timestamp = as.numeric(levels(timestamp))[timestamp]) 
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3) 
colnames(movies) <- c("movieId", "title", "genres") 
movies <- as.data.frame(movies) %>% 
  mutate(movieId = as.numeric(levels(movieId))[movieId], 
         title = as.character(title), 
         genres = as.character(genres))
movielens <- left_join(ratings, movies, by = "movieId") 
rm(dl, movies, ratings) # remove unneeded variables in global environment

head(movielens)


##### METHODS AND ANALYSIS #####

##### SPLITTING THE DATASET IN TWO : Train (edx) and Test (validation) Datasets #####

set.seed(1) # set seed to 1
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE) # create index with 10% of data for test set
edx <- movielens[-test_index,]  # create edx (train) dataset from test index
temp <- movielens[test_index,]  
validation <- temp %>%  # create final validation (test) dataset from temporary dataset
  semi_join(edx, by = "movieId") %>%  
  semi_join(edx, by = "userId")
removed <- anti_join(temp, validation)  # identify movies and users that were removed from temporary test set
edx <- rbind(edx, removed)  # add movies and users removed from temporary test set to edx train set
rm(test_index, temp, movielens, removed)  # remove unneeded data from global environment

head(edx) 

head(validation) 


##### DATA CLEANING #####

head(edx) 
edx <- edx %>% 
  mutate(timestamp = as_datetime(timestamp)) %>%  
  mutate(year = substring(title, nchar(title) - 6)) %>% 
  mutate(year = as.numeric(substring(year, regexpr("\\(", year) + 1, regexpr("\\)", year) - 1))) %>% 
  separate_rows(genres, sep = "\\|") 
head(edx, 19) 

validation <- validation %>%  
  mutate(timestamp = as_datetime(timestamp)) %>%  
  mutate(year = substring(title, nchar(title) - 6)) %>% 
  mutate(year = as.numeric(substring(year, regexpr("\\(", year) + 1, regexpr("\\)", year) - 1))) %>% 
  separate_rows(genres, sep = "\\|")  



##### Cursory Data Analysis and Visualizations #####

# calculation of average and median rating

avg_rating <- mean(edx$rating)  
med_rating <- median(edx$rating) 


# grouping by Rating

edx_ratings <- edx %>%  
  group_by(rating) %>%  
  summarize(num_ratings = n()) %>%
  arrange(desc(num_ratings))
edx_ratings
edx_ratings %>% # for each rating, plot frequency
  ggplot(aes(rating, num_ratings, color = rating)) +
  geom_point(aes(size = num_ratings)) +
  scale_color_gradientn(colours = rainbow(5)) +
  scale_size_continuous(limits = c(0, 7e+06)) +
  xlim(0,5) +
  labs(x = "Rating", y = "Number of Ratings", title = "Ratings by Rating", color = "Rating", size = "Number of Ratings")

# grouping by Movie

edx_movies <- edx %>%
  group_by(movieId) %>% 
  summarize(num_ratings = n(), avg_rating = mean(rating)) %>% 
  arrange(desc(num_ratings))
headTail(edx_movies)  # display top and bottom movies by number of ratings
edx_movies %>% # for each movie, plot the number of ratings v num of ratings
  ggplot(aes(movieId, num_ratings, color = avg_rating)) +
  geom_point() +
  scale_color_gradientn(colours = rainbow(5)) +
  labs(x = "MovieId", y = "Number of Ratings", title = "Ratings by Movie", color = "Average Rating")

# grouping by User

edx_users <- edx %>% 
  group_by(userId) %>% 
  summarize(num_ratings = n(), avg_rating = mean(rating)) %>%
  arrange(desc(num_ratings)) 
headTail(edx_users) # display top and bottom users by number of ratings
edx_users %>% # for each movie, plot the number of ratings v num of ratings
  ggplot(aes(userId, num_ratings, color = avg_rating)) +
  geom_point() +
  scale_color_gradientn(colours = rainbow(5)) +
  labs(x = "UserId", y = "Number of Ratings", title = "Ratings by User", color = "Average Rating")

# grouping by Genre

edx_genres <- edx %>% 
  group_by(genres) %>% 
  summarize(num_ratings = n(), avg_rating = mean(rating)) %>% 
  arrange(desc(num_ratings)) 
edx_genres
edx_genres %>% # for each genre, plot the number of ratings and average rating
  ggplot(aes(genres, avg_rating)) +
  geom_point(aes(size = num_ratings)) +
  scale_size_continuous(limits = c(0, 7e+06)) +
  labs(x = "Genre", y = "Average Rating", title = "Ratings by Genre", size = "Number of Ratings") +
  theme(axis.text.x = element_text(angle = 90))

# grouping by Year

edx_years <- edx %>% 
  group_by(year) %>% 
  summarize(num_ratings = n(), avg_rating = mean(rating)) %>% 
  arrange(desc(num_ratings)) 
headTail(edx_years) 
edx_years %>% # for each year, plot the number of ratings and average rating
  ggplot(aes(year, avg_rating, size = num_ratings)) +
  geom_point(aes(size = num_ratings)) +
  scale_size_continuous(limits = c(0, 7e+06)) +
  labs(x = "Year", y = "Average Rating", title = "Ratings by Year", size = "Number of Ratings")



##### Define Function to Calculate RMSE ######

rmse <- function(true_ratings, predicted_ratings){  # define function that takes true ratings and predicted ratings and...
  sqrt(mean((true_ratings - predicted_ratings)^2))  # ...calculates residual mean squared error
}


###### DIFFERENT MODELS ######


##### Model 1 - Average #####

average <- mean(edx$rating)  # calculate the average rating
rmse_average <- rmse(validation$rating, average) # calculate rmse for model
model_rmses <- tibble(model = "Average", rmse = rmse_average) # create a table to display all the calculated rmses

##### Model 2 - Movie Effect #####

movie_effect <- edx %>% 
  group_by(movieId) %>% 
  summarize(e_m = mean(rating - average)) # calculate e_m for each movie by taking average of difference between each rating and average rating

movie_pred_ratings <- validation %>% # take testing dataset and...
  left_join(movie_effect, by ="movieId") %>% # ...join it to training data set by movie and...
  mutate(predicted_rating = average + e_m) %>% # ...calculate predicted ratings and...
  pull(predicted_rating)  # ...pull predicted ratings

rmse_movie_effect <- rmse(validation$rating, movie_pred_ratings)  # calculate rmse for model
model_rmses <- bind_rows(model_rmses,
                         tibble(model = "Average + Movie Effect", rmse = rmse_movie_effect)) # add calculated rmse to rmse table


##### Model 3 - User Effect #####

user_effect <- edx %>% 
  group_by(userId) %>%  
  summarize(e_u = mean(rating - average)) # ...calculate e_u for each user by taking average of difference between each rating and average rating

user_pred_ratings <- validation %>% # take test dataset and...
  left_join(user_effect, by = "userId") %>% # ...join it to training dataset by user and...
  mutate(predicted_rating = average + e_u) %>% #...calculate predicted ratings and...
  pull(predicted_rating)  # ...pull predicted ratings

rmse_user_effect <- rmse(validation$rating, user_pred_ratings)  # calculate rmse for model
model_rmses <- bind_rows(model_rmses,
                         tibble(model = "Average + User Effect", rmse = rmse_user_effect)) # add calculated rmse to rmse table


##### Model 4 - Genre Effect #####

genre_effect <- edx %>% 
  group_by(genres) %>% 
  summarize(e_g = mean(rating - average)) # calculate e_g for each genre by taking average of difference between each rating and average rating

genre_pred_ratings <- validation %>% # take test dataset and...
  left_join(genre_effect, by = "genres") %>%  # ...join it to training dataset by genre and...
  mutate(predicted_rating = average + e_g) %>% # ...calculate predicted ratings and...
  pull(predicted_rating)  # pull predicted ratings

rmse_genre_effect <- rmse(validation$rating, genre_pred_ratings)  # calculate rmse for model
model_rmses <- bind_rows(model_rmses,
                         tibble(model = "Average + Genre Effect", rmse = rmse_genre_effect)) # add calculated rmse to rmse table


##### Model 5 - Year Effect #####

year_effect <- edx %>% 
  group_by(year) %>% 
  summarize(e_y = mean(rating - average)) # calculate e_y for each year by taking average of difference between each rating and average rating

year_pred_ratings <- validation %>% # take test dataset and...
  left_join(year_effect, by = "year") %>% # join it to training dataset by time between release and ratings and...
  mutate(predicted_rating = average + e_y) %>% # ...calculate predicted ratings and...
  pull(predicted_rating)  # ...pull predicted ratings

rmse_year_effect <- rmse(validation$rating, year_pred_ratings)  # calculate rmse for model
model_rmses <- bind_rows(model_rmses,
                         tibble(model = "Average + Year Effect", rmse = rmse_year_effect)) # add calculated rmse to rmse table


### FINAL RESULTS ###
##### Model 6 - Combine the Best Effects #####

model_rmses  # display calculated RMSES

movie_effect <- edx %>% # take data and...
  group_by(movieId) %>% # ...group it by movie and...
  summarize(e_m = mean(rating - average)) # calculate e_m for each movie by taking average of difference between each rating and average rating

movie_user_effect <- edx %>% 
  left_join(movie_effect, by = "movieId") %>% 
  group_by(userId) %>% 
  summarize(e_u = mean(rating - average - e_m)) # ... calculate e_u for each user by taking average of difference between each rating and average rating and calculated e_m

movie_user_pred_ratings <- validation %>% # take test dataset and...
  left_join(movie_effect, by = "movieId") %>%  # join it to movie_effect dataset by movie and...
  left_join(movie_user_effect, by = "userId") %>% # join it to movie_effect dataset by movie and...
  mutate(predicted_rating = average + e_m + e_u) %>% # ...calculate predicted ratings and...
  pull(predicted_rating)  # ...pull predicted ratings

best_model_rmse <- rmse(validation$rating, movie_user_pred_ratings)  # calculate rmse for model
model_rmses <- bind_rows(model_rmses,
                         tibble(model = "Average + Movie + User Effects", rmse = best_model_rmse)) # add calculated rmse to rmse table
model_rmses[6,]
