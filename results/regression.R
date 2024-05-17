# install.packages("lme4")
# install.packages("here")
library(lme4)
library(here)
library(ggplot2)
library(lattice)


data <- read.csv(here("results", "sample_with_freq.csv"))

sentence_lengths <- data %>%
  group_by(turn_id, sentence_id_in_turn) %>%
  summarise(sentence_length = n()) %>%
  ungroup()

data <- data %>%
  left_join(sentence_lengths, by = c("turn_id", "sentence_id_in_turn"))

head(data)


data$duration <- as.numeric(data$duration)
data$left_sentence <- as.numeric(data$left_sentence)
data$right_sentence <- as.numeric(data$right_sentence)
data$bidi_sentence <- as.numeric(data$bidi_sentence)
data$left_bigram <- as.numeric(data$left_bigram)
data$right_bigram <- as.numeric(data$right_bigram)
data$bidi_bigram <- as.numeric(data$bidi_bigram)
data$frequency <- as.numeric(data$frequency)
data$n_syllables <- as.numeric(data$n_syllables)

data$word <- as.factor(data$word)

# model <- lmer(log(duration) ~ left_sentence + right_sentence + bidi_sentence + left_bigram + right_bigram + bidi_bigram + log(frequency) + (1 | word), data = data)
# model <- lmer(log(duration) ~ left_sentence + right_sentence + bidi_sentence + left_bigram + right_bigram + bidi_bigram + (1 | word), data = data)
# model <- lm(log(duration) ~ left_sentence + right_sentence + bidi_sentence + left_bigram + right_bigram + bidi_bigram, data = data)
model <- lm(log(duration) ~ left_sentence + right_sentence + bidi_sentence + left_bigram + right_bigram + bidi_bigram + log(frequency), data = data)
# model <- lm(log(duration) ~ (left_sentence + right_sentence + bidi_sentence + left_bigram + right_bigram + bidi_bigram ) * sentence_length, data = data)

summary(model)
# Extract residuals
residuals <- residuals(model)
fitted <- fitted(model)

plot(fitted, residuals)
abline(h = 0, col = "red")
title("Residuals vs Fitted Values")

qqnorm(residuals)
qqline(residuals, col = "red")
title("Normal Q-Q Plot")

hist(residuals, breaks = 30, main = "Histogram of Residuals", xlab = "Residuals")

sqrt_abs_resid <- sqrt(abs(residuals))
plot(fitted, sqrt_abs_resid)
abline(h = 0, col = "red")
title("Scale-Location Plot")

cooks_d <- cooks.distance(model)
plot(cooks_d, type = "h", main = "Cook's Distance", ylab = "Cook's Distance")

# # Residuals vs Fitted
# ggplot(data.frame(fitted, residuals), aes(x = fitted, y = residuals)) +
#   geom_point() +
#   geom_hline(yintercept = 0, col = "red") +
#   labs(title = "Residuals vs Fitted Values", x = "Fitted values", y = "Residuals")

# # Normal Q-Q Plot
# ggplot(data.frame(sample = residuals), aes(sample = sample)) +
#   stat_qq() +
#   stat_qq_line(col = "red") +
#   labs(title = "Normal Q-Q Plot", x = "Theoretical Quantiles", y = "Sample Quantiles")

# # Histogram of Residuals
# ggplot(data.frame(residuals), aes(x = residuals)) +
#   geom_histogram(bins = 30) +
#   labs(title = "Histogram of Residuals", x = "Residuals", y = "Frequency")

# # Scale-Location Plot
# ggplot(data.frame(fitted, sqrt_abs_resid), aes(x = fitted, y = sqrt_abs_resid)) +
#   geom_point() +
#   geom_hline(yintercept = 0, col = "red") +
#   labs(title = "Scale-Location Plot", x = "Fitted values", y = "sqrt(abs(Residuals))")

# Summary of the model




