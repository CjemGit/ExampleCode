#Logistic and linear regression models to inform strategy in League of Legends, written in R
#Data downloaded from the league of legends API

library(tidyjson)
library(dplyr)

# Example JSON
# match_json <- ’
# [
#    {
#       "matchid": 2974226866,
#       "stats": {
#          "assists": 19,
#          "goldEarned": 17399,
#          "kills": 6,
#          "lane": "TOP",
#          "totalDamageDealt": 200693,
#          "winner": true
# } },

fpete <- match_json %>%
  gather_array %>%
  spread_values(matchid = jstring("matchid")) %>%
  enter_object("stats") %>%
  spread_values(
   stats.assists = jnumber("assists"),
   stats.goldEarned = jnumber("goldEarned"),
   stats.kills = jnumber("kills"),
   stats.lane = jstring("lane"),
   stats.totalDamageDealt = jnumber("totalDamageDealt"),
   stats.winner = jstring("winner")
   ) %>%
select(matchid, stats.assists, stats.goldEarned, stats.kills,
    stats.totalDamageDealt, stats.winner, stats.lane)

#linear models to predict the relationship between game values and Gold Earned

lm(formula = Gold ~ Damage, data = fpete)

lm(formula = Gold ~ Damage + rKills + rAssists, data = fpete)

lm(formula = Gold ~ Damage + rKills + rAssists + DumJung + DumMid +DumTop, data = fpete)

#binary linear models to predict the relationship between game values and win or lose

glm(formula = Win ~ Gold + Damage, family = binomial(link = "logit"), data = train)

glm(formula = Win ~ Gold + Damage + rKills + rAssists, family = binomial(link = "logit"), data = train)

glm(formula = Win ~ Gold + Damage + rKills + rAssists + DumJung + DumMid + DumTop, family = binomial(link = "logit"), data =train)

# # EXAMPLE OUTPUT FROM FINAL CALL
# # THERE IS A STRONGER RELATIONSHIP BETWEEN ASSISTS AND WINNING THAN SCORING KILLS AND WINNING

# Coefficients:
#           Estimate Std. Error z value Pr(>|z|)
# (Intercept) -3.406e+00 4.677e-01 -7.282 3.28e-13 ***
# Gold 3.065e-04 7.725e-05 3.967 7.27e-05 ***
# Damage -1.678e-05 3.308e-06 -5.072 3.93e-07 ***
# rKills 2.716e-01 1.306e-01 2.080 0.0375 *
# rAssists 3.087e-01 1.182e-01 2.612 0.0090 **
# DumJung 7.894e-01 8.029e-01 0.983 0.3255
# DumMid 4.291e-01 3.144e-01 1.365 0.1723
# DumTop 6.857e-01 2.938e-01 2.334 0.0196 *
