install.packages("hockeyR")
install.packages("dplyr")
...
remotes::install_github("danmorse314/hockeyR")

library(hockeyR)
library(dplyr)
library(tidyr)
library(purrr)
library(stringr)
library(Matrix)
library(data.table)

#Load the 2023/24 into R
pbp <- hockeyR::load_pbp(season = 2024)

#Filter out the shots data
shots <- pbp %>%
  filter(event_type %in% c("SHOT","GOAL","MISSED_SHOT","BLOCKED_SHOT")) %>%
  filter(strength_state %in% c("5v5")) %>%
  filter(!is.na(x), !is.na(y)) %>%
  mutate(goal = as.integer(event_type == "GOAL"))

shots <- shots %>%
  mutate(
    shot_distance = as.numeric(shot_distance),
    shot_angle    = as.numeric(shot_angle)
  ) %>%
  filter(is.finite(shot_distance), is.finite(shot_angle)) %>%
  drop_na(goal, shot_distance, shot_angle)

#XgBoost
library(dplyr)
library(tidyr)
library(xgboost)

#Shots Base for xgboost
shots_ml <- shots %>%
  mutate(
    goal          = as.numeric(goal),           # Zielvariable 0/1
    shot_distance = as.numeric(shot_distance),
    shot_angle    = as.numeric(shot_angle),
    x             = as.numeric(x_fixed),
    y             = as.numeric(y_fixed),
    secondary_type = factor(secondary_type)
  ) %>%
  drop_na(goal, shot_distance, shot_angle, x, y, secondary_type)

#XgBoot
xgb_formula <- ~ shot_distance + shot_angle + x + y + secondary_type

mm <- model.matrix(xgb_formula, data = shots_ml)

label <- shots_ml$goal

dtrain <- xgb.DMatrix(data = mm, label = label)

set.seed(123)

xgb_fit <- xgboost(
  data          = dtrain,
  objective     = "binary:logistic",  # Output = Wahrscheinlichkeit für Tor
  eval_metric   = "logloss",
  nrounds       = 200,
  max_depth     = 4,
  eta           = 0.1,
  subsample     = 0.8,
  colsample_bytree = 0.8,
  verbose       = 1
)

shots_ml$xG_xgb <- predict(xgb_fit, newdata = dtrain)

home_cols <- paste0("home_on_", 1:6)
away_cols <- paste0("away_on_", 1:6)

#Distribute the xG value for each shot on every offense player that was on the ice for the specific shot, gives xGF
xGF <- bind_rows(
  shots_ml %>%
    filter(event_team_type == "home") %>%
    select(xG_xgb, all_of(home_cols)) %>%
    pivot_longer(all_of(home_cols), names_to = "slot", values_to = "player"),
  shots_ml %>%
    filter(event_team_type == "away") %>%
    select(xG_xgb, all_of(away_cols)) %>%
    pivot_longer(all_of(away_cols), names_to = "slot", values_to = "player")
) %>%
  group_by(player) %>%
  summarise(xGF = sum(xG_xgb, na.rm = TRUE), SF = n())

#Distribute the xG value for each shot on every defense player that was on the ice for the specific shot, gives xGA
xGA <- bind_rows(
  shots_ml %>%
    filter(event_team_type == "home") %>%
    select(xG_xgb, all_of(away_cols)) %>%
    pivot_longer(all_of(away_cols),names_to = "slot",values_to = "player"),
  shots_ml %>%
    filter(event_team_type == "away") %>%
    select(xG_xgb, all_of(home_cols)) %>%
    pivot_longer(all_of(home_cols),names_to = "slot",values_to = "player")
) %>%
  filter(!is.na(player), player != "") %>%
  group_by(player) %>%
  summarise(
    xGA = sum(xG_xgb, na.rm = TRUE),
    SA  = n(),
    .groups = "drop")

#Put all into one Dataset and calculate the xGDiff
xG_by_player <- full_join(xGF, xGA, by = "player") %>%
  mutate(xGF = replace_na(xGF, 0), xGA = replace_na(xGA, 0), xGDiff = xGF - xGA) 

#Include the IceTimes
IceTime <- IceTime2023_24 %>%
  mutate(
    # Zeitanteil extrahieren: Differenz zur Mitternacht des gleichen Tages
    TOI_min = as.numeric(difftime(TOI_EV, as.Date(TOI_EV), units = "mins")),
    TOI_sec = TOI_min * 60,
    Total_min = GP * TOI_min,
    Total_IceTime = Total_min / 60
  )

#Calculate the per 60 xG values
xG_by_player <- xG_by_player %>%
  left_join(IceTime, by = c("player" = "Player"))

xG_by_player <- xG_by_player %>%
  mutate(
    xGF60 = (xGF * 60) / Total_IceTime,
    xGA60 = (xGA * 60) / Total_IceTime,
    xG60 = (xGDiff * 60) / Total_IceTime
  )

xG_by_player_15more <- xG_by_player %>%
  filter(GP > 14)

#Part 2, we now have the xG values per 60 and now want to calculate the weighted xG based on player strength on the ice
#Every shot needs an ID
shots_withid <- shots_ml %>%
  arrange(game_id, event_idx) %>%
  mutate(shot_id = row_number())

#Defense-Skill per Player: xGA60
def_strength <- xG_by_player %>%
  select(player, xGA60) %>%
  filter(!is.na(player), !is.na(xGA60))

league_xGA60 <- mean(def_strength$xGA60, na.rm = TRUE)

#Mean defense quality of opponents per shot
def_quality <- bind_rows(
  shots_withid %>%
    filter(event_team_type == "home") %>%
    select(shot_id, all_of(away_cols)) %>%
    pivot_longer(all_of(away_cols), names_to = "slot", values_to = "def_player") %>%
    left_join(def_strength, by = c("def_player" = "player")) %>%
    mutate(xGA60 = ifelse(is.na(xGA60), league_xGA60, xGA60)) %>%
    group_by(shot_id) %>%
    summarise(mean_xGA60 = mean(xGA60), .groups = "drop"),
  shots_withid %>%
    filter(event_team_type == "away") %>%
    select(shot_id, all_of(home_cols)) %>%
    pivot_longer(all_of(home_cols), names_to = "slot", values_to = "def_player") %>%
    left_join(def_strength, by = c("def_player" = "player")) %>%
    mutate(xGA60 = ifelse(is.na(xGA60), league_xGA60, xGA60)) %>%
    group_by(shot_id) %>%
    summarise(mean_xGA60 = mean(xGA60), .groups = "drop")
)

library(scales)

# Inverte: low mean_xGA60 means high inv_scaled
def_quality <- def_quality %>%
  mutate(
    inv_strength = max(mean_xGA60, na.rm = TRUE) - mean_xGA60,
    inv_scaled   = rescale(inv_strength, to = c(0, 1))
  )

lambda_def <- 1

#Weighted xG per shot
shots_weighted_xGA <- shots_withid %>%
  left_join(def_quality, by = "shot_id") %>%
  mutate(
    inv_scaled      = replace_na(inv_scaled, 0),
    xG_weighted_def = xG_xgb * (1 + lambda_def * inv_scaled)
  )


#Offensive Skill per Player: xGF60
off_strength <- xG_by_player %>%
  select(player, xGF60) %>%
  filter(!is.na(player), !is.na(xGF60))

league_xGF60 <- mean(off_strength$xGF60, na.rm = TRUE)

#Mean offense quality per shot
off_quality <- bind_rows(
  shots_withid %>%
    filter(event_team_type == "home") %>%
    select(shot_id, all_of(home_cols)) %>%
    pivot_longer(all_of(home_cols), names_to = "slot", values_to = "off_player") %>%
    left_join(off_strength, by = c("off_player" = "player")) %>%
    mutate(xGF60 = ifelse(is.na(xGF60), league_xGF60, xGF60)) %>%
    group_by(shot_id) %>%
    summarise(mean_xGF60 = mean(xGF60), .groups = "drop"),
  shots_withid %>%
    filter(event_team_type == "away") %>%
    select(shot_id, all_of(away_cols)) %>%
    pivot_longer(all_of(away_cols), names_to = "slot", values_to = "off_player") %>%
    left_join(off_strength, by = c("off_player" = "player")) %>%
    mutate(xGF60 = ifelse(is.na(xGF60), league_xGF60, xGF60)) %>%
    group_by(shot_id) %>%
    summarise(mean_xGF60 = mean(xGF60), .groups = "drop")
)

# Invert: high mean_xGF60 means low weighting,
range_vals_off <- range(off_quality$mean_xGF60, na.rm = TRUE)

off_quality <- off_quality %>%
  mutate(
    inv_strength = range_vals_off[2] - mean_xGF60,
    inv_scaled   = rescale(inv_strength, to = c(0, 1))
  )

lambda_off <- 1

shots_weighted_xGF <- shots_withid %>%
  left_join(off_quality, by = "shot_id") %>%
  mutate(
    inv_scaled      = replace_na(inv_scaled, 0),
    xG_weighted_off = xG_xgb * (1 + lambda_off * inv_scaled)
  )

sw_def <- shots_weighted_xGA %>%
  mutate(xGw = xG_weighted_def)

wxGF_by_player <- bind_rows(
  sw_def %>%
    filter(event_team_type == "home") %>%
    select(xGw, all_of(home_cols)) %>%
    pivot_longer(all_of(home_cols), names_to = "slot", values_to = "player"),
  sw_def %>%
    filter(event_team_type == "away") %>%
    select(xGw, all_of(away_cols)) %>%
    pivot_longer(all_of(away_cols), names_to = "slot", values_to = "player")
) %>%
  filter(!is.na(player), player != "") %>%
  group_by(player) %>%
  summarise(
    wxGF = sum(xGw, na.rm = TRUE),
    wSF  = n(),
    .groups = "drop"
  ) %>%
  arrange(desc(wxGF))


#wxGA
sw_off <- shots_weighted_xGF %>%
  mutate(xGw = xG_weighted_off)

wxGA_by_player <- bind_rows(
  sw_off %>%
    filter(event_team_type == "home") %>%
    select(xGw, all_of(away_cols)) %>%
    pivot_longer(all_of(away_cols), names_to = "slot", values_to = "player"),
  sw_off %>%
    filter(event_team_type == "away") %>%
    select(xGw, all_of(home_cols)) %>%
    pivot_longer(all_of(home_cols), names_to = "slot", values_to = "player")
) %>%
  filter(!is.na(player), player != "") %>%
  group_by(player) %>%
  summarise(
    wxGA = sum(xGw, na.rm = TRUE),
    wSA  = n(),                 
    .groups = "drop"
  ) %>%
  arrange(wxGA) 


#Put all into one table
wxG_values <- xG_by_player %>%
  select(player, xGF, xGA, Total_IceTime, GP, Tm) %>%   # Basiswerte
  left_join(wxGF_by_player %>% select(player, wxGF), by = "player") %>%
  left_join(wxGA_by_player %>% select(player, wxGA), by = "player") %>%
  mutate(
    wxGF   = replace_na(wxGF, 0),
    wxGA   = replace_na(wxGA, 0),
    wxG    = wxGF - wxGA,
    wxGF60 = ifelse(Total_IceTime > 0, wxGF * 60 / Total_IceTime, NA_real_),
    wxGA60 = ifelse(Total_IceTime > 0, wxGA * 60 / Total_IceTime, NA_real_),
    wxG60  = wxGF60 - wxGA60
  )

wxG_values_filter <- wxG_values %>%
  filter(GP > 14)

#Plotting
library(dplyr)
library(ggplot2)
library(ggrepel)


plot_diff <- xG_by_player %>%
  select(player, Tm, xG60) %>%
  left_join(wxG_values %>% select(player, wxG60), by = "player") %>%
  mutate(
    diff = wxG60 - xG60 
  ) %>%
  filter(!is.na(diff)) %>%
  arrange(desc(diff))

top20 <- bind_rows(
  plot_diff %>% slice_head(n = 10),   # größte Profiteure
  plot_diff %>% slice_tail(n = 10)    # größte Benachteiligte
) %>%
  arrange(diff)

#Plot 1
ggplot(top20, aes(x = reorder(player, diff), y = diff, fill = diff > 0)) +
  geom_col(alpha = 0.9) +
  coord_flip() +
  scale_fill_manual(values = c("TRUE" = "#4C9F70", "FALSE" = "#CC444B")) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "none",
    panel.grid.minor = element_blank()
  ) +
  labs(
    title = "Most Gains and Losses",
    x = "",
    y = "Difference in xG60",
  )


#Plot 2
library(dplyr)
library(ggplot2)
library(ggrepel)


cluster_df <- wxG_values %>%
  filter(GP > 20, !is.na(wxGF60), !is.na(wxGA60)) %>%
  select(player, Tm, wxGF60, wxGA60)

set.seed(42)
clusters <- kmeans(cluster_df %>% select(wxGF60, wxGA60), centers = 4)

cluster_df$cluster <- factor(clusters$cluster)

p7 <- ggplot(cluster_df, aes(x = wxGA60, y = wxGF60, color = cluster, label = player)) +
  geom_point(size = 2, alpha = 0.85) +
  geom_text_repel(
    data = cluster_df %>% top_n(12, wxGF60 - wxGA60),
    size = 3.4,
    fontface = "bold"
  ) +
  geom_vline(xintercept = median(cluster_df$wxGA60), linetype = "dashed", color = "grey60") +
  geom_hline(yintercept = median(cluster_df$wxGF60), linetype = "dashed", color = "grey60") +
  
  scale_color_viridis_d(option = "plasma") +
  
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "top",
    panel.grid.minor = element_blank()
  ) +
  labs(
    title = "Clustering based on Off & Def Impact",
    x = "wxGA60",
    y = "wxGF60",
    color = "Playertype:",
  )

p7