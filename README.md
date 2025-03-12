
## An LoL Analysis

by Lucas Sherles

---

### Introduction

In competitive League of Legends, the part of the game that is make or break is the draft. This is where each team selects and bans 5 champions, going in a snake order. The goal is to pick champions that are strong and synergize well with each other, while also banning champions that are strong against your own.

This project focuses on the the tier 1 leagues (LCK, LEC, LCS, LPL), which are the highest level of League of Legends anywhere around the world, and explores how the draft phase affects the outcome of each game. 

In the first few parts of this project, we will look at at the most picked and banned champions, and how they affect the win rate of the teams that pick or ban them. Later on, we will look at how the champions picked in the draft can predict which team obtains the first dragon in the game.

---

### Cleaning and EDA

The first step of the cleaning part of this process was to query the dataset for only the top domestic leagues, not including international events. In the data, each entry is assigned to a specific player. Since we aren't interested in player-specific statistics, we can group the data by _gameid_ and _teamname_. This gives us the team statistics for each game, and includes entries for both teams. Specifically, let's figure out which champions were picked or banned the most times. 

Since there are 5 ban slots, each of which have their own column, we need to melt these columns into one column. We can do this by using _groupby_, and since we aren't interested in player-specific statistics, we can group the data by _gameid_ and _teamname_ and keep the relevant columns. Then, we can get the value counts to see which champions were banned the most, and when we combine this with the value counts of the 'champion' column in the original dataset, we can plot the results.

```py
print(pickban_df.head(5).to_markdown(index=False))
```

| champion    |     count    |
|:------------|-------------:|
| Ashe        |       1171.0 |
| K'Sante     |       1100.0 |
| Kalista     |       1096.0 |
| Rumble      |       1094.0 |
| Nautilus    |        967.0 |



<iframe
  src="assets/valcounts.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>
As we can see, the champion that is picked or banned the most is Ashe. Let's explore Ashe's effect on how teams perform by seeing if teams experience more or less success when they pick or ban Ashe.

To explore Ashe's effect on how teams perform, we start by filtering the original dataframe for when '_champion_' is equal to 'Ashe', and then simply find the mean of the '_result_' column. To find the win rate for teams who banned, we just need to filter the previous dataframe that we melted to contain games where Ashe was banned, and then take the mean of '_result_'. 

<iframe
  src="assets/ashe_wr_bc.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>
Interestingly enough, teams that ban Ashe not only lose more often than they win, but they also lose staggeringly more than teams who pick Ashe, who do very well.

---

### Assessment of Missingness


---

### Framing a Prediction Problem


---
### Baseline Model


---

### Final Model


---

### Fairness Analysis


---