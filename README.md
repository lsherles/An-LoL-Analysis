
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

To explore Ashe's effect on how teams perform, we start by filtering the original dataframe for when '_champion_' is equal to Ashe, and then simply find the mean of the '_result_' column. To find the win rate for teams who banned, we just need to filter the previous dataframe that we melted to contain games where Ashe was banned, and then take the mean of '_result_'. We can then combine these dataframes to create one larger dataframed, '_gameid_' and '_teamname_', that contains a new column, '_status_', dictating whether Ashe was picked or banned by the specific team in the game. Games where Ashe was picked will, as a result, have a value of NaN under the '_ban slot_' column.  

```py
print(ashe_df.head(5).to_markdown(index=(False)))
```

|gameid	            |teamname	            |league	|result	|champion	|status	|ban_slot|
|:------------------|-----------------------|-------|-------|-----------|-------|-------:|
|10665-10665_game_1	|Bilibili Gaming	    |LPL	|1	    |Ashe	    |Picked	|NaN     |
|10665-10665_game_2	|Bilibili Gaming	    |LPL	|0	    |Ashe	    |Banned	|ban2    |
|10665-10665_game_3	|Top Esports	        |LPL	|0	    |Ashe	    |Banned	|ban2    |
|10666-10666_game_1	|Royal Never Give Up	|LPL	|0	    |Ashe	    |Banned	|ban3    |
|10666-10666_game_2	|Royal Never Give Up	|LPL	|1	    |Ashe	    |Banned	|ban1    |


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

### Hypothesis Testing

We will use hypothesis testing to see if the difference in win rates between teams who ban Ashe versus teams who pick Ashe is significant. Based on the previous graph comparing win rates for when Ashe is picked or banned, we can outline our test is outlined as follows:
- Null: The win rate of teams that pick Ashe is the same as the win rate of teams that ban Ashe.
- Alternative: The win rate of teams that pick Ashe is higher than the win rate of teams that ban Ashe.
- Significance Level: 0.05 or 5%

To test whether the difference is singificant, we can use z-score to measure the standardized difference between the two. If our z-score is large, then we find that picking Ashe leads to a win a signficantly higher amount than banning Ashe. After calculating the total win rate and standard error, we can take the differences in individual win rates and divide by the SE to find the z-score. We can further use _norm.cdf_ to calculate the p-value as well.

```py
# Compute Z-score
z_score = (ashe_pick_wr - ashe_ban_wr) / SE

# Compute one-tailed p-value (right-tailed test)
p_value = 1 - norm.cdf(z_score)

print(f"Z-score: {z_score:.4f}")
print(f"P-value: {p_value:.4f}")
```
Z-score: 1.9584
P-value: 0.0251






---

### Framing a Prediction Problem


---
### Baseline Model


---

### Final Model


---

### Fairness Analysis


---