# Pitch Sequences that Lead to Strike Outs

## Intro
The essence of baseball can be described by the duel between the pitcher and the batter. When the batter wins, they get on base, drive in runs, or hit a home run. For the picther, the currency is an out. Some pitchers rely on their defense by giving up contact but still getting outs. However, the most lauded arms get their outs by striking out batters. Pitchers who get strikeouts use a wide breadth of methods, but to keep it simple, this study breaks them down into the following categories: (a) flame-throwers: pitchers who use the fastball to blow their picthes by batters, (b) junk throwers: these arms use less velocity but more use movement to punch hitters out, and (c) crafty hybrids: throwers who utilize both heat and junk to get strike three. 

In case you stumbled on this post with little or knowledge of baseball, I recommend that you read this [post](http://dearsportsfan.com/2014/10/24/different-kinds-pitchers-baseball/#:~:text=%20Pitchers%20classified%20by%20throwing%20motion%20%201,League%20Baseball%20throw%20with%20the%20same...%20More%20) by Ezra Fischer and this [post](https://www.baseball-reference.com/bullpen/Pitches) by baseball-reference to get a basic understanding of picthing at pitch types so that the data makes more sense. I would also recommend reading ["How Baseball Works"](https://entertainment.howstuffworks.com/baseball1.htm) by Kevin Bonsor and Joe Martin to get a broader understanding of the fundamentals of baseball. 

For those more techinically inclined, Additionally, I read two publications that helped get a greater technical understanding of pitch sequence. Both are written about the Japanese baseball league, Nippon Professional Baseball. The first [paper](https://poseidon01.ssrn.com/delivery.php?ID=616064002106074070126030116028120121037016025093044007014074021026002102074069114120028062030124045033010027077093090098122115046083078061083097069027065094116029094010018046069065068119097115065116098088127103069125113120103103098075074068097015123098&EXT=pdf&INDEX=TRUE) written by Yoshiharra and Takahashi (2020) to examines "the relationship between pitch sequences and pitcher/hitter characteristics, game situations, and hitter reslts" with a topic model.  The other [paper](https://pubmed.ncbi.nlm.nih.gov/32182276/) written by Kidokoro and colleagues (2020) studied how the timing of a small set of high school batters was altered with varying pitch sequences and information about the incoming pitch.  

Back to the project, to understand how the sequence of pitches leads to a strikeout, I examined Major League Baseball (MLB) pitch-by-pitch [data](https://www.kaggle.com/pschale/mlb-pitch-data-20152018) that was kindly collected by Paul Schale on Kaggle ([pschale](https://www.kaggle.com/pschale)). 

This repository reflects a portion of a three part series to undestand how pitch sequences affect picther and batter productivity. In this first portion of the series, I explore if data from pitches thrown can classify baseball events. That is to say, the data from the pitches thrown across the MLB are able to classify if an at-bat ends in a defensive out, home run, strikeout, or another outcome within the sport. Furthermore, this portion of the series was conducted for the puporses of the 2021 Kaggle BIPOC Grant program. The next portion of the series focuses on finding the characteristics of sequences of pitches that lead to strikeouts and home runs. Finally, the last portion of the series explores pitchers at the individual level and how they are successful at creating outs through their picthing sequences. 

## Part 1
Classifying Baseball Events

## Part 2
Leveraging the Sequences of Pitches to Understand Strikeouts and Home Runs

## Part 3 
Exploring Power, Crafty, and Hybrid Pitchers at the Indivudual Level
