# String Reversal Experiment

Results did not appear correct on first and second tries. Increasing the amount of tokens in sample.py still did not produce any tangible results. May have to work on adding more training data or reconfiguring the problem configuration or set.

# Echo Experiment

Changed the experiment to have the model repeat a word identically. It still does not produce good results, so I am going to switch the data again.

# Alphabet Experiment

Finally, I am having the model just repeat the alphabet over and over again. The loss reduced to 0 on my CPU, so compared to the others, which remained consistently at around 3-4, it performed significantly better. In my sampling, it produces perfect results.

![Train/Loss Graph](https://raw.githubusercontent.com/nishcola/comp560-nishkolla/refs/heads/main/media/W%26B%20Chart%201_27_2026%2C%2010_29_19%20PM.png)
