# Users Accumualted Tweets Classification
This project aims to solve long text limitation of input token(512 token). Our task was to classify each user into his/her personality MBTI types based on tweets that he/she has posted so far. So we collected user tweets and on average each user have above 2000 tweets and each tweet can take up 100 tokens on average. so 200000 is much more than 512 that is BERT max length sequence.  

## Limitation
At first I wanted to use two consecutive BERT that both fine-tunes together. The idea was to feed each user as a batch so we have all users tweets inserted as batch into the first BERT and then this BERT gives us an embedding for each tweet. Now we can feed this embedding to the second BERT as a batch with one element. we Concatenate each embedding to form at most sequence with length 512. At the end we each some embedding with following shape (1, 512, 768). then we extract second BERT CLS layer and pass it through FC layer and then sigmoid for binary-classification. 
<br />
So the idea was great but lack of enough GPU didn't let me to fine tune both model at once, eventually I had to freeze first model.

## Impelementation
this project has been Implemented in two section. Since I ran into GPU shortage on any device, I decided to use google colab free GPU(15 GB T4). 

### section I
Since the first model was frozen, I had to in some way teach how the user's tone were in their tweets and the only way that I came up into was using BERT Masked Language Modeling. so I fine tuned BERT on users tweets to learn about how was users way of tweeting. 

### section II
after that I have fine-tuned BERT model on my dataset, I could then start training my classifier BERT which the explanation is mentioned above. through this way, the most amount of GPU usage was at most 6 GB which was great.
