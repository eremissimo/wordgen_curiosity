## Curiosity Reward or Entropy Penalty?
This small project was inspired by the article 
[Can Language Models Generate All Molecules with Given Properties?](https://storage.googleapis.com/public.storage.yerevann.com/drafts/molecular-generation.pdf). I've been curious 
if the usage of some fancy curiosity models from RL could be beneficial for
generation of unseen valid sequences by sampling from the trained model. 
In my opinion it should encourage to produce more diverse outputs. And it is,
but here I want to compare it with a much simpler approach: the entropy penalty.

Instead of SMILES generation I consider a much simpler task of char-by-char 
word generation. The target property is being a transitive verb.

## TLDR: Entropy Penalty
In the small-scale task of short sequences generation there is no measurable 
difference between the curiosity intrinsic reward and the entropy penalty in 
the loss. Both trained models performed at the pretty much the same level.

So the entropy penalty is clearly the winner, at least in this specific task
considered, because it's way much simpler not only from conceptual but also from a computational 
point of view.

## Training setups
The whole dictionary of words is split up in three sets
* word (all words except transitive verbs, ~102000 words)
* training word (training set of transitive verbs, ~3000 words)
* test word (test set of transitive verbs, ~6000 words)  

For the sake of simplicity and to prevent any data leakage to from test set,  
we take the words belongs to only one part of speech. If the word can be regarded 
as a transitive verb then it's treated as a transitive verb only. For the rest
words duplicates are deleted as their p.o.s is not relevant.

The pretraining phase is performed over the union of 'word' and 
'training word' sets.

Then comes the RL-tuning phase. The models are trained to minimize 
a policy gradient loss 
```math
L(\theta) = - \frac 1 K \sum_{k} (Q(s_{1..k}) - \bar Q(\bar s_{1..k})) \log p(s_k | s_{1..k-1}, \theta)
```
where $s_i$ are generated token sequences, $\theta$ are the model weights, 
$Q$ and $\bar Q$ are cumulative 
rewards-to-go for a generated sequences in sampling and argmax mode (as a stable 
deterministic baseline).

For the word generation reward of +1 is given for the current token $s_i$ if the 
prefix-so-far $s_1,...s_i$ is the prefix of some transitive verb 
from the training set, reward -1 is given if the prefix is not represented by 
any word at all. In all other cases the reward is 0. Also, a small reward of +0.5
is added at the end token if a generated word is a full word presented in the dictionary.

Then we have two different setups for output diversity encouraging
1. Entropy Penalty.  
In this setup the additional entropy term $ - \beta H(p(.|s_{1..k-1}, \theta))$
is added to the loss
2. Curiosity Reward.  
In this setup additional curiosity reward is added to the word generation reward, 
a number from the interval $[0, 1]$. If the sequence $s_1,...s_i$ (regardless of 
the $s_1,...s_{i-1}$ status) is pretty new to the curiosity model, then it 
should be surprised by it and give a reward close to 1 to the token %s_i%.
Otherwise, if the sequence is generated too often then the curiosity model is 
bored and gives a reward close to 0. 

After the pretraining and rl-finetuning phases the model is evaluated
on the amount of unseen words from the test set it can produce by sampling.
Overall, both approaches show pretty much the same results of ~1500 out of 6000 
unseen test words found from 200k samples.

## Short Memory Curiosity
For the model of curiosity I use a distillation based approach i.e. when we
have two models, a fixed teacher model and a learnable student model. 
When an input sequence (state) is fed into the curiosity model the discrepancy 
between the models' output is computed as a measure of surprise. Also, a 
single backpropagation step for the student model is performed to match the
teacher's output. Hence the boredom mechanism: the more often the curiosity
'sees' an example sequence, the more close the student's output would be to 
the teacher's.

This approach have a property that is not quite suitable for our task: it has 
infinite memory across the training process. It is equally (well, not quite, 
but almost) bored by states at the beginning of the training and more recent
states. 

To introduce a forgetting mechanism in curiosity the teacher model is not 
fixed but allowed to slowly drift towards another fixed 'mastermind' model.

So overall the curiosity forward method looks somewhat like this:
```python
def forward(self, state: torch.Tensor) -> torch.Tensor:
    prob_m = self.mastermind(state).softmax(-1)  # frozen
    out_t = self.teacher(state)               # trainable
    out_s = self.student(state)               # trainable
    prob_t = out_t.detach().softmax(-1)
    prob_s = out_s.detach().softmax(-1)
    surprize = surprize_metric(prob_s, prob_t)
    # student learns from teacher
    loss_s = torch.nn.functional.cross_entropy(out_s, prob_t)
    loss_s.backward()
    self.student_opt.step()
    self.student_opt.zero_grad()
    # teacher learns from mastermind (hence the model drift)
    loss_t = torch.nn.functional.cross_entropy(out_t, prob_m)
    loss_t.backward()
    self.teacher_opt.step()
    self.teacher_opt.zero_grad()
    return surprize
```