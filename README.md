# comment-sentence-transformer

## TL;DR 
Using some NLP for classifying sentences in google reviews.

## Which tasks
With one task, the structure could be simpler. The forward pass wouldn't need to differentiate by task.
But having both tasks makes the model a bit more useful, and it works out well that the two tasks care share the representations

## What is frozen
This uses distilled BERT which provides a very helpful starting place/ helpful.
Freezing the transformer and only adjusting weights for the head didn't give enough opportunity to learn the tasks.
I did try that though, to compare, and with just the one layer that I added, it didn't perform well. 
Unfreezing the transformer as well allowed the last layers to tailer the embeddings meanings to the specific tasks.

## Transfer learning considerations
Using a BERT based model gets the model most of the way there. 
I unfroze all the layers, but it probably would be better to only unfreeze the top couple of layers only given the small dataset.
Overall using the existing weights in BERT as a starting piont and fine tuning on my own data gave good results.



