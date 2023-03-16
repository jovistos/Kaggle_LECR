**Kaggle: Learning Equality - Curriculum Recommendations**
===============

This is a trainig pipeline that I've used for Kaggle competition called Learning Equality - Curriculum Recommendations.

End-to-end approach to document retrieval was proposed very recently [1]. For this competition I've decided to limit myself to using only this approach and see how it goes. I've scored very low with it (885 place on PL, 0.25 score), but nevertheless I've got it to work. I haven't noticed anyone trying it for this competition so I thought it would be interesting to write a bit about it. This could also be a winning approach on the next similar competition, who knows. (wouldn't surprise me)

Whole inference pipeline is just one T5 [2] generative model where the input is the topic and the output is a list of document identifiers, simple as that :)It's a specific type of multi-task, where one model task is to memorize the identifiers for all of the content documents ( input is the document, output is a document identifier), and other task is, given the topic, predict the list of content document identifiers given in the training data. ( I've also tried the third task, where the topic is the input, and the exact location of that topic in the topic tree is the output, but I've not seen any improvement.) It's a specific type of multi-task in a sense that the model first needs to memorize the content in order for it to do the recommendation. This whole approach is very hyperparameter sensitive and requires a lot of computation, nevertheless it was a very interesting experience. I am still glad I've spent some time with this, regardless of the low score.

Competition overview
https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations/overview

Kaggle notebook with further details
https://www.kaggle.com/code/joviis/end-to-end-approach/notebook

[1] https://arxiv.org/pdf/2206.02743.pdf

[2] https://arxiv.org/abs/1910.10683
