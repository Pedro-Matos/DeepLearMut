 BioCreative V.5. BeCalm (Biomedical annotation metaserver) task  (Version 1.0 September 2016)
-------------------------------------------------------------------

This directory contains the training abstracts and manual annotations for the tasks CEMP and GPRO.

1) BioCreative V.5 training set.txt : Training set abstracts

This file contains plain-text, UTF8-encoded patent abstracts (titles and abstracts in English) from
patents published between 2005 and 2014 that had been assigned to the IPC codes
A61P and A61K31. 

1- Article identifier (Patent identifier)
2- Title of the article
3- Abstract of the article

In total 21000 abstracts are provided in this training set.
Note that the test set will be provided in the same format. 


2) Training data annotations: 

CEMP_BioCreative V.5 training set.txt
GPRO_BioCreative V.5 training set.txt

This file contains manually generated annotations of chemical entities of the training dataset.

It consists of tab-separated fields containing:

1- Article identifier (Patent identifier)
2- Type of text from which the annotation was derived (T: Title, A: Abstract)
3- Start offset
4- End offset
5- Text string of the entity mention
6- Type of chemical entity mention (You can view all types rquired in the tasks at the BeCalm task view section)

Additional comments
-------------------
Use the Becalm platform to evaluate the performance of your system. You have some prediction examples (in different formats) in "Your predictions files repository" section


