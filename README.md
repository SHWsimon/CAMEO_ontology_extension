# CAMEO_ontology_extension

1. pretrain.py
	><ul>
	><li>Read codedSentences.txt</li>
	><li>Generate Vectorizer from sentences read(Bag of word)</li>
	><li>Separate raw data into training set and testing set</li>
	><li>Tokenize the training set and test set with Vectorizer respectively</li>
	><li>Train model by Logistic Regression</li>
	><li>Analysis the result model with testing data set</li>
	
	Output result:
	><ul>
	><li>Model</li>
	><li>Vectonizer</li>
 	></ul>

2. predict.py : Spectral clustering
	><ul>
	><li>Read notCodedStences.txt</li>
	><li>Predict label by pretrained model</li>
	><li>Output the predict labels</li>
	></ul>

3. project.py
	><ul>
	><li>Read notCodedStences.txt</li>
	><li>Predict label by pretrained model</li>
	><li>Applied five different extraction method</li>
		<ul>
		<li>POS tagging</li>
		<li>Most frequent n collocations</li>
		<li>Most frequent n noun phrases</li>
		<li>RAKE (rapid automatic keyword extraction)</li>
		<li>TF-IDF</li>
		</ul>
	</ul>
	
Output the predict labels and new patterns
Ex:
![image](https://github.com/SHWsimon/CAMEO_ontology_extension/blob/master/pic/Picture1.png) 
