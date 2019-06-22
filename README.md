# stat-learn-2019
## Ideas proposals 
All topics are to be developed with clear and concise notes . Then we will have to choose one to work on .
Delay is on March 28

### Ideas based on motion
 Build motion classification model - Standing, Sitting, going down-stairs, going up stairs, walking, running



### Project structure 
<pre>
├── data
│   ├── experiments_cleaned_dataset
│   ├── expriments_rawdata
│   ├── hapt_dataset
│   ├── output_data
│   └── project_dataset
│       ├── dataset.csv
├── dataset.py
├── constants.py
├── __main__.py
├── models
│   ├── NN_Model.py
│   └── SVM_Model.py
├── README.md

</pre>

The data folder contains all the datas 
Our project dataset is located in #project_dataset 
The experiments data from Science journal app are put in experiments_rawdata folder 
the HAPT data must be unzipped an renamed as hapt_dataset

The constants.py module conatins all the constant uses in the projects 
the dataset.py module processes dataset 
the __main__.py run the project 


The models are stored in model folder 
SVM_Model.py defines an svm model 
NN_model.py define and neural network model 











