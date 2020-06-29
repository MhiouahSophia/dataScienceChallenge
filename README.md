## Engie_dataScienceChallenge

https://datascience-challenge.engie.com/#/challenge/128738

Ready to run the code   

0. Clone the code into a folder, name it as you want. In the same folder create 2 folders named Engie_Result (result of the prediction will be exported in ) and FastText (extract the word embedding information)

1. In the folder Engie_result create 2 other folders: named ML and  DL 

2. In the FastText folder download the word embeding named : cc.fr.300.bin   
```
$ ./download_model.py fr

```

3.Create your env, name it as you want and activate it. Install all the librairies required to run the project.
```
$ source yourenv/bin/activate
$ pip install -r requirement.txt

```
4.Change the data_path in the config ini with your own path    
5. Launch the code    
 ```
$ python train-test.py 

```
 
                        
