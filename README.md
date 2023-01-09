# NER

This project is part of [dylog.ai](https://www.dylog.ai/). It helps to identify different entities that would help better elastic search algroithm.

## Files
```
|- train.py : Is used to for training the model
|- test.py : Is used to test the model
|- config.py : Is used to configure the model
|- utils
    |- dataloader.py : Is used to preprocess the data and loading it
    |- datautils.py : save and load pickel files
|- save
    |- model 
        |- config.json : Is used to save the model configuration
    |- data_roberta-base.pkl : Is used to save the data (will be created after data processing)
    |- roberta-base_best.pth : Best model (will be saved after training)
|- batch_run.sh : Is used to run the model in batch mode (Modify it to your needs)
```

## License
For open source projects, say how it is licensed.