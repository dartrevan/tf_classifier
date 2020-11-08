### Info
This repository contains code for text classification with simple one layer bidirectional GRU <br> 
model

#### Requirements installation
    1. pip install -r requirements
    2. python -c "import nltk; nltk.download('punkt')"
    
#### Usage example
1. Model training <br>
    All the models are trained using configuration files. To train model simply run the command: <br>
            ```
                python train.py --config_file configs/config.yml
            ```
            <br>
    Configuration file must be in YAML format and contain following parameters: <br>
    <ul>
        train_texts - 
    </ul>
    <ul>
        train_target - 
    </ul>
    <ul>
        train_lengths -
    </ul>
    <ul>
        eval_texts -
    </ul>
    <ul>
        eval_target -
    </ul>
    <ul>    
        eval_lengths -
    </ul>
    <ul>
        token_vocab_file -
    </ul>
    <ul>
        label_vocab_file -
    </ul>
    <ul>
        pretrained_embeddings_file - 
    </ul>
    <ul>
        spell_checker_vocab: -
    </ul>
    <ul>
        classes_count: -
    </ul>
    <ul>
        rnn_num_units: -
    </ul>
    <ul>    
        attention_size: -
    </ul>
    <ul>    
        max_steps: -
    </ul>
    <ul>
        save_chkpts_secs: -
    </ul>
    <ul>
        learning_rate: -
    </ul>
    <ul>   
        model_dir -
    </ul>

2. Prediction using trained model
    ```
    python predict --config_file \
                   --path_to_texts /path/to/texts
                   --path_to_lengths /path/to/lengths
                   --path_to_afeatures /path/to/afeatures
    ``` 