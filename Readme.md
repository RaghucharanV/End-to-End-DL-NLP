**CUSTOM AI MODEL DEPLOYMENT PIPELINE**
`--------------------------------------------------------------------------------`
It is pipeline that Extract data from local files predcits a deep learning model using NLP techniques.Load the data in Postgresql db table.Also bulid a frontend interface to interact with data. Future it contained using Docker and Build a cluster using minikube, inserted in kubernetes
`--------------------------------------------------------------------------------`
***Requirements**🚀🚀🚀🚀
- Install in conda env or python env these dependencies
  - python
  - Tensorflow
  - NLTK
  - Numpy,pandas,matplotlib,seaborn,scikit-Learn
  - minikube
  - Docker desktop install  
BREIF:
**1. AI model Development**
    Here, Developed a simple Deep Learning model that predict Emotions of labels are ['joy', 'fear', 'anger', 'sadness', 'neutral'] in the given feedback text.
    It import train, test csv files with cols are text and its labels.Which undergoes to EDA and Feature Engineering. Perform NLP techniques cleaning text, labeling, tokenization, padding, word embedding,etc. After all it split data in train set and test set, next model with  LSTM layer and dense with label and softmax,predict model with sample text.
    Here the code,[Emotion-Analysis notebook](nlp.ipynb)

    Specifically to save model using tensorflow.And also it is important to save its tokens of each text with its corresponding embedding to get accurate result after model save and use future.

    ```python
    #save tokens
    #it output a class and config 
    tokenizer_config = tokenizer.get_config()
    #save tokens in json
    tokenizer_json = tokenizer.to_json()
    with open('saved_tokenizer_config.json', 'w', encoding='utf-8') as f:
        f.write(tokenizer_json)
    #when reading again in use....
    import json
    with open('saved_tokenizer_config.json', 'r', encoding='utf-8') as f:
        tokenizer_config = json.load(f)
    # in config key a word-index sub dict is present need to copy to
    custom_tokens = tokenizer_config['config']['word_index']
    dictionary = json.loads(custom_tokens)#convert to dict

    #to save model
    model.save("s_model")
    #to load model
    model1 = tf.keras.models.load_model('s_model')
    ```
**2.Web Service**
    To build web service a frontend, Used flask lib from python,Flask is a micro web framework written in Python. It is designed to be lightweight, modular, and easy to use, allowing developers to quickly build web applications with minimal boilerplate code.

    here, the code of [webapp](app.py)
    imported flask lib with app.route('/') return the predcition of model with logging in database.
    For app.py contain code 
    you can check by runing on cmd....
    ```bash
    python app.py 
    ```

    ```py
    def predict_text(input_text):
    res = predict_text_and_log(input_text)
    return res

    @app.route('/', methods=['GET', 'POST'])
    def index():
        result = None
        if request.method == 'POST':
            # Get the input text from the form
            input_text = request.form['input_text']

            #result of prediction
            prediction = predict_text(input_text)

            # Prepare the result to be displayed on the same page
            result = {'input_text': input_text, 'prediction': prediction}

        # Render the home template with the result
        return render_template('index.html', result=result)
    ```
    for [load.py](load.py) which support the app.py file to load model and predict
    text and log to database

    ```py
    # main fun to work with model and pred.
    def model_pred(text):
        #to predict model
        text = clean_text(text)
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=len(word_indexs) + 1, oov_token='<OOV>')
        tokenizer.word_index = word_indexs
        seq = tokenizer.texts_to_sequences(text)
        flattened_list = [item for sublist in seq if sublist and sublist[0] is not None for item in sublist]
        flattened_list = [flattened_list]
        pad_seq = pad_sequences(flattened_list,maxlen=500)
        pred = model.predict(pad_seq)
        class_names = ['joy', 'fear', 'anger', 'sadness', 'neutral']
        return class_names[np.argmax(pred)]
    
    #to load to Database a fun
    def predict_text_and_log(input_text):
        # Make prediction
        prediction = model_pred(input_text)

        # Log the prediction request and result in the database
        insert_query = sql.SQL("INSERT INTO prediction_logs (request_id, input_data, prediction_result) VALUES (uuid_generate_v4(), %s, %s);")
        db_cursor.execute(insert_query, (input_text, prediction))
        db_connection.commit()
        return prediction

    ```
**3.Containaize with docker**
    About Docker:
    Docker is a platform and tool designed to make it easier to create, deploy, and run applications by using containers. Containers allow a developer to package up an application with all parts it needs, such as libraries and other dependencies, and ship it all out as one package.
    [Dockerfile]:Dockerfile

    To build the Image:

    ```bash
        docker build -t nlp-app .
    ```
    To run image:

    ```bash
        docker run -p 8080:8080 nlp-app
    ```
    Docker hub image

    ```bash
        docker tag nlp-app username/nlpapp1
    ```
    Docker login and push with

    ```bash
        docker login

        docker push username/nlpapp1
    ```
**4.Kubernetes**
    About Kubernetes:

    Kubernetes, often abbreviated as K8s, is an open-source container orchestration platform for automating the deployment, scaling, and management of containerized applications. It was originally developed by Google and is now maintained by the Cloud Native Computing Foundation (CNCF). Kubernetes provides a robust framework for running distributed systems and microservices efficiently.

    About minikube:
    Minikube is a lightweight, local Kubernetes distribution designed for development and testing purposes. It enables developers to run a single-node Kubernetes cluster on their local machine, providing an easy way to experiment with Kubernetes without the need for a full-scale cluster. 

    First install [minikube](https://minikube.sigs.k8s.io/docs/start/)
    add path to Environmental variables

    in powershell open with run as admin

    ```bash
        minikube start

    ```
    check status it running

    ```bash
        minikube status
    ```
    Now build a yaml file for deploy into kubernetes cluster
[yaml][yaml]
    commands to exicute yaml file deploy in cluster using minikube

    ```bash

        docker build -t nlp-app .
    ```
    ```bash
        kubectl apply -f k8.yaml
    ```
    check depolyment

    ```bash

        kubectl get pos,svc
    ```
    run cluster

    ```bash
        minikube service nlp-app-svc
    ```

its end untill.............👋 (Waving Hand)🎉 🌟 🚀🙌 🤗
    




[yaml]: k8.yaml