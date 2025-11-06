<pre style="white-space: pre-wrap; word-wrap: break-word; font-size: 2vw;">
 █████╗ ██╗   ██╗████████╗ ██████╗  ██████╗ ██████╗ ███╗   ███╗██████╗ ██╗     ███████╗████████╗███████╗    ███╗   ███╗███████╗
██╔══██╗██║   ██║╚══██╔══╝██╔═══██╗██╔════╝██╔═══██╗████╗ ████║██╔══██╗██║     ██╔════╝╚══██╔══╝██╔════╝    ████╗ ████║██╔════╝
███████║██║   ██║   ██║   ██║   ██║██║     ██║   ██║██╔████╔██║██████╔╝██║     █████╗     ██║   █████╗      ██╔████╔██║█████╗  
██╔══██║██║   ██║   ██║   ██║   ██║██║     ██║   ██║██║╚██╔╝██║██╔═══╝ ██║     ██╔══╝     ██║   ██╔══╝      ██║╚██╔╝██║██╔══╝  
██║  ██║╚██████╔╝   ██║   ╚██████╔╝╚██████╗╚██████╔╝██║ ╚═╝ ██║██║     ███████╗███████╗   ██║   ███████╗    ██║ ╚═╝ ██║███████╗
╚═╝  ╚═╝ ╚═════╝    ╚═╝    ╚═════╝  ╚═════╝ ╚═════╝ ╚═╝     ╚═╝╚═╝     ╚══════╝╚══════╝   ╚═╝   ╚══════╝    ╚═╝     ╚═╝╚══════╝                                                                                </pre>  


## ***Autocomplete me*** is a project I have built to try to replicate Google/Apple's Autocomplete feature in their systems 

There are 2 models in this project serving the same functionality (prediction of charecters that comes next according to the previous charecters and words)

***GRU MODEL***

The first model is made of a GRU architecture it works heres a demo of the GRU model
"hi mom how ar(ound to organizing a)..." 
![GRU EXAMPLE](./assets/gruexample.png)
This sentence shows a lesser accuracy to how we actually speak although each words it forms kind of make sense the words together dosent make too much sense
The way it arranges each charecters make a good enough sequence model to form cohorent words but not a cohorent sentence!!! 

***LSTM MODEL***

The next model i.e. made of of the lstm architechture shows considerable improvement than the previous one it correctly guesses and forms a coherent sentence 
"hi mom how ar(e you did you see th)..." 
![GRU EXAMPLE](./assets/lstmexample.png)
The LSTM model is making a much better prediction of sequence to form cohorent sentences...
It seems like the model has now a better memory to guess the sequence better it seems like its asking mom how she is and if she saw something?... THAT IS A SIGNIFICANT IMPROVEMENT!!!

***Dataset***

I have listed the dataset publicly in this repository too incase anyone wants to add anything to it and advance the model further

***API***

You can find api server that serves the api through FastAPI for both the models in ports **http://127.0.0.1:8000 (GRU) | http://127.0.0.1:8001 (LSTM)** respectively

***Usage***

I have also made a cool frontend linked as the index.html file in this repository I would like you to use the model yourself in there


**steps**

*To run the models locally*
1. install the requirements file 
```python
pip install -r requirements.txt
```
2. run the server file called api_server.py
3. run the index.html file frontend
   *for switching models change the model in the topbar area*

## License

This project is licensed under the [MIT License](LICENSE).

You are free to use, modify, and distribute this project for any purpose, provided that proper credit is given to the original author — **Sagnik Roy (github.com/@Sagnikkroy)**


