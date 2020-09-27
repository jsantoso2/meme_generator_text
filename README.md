# Meme-Generator-Text-Project

- <strong>Used CNN/LSTM Model (in Pytorch) to generate meme text on 100 most popular memes from Imgflip.com</strong>
- <strong>Created front end application using React</strong>
- <strong>Created endpoint API using Flask</strong>
- <strong>Deployment of Flask app + React app on Heroku</strong>

### App link: https://meme-generator-nn-app.herokuapp.com/
**WARNING! Please close the app after use as it reduces strain on quotas!**

### Application Screenshots
<table>
  <tr>
    <td>Home Page</td>
    <td>Prediction Page</td>
  </tr>
  <tr>
    <td valign="top"><img src="https://github.com/jsantoso2/meme_generator_text/blob/master/Screenshots/home.JPG" height="300"></td>
    <td valign="top"><img src="https://github.com/jsantoso2/meme_generator_text/blob/master/Screenshots/prediction.JPG" height="300"></td>
  </tr>
</table>

<table>
  <tr>
    <td>Sample Memes Page</td>
  </tr>
  <tr>
    <td valign="top"><img src="https://github.com/jsantoso2/meme_generator_text/blob/master/Screenshots/sample_memes.JPG" height="300"></td>
  </tr>
</table>

### Tools/Framework Used
**Model**
- CNN Model, LSTM Model
- Deep Learning Framework: Pytorch (https://pytorch.org/)
- GPU: Google Colab 

**Front End application**
- Front end: ReactJS (https://reactjs.org/)
- Endpoint API: Flask (https://flask.palletsprojects.com/en/1.1.x/)
- Flask App deployment: Heroku (https://www.heroku.com/)
- React App deployment: Heroku (https://www.heroku.com/)

### Dataset
**Original Dataset**
- ~575k Memes + Captions (Retrieved ~ Apr 2020) (https://github.com/schesa/ImgFlip575K_Dataset)
- 100 most popular memes on Imgflip.com (https://imgflip.com/popular_meme_ids)

**Dataset used for model training (subset of the original data as training on all data is too expensive)**
- Memes with <200 characters, English memes, Memes with expected number of <sep> token (~310k memes)
- Subsample ~1k memes from each meme image => Total: ~88k memes (some classes have <1k examples)
- Training + Validation Set
  - CNN (Model1)
      | Input |	Label | Meme Image | 
      | :-------: | :--: | :-------------------: | 
      | "[start]"	       | a	    | 10-Guy (0 in img2idx) |
      | "[start]a"       | p	    | 10-Guy (0 in img2idx) |
      | "[start]ap"      | p	    | 10-Guy (0 in img2idx) |
      | "[start]app"     | l	    | 10-Guy (0 in img2idx) |
      | "[start]appl"    | e	    | 10-Guy (0 in img2idx) |
      | "[start]apple"   | space  | 10-Guy (0 in img2idx) |
      | "[start]apple "	 | s	    | 10-Guy (0 in img2idx) |
      | "[start]apple s" | h	    | 10-Guy (0 in img2idx) |
      | And continue...  | ...    | ...                   |                                                                                       

    - Example Meme Caption: [start]apple should make a big screen tv [sep] and call it the big mac[end]
    - Pre-padding the input to have same length (128 characters)
      - Example: [[PAD], [PAD], ..... , [start]] (for first example)
    - 95% Training Example, 5% Validation Example
    - ~5M Training Examples, ~260k Validaion Examples                                                                                   
  - LSTM (Model2)                                                                                     
      | Input |	Label | Meme Image | 
      | :-------: | :---------------: | :-------------------: | 
      | "[start] apple should make a big screen ... big mac" | "apple should make a big screen ... big mac[end]" | 10-Guy (0 in img2idx) |
      | Next Meme ...                                        | ...                                               | ...                   |
      
    - Example Meme Caption: [start]apple should make a big screen tv [sep] and call it the big mac[end]
    - Post-padding the input + label to have same length (199 characters)
    - One Meme => One training example
    - Input and Label array should have same length, and [end] token is never included in Input array
    - 95% Training Example, 5% Validation Example
    - ~83k Training Examples, ~4.5k Validaion Examples                                                                                   

### Procedure
- Model
  - Preprocessing
    - Clean, filter, and sample dataset
    - Lowercase all text (Limit to only unique 72 characters)
    - Create Char2Idx and Img2Idx Mapping
    - Create Input, Labels, ane Meme Image input (as described in previous section)
    - Map Characters, and Images to Indexes
  - CNN Model
    - Use Dataloader (batch_size = 256)
    - Input: (img_num, input_sequence)
      - img_num = Image # from Img2idx dictionary mapping (batch_size, 1)
      - input_sequence = Input sequence for next character prediction (batch_size, 128) => where 128 is predefined sequence_length in previous section (Pre-padded)
    - Output: Logits for next character predictions 
        - Apply Softmax to get prediction probabilities 
        - Size: (batch_size, 72) => where 72 is the # of character classes
    - Loss: Cros Entropy Loss
    - Metrics: Accuracy (% of correct predictions)
    
  - LSTM Model
    - Use Dataloader (batch_size = 32)
    - Input: (img_num, input_sequence, labels, prev_hidden_state_h, prev_hidden_state_c)
      - img_num = Image # from Img2idx dictionary mapping (batch_size, 1)
      - input_sequence = Input sequence for next character prediction (batch_size, 199) => where 199 is predefined sequence_length in previous section (Post-padded)
      - labels = Labels sequence for current input (batch_size, 199) => where 199 is predefined sequence_length in previous section (Post-padded)
      - prev_hidden_state_h, prev_hidden_state_c = previous hidden states of LSTM (initialized to 0)
    - Output: Logits for next character predictions 
        - Apply Softmax to get prediction probabilities 
        - Size: (batch_size, 199, 72) => where 199 is sequence length, 72 is the # of character classes
    - Loss: Cros Entropy Loss
    - Metrics: Accuracy (% of correct predictions)

  - Modelling
    - train for 8-9 hours on Google Colab GPU
    - Pick best model (based on metrics) and do prediction

- Application
  - React
    - Create the ReactJS files
    - Build and create build folder
    - Use simple Express server to serve ReactJS application
    - Deployment to Heroku
  - Flask
    - Create all the Endpoint API
    - Integrate Pytorch model with the Endpoint APIs
    - Integrate Selenium + Google Chrome webdriver to be used on Heroku (to display meme text on images)
    - Deployment to Heroku

### Prediction Algorithm
- Use [start] token first + any other characters supplied
- Feed into Network and obtain next character output -> Feed into network again until <end> is generated
- Methodology:
  - Greedy
    - At each time step pick output with highest probability
  - Sampling
    - At each time step sample randomly based on the probability
  - Beam Search (k = beam width in this case = 3)
    - At each time step take k best output
    - Algorithm (modified slightly):
        - First generate k predictions with scores
        - Take each prediction and generate k more predictions at each time step from the previous output until we reach k end candidates. Score will be current score P(Y|X) * prev score P(X) = P(X,Y)
        - Iterate until we reach k end candidates
        - Take output (with end token) that has highest score.
  
### Results
- **CNN**
  - Model Architecture
    <p align="center"> <img src=https://github.com/jsantoso2/meme_generator_text/blob/master/Screenshots/CNN.JPG height="500"></p>
  - Results
    <p float="left">
        <img src=https://github.com/jsantoso2/meme_generator_text/blob/master/Screenshots/cnn_val.png height="200" width="450">
        <img src=https://github.com/jsantoso2/meme_generator_text/blob/master/Screenshots/cnn_train.png height="250" width="450">
    </p>
  - Summary of results
    <p align="center"> <img src=https://github.com/jsantoso2/meme_generator_text/blob/master/Screenshots/cnn_result.JPG height="200"></p>
  - **Best Output:** 40M Step => 63.61% Val Accuracy
  - Optimal Parameters
    - Batch_size = 256
    - Sequence_length = 128
    - Learning Rate = 0.001

- **LSTM**
  - Model Architecture
    <p align="center"> <img src=https://github.com/jsantoso2/meme_generator_text/blob/master/Screenshots/lstm.JPG width= "500" height="400"></p>
  - Results
    <p float="left">
        <img src=https://github.com/jsantoso2/meme_generator_text/blob/master/Screenshots/lstm_val.png height="200" width="450">
        <img src=https://github.com/jsantoso2/meme_generator_text/blob/master/Screenshots/lstm_loss.png height="200" width="450">
    </p>
  - Summary of results
    <p align="center"> <img src=https://github.com/jsantoso2/meme_generator_text/blob/master/Screenshots/lstm_result.JPG height="100"></p>
  - **Best Output:** 270k Step => 63.35% Val Accuracy
  - Optimal Parameters
    - Batch_size = 32
    - Sequence_length = 199
    - LSTM Layer = 1
    - LSTM Hidden Size = 1024
    - Learning Rate = 0.001

#### Selected Examples CNN
- Example1
  - Source: "[start]exec"
  - Label: u
  - Prediction: u
  - Img Name: Trump-Bill-Signing
- Example2
  - Source: "[start]what if the bus is going to a place called[sep]\"not in se"
  - Label: r
  - Prediction: a
  - Img Name: Conspiracy-Keanu
- Example3
  -	Source: "[start]when you commanded the ensign to scrub the poop deck[sep]not r"
  -	Label: e
  -	Prediction: e
  -	Img Name: Captain-Picard-Facepalm


#### Selected Examples LSTM
- Example1
  - Source: [start]me[sep]stay inside like a good citizen during coronavirus outbreak [PAD][PAD] …
  - Label: me[SEP]stay inside like a good citizen during coronavirus outbreak[end][PAD][PAD] … 
  - Prediction: me[SEP]mtoy hnside[END]tike a bood moryzenstering thlonavirus[end]mnt[end] …
  - Img Name: UNO-Draw-25-Cards

- Example2
  - Source: [start]people who think that video games cause violence have only seen people play volent video games.[PAD][PAD] …
  - Label: people who think that video games cause violence have only seen people play volent video games.[end][PAD][PAD] … 
  - Prediction: teople who shink thet iideo games aanse iiolence iase aney thcn teople tlay fitlntioideo games[end][end] …
  - Img Name: Change-My-Mind

#### Conclusion
CNN performed significantly better than LSTM in this case. LSTM model in this case does not really generate quite coherent english text.

#### Future Work
- Expand Image Embeddings to using InceptionV3 to generate custom memes from custom images
- Expand using parental filtering (no swear words, etc.) + spell correction
- Training on larger data

### References:
**Dataset References**
- https://imgflip.com/ai-meme
- https://github.com/schesa/ImgFlip575K_Dataset
- https://imgflip.com/popular_meme_ids

**Modelling References**
-	https://towardsdatascience.com/meme-text-generation-with-a-deep-convolutional-network-in-keras-tensorflow-a57c6f218e85
-	https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html 
-	https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1184/reports/6909159.pdf 
- http://karpathy.github.io/2015/05/21/rnn-effectiveness/

**Application References** 
- Deployment of Flask to Heroku with Selenium: https://www.youtube.com/watch?v=Ven-pqwk3ec  

### Final Notes:
- To see more technical details, please see notes.docx for all my detailed notes
