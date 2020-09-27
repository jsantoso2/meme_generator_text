import torch
import torch.nn as nn

import json
import numpy as np
import pandas as pd

import os
import time
import urllib.request
from selenium import webdriver
import base64
import io 
from PIL import Image
import re
import random 
import socket
from joblib import Parallel, delayed
import gc

# set default timeout to 10 seconds to prevent getting bad images
socket.setdefaulttimeout(10)

from CNN import MemeGeneratorCNN
from LSTM import MemeGeneratorM2

# set random seed for initialization
np.random.seed(224)

from flask import Flask, jsonify, request, make_response
from flask_cors import CORS


# initialize flask app
app = Flask(__name__)
CORS(app)

# read json for dictionary mapping
# open from json file
with open('char2idx.json', 'r', encoding = 'UTF-8') as json_file:
    char2idx = json.load(json_file)

with open('char2idx_m2.json', 'r', encoding = 'UTF-8') as json_file:
    char2idx2 = json.load(json_file)

# open from json file
with open('img2idx.json', 'r', encoding = 'UTF-8') as json_file:
    img2idx = json.load(json_file)

idx2char = {value:key for key, value in char2idx.items()}
idx2char2 = {value:key for key, value in char2idx2.items()}
idx2img = {value:key for key, value in img2idx.items()}


#################################3 Helper Methods
############ Get available devices
def get_available_devices():
    """Get IDs of all available GPUs.

    Returns:
        device (torch.device): Main device (GPU 0 or CPU).
        gpu_ids (list): List of IDs of all GPUs that are available.
    """
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    return device, gpu_ids

############ Load model from checkpoint
def load_model(model, checkpoint_path, gpu_ids, return_step=True):
    """Load model parameters from disk.

    Args:
        model (torch.nn.DataParallel): Load parameters into this model.
        checkpoint_path (str): Path to checkpoint to load.
        gpu_ids (list): GPU IDs for DataParallel.
        return_step (bool): Also return the step at which checkpoint was saved.

    Returns:
        model (torch.nn.DataParallel): Model loaded from checkpoint.
        step (int): Step at which checkpoint was saved. Only if `return_step`.
    """
    device = "cuda:" + gpu_ids[0] if gpu_ids else 'cpu' 
    ckpt_dict = torch.load(checkpoint_path, map_location=device)

    # Build model, load parameters
    model.load_state_dict(ckpt_dict['model_state'])

    if return_step:
        step = ckpt_dict['step']
        return model, step

    return model

##########3 Setup Model
def setup_model(model_type):
    # load model
    device, gpu_ids = get_available_devices()
    
    if model_type == "CNN":
        # load from state_dict
        model = load_model(MemeGeneratorCNN(), 'CNN_step_40002497.pth.tar', gpu_ids, return_step=False)
    elif model_type == "LSTM":
        # load from state_dict
        model = load_model(MemeGeneratorM2(), 'LSTM_step_270150.pth.tar', gpu_ids, return_step=False)

    model = model.to(device)
    model.eval()

    return model, device


########### Prediction for CNN
def do_prediction_CNN(device, model, img_num, prediction_mode, beam_width = 3, maxcharpred = 250, test_string = ""):
    # device = device to predict on "cpu" or "cuda:0" for GPU
    # model = character model
    # img_num = encoded image number (0-98)
    # prediction_mode = "greedy", "sampling", "beam"
    # beam_width = parameter of beam search
    # maxcharpred = maximum character length of predictions
    # test_string = starter string for meme
    
    ### Method to preprocess input for padding
    def preprocess_input(test_string):
        # convert string to number array
        test_string_proc = [char2idx["<start>"]]
        # add any test string
        for char in test_string:
            test_string_proc.append(char2idx[char])

        # add the prepadding
        currlen = len(test_string_proc)
        rem = 128 - currlen
        for i in range(rem):
            test_string_proc.insert(0, 0)
        return test_string_proc

    # apply method
    # test_string_proc (list of length number of input string)
    test_string_proc = preprocess_input(test_string)

    # keep text generated
    text_generated = [char2idx["<start>"]]
    # add any other from test string
    for elem in test_string:
        text_generated.append(char2idx[elem])

    # Starter string
    input_eval = torch.tensor(test_string_proc).to(device)
    input_eval = input_eval.unsqueeze(0)
    
    # create input image
    # input_img size ([1])
    input_img = torch.zeros(1).long() 
    input_img[0] = img_num # Pick Images
    input_img = input_img.to(device)

    # just for beam search scores
    beam_table = pd.DataFrame()
    beam_candidates = []

    if prediction_mode == "greedy" or prediction_mode == "sampling":
        # iterate until maximum length
        for i in range(maxcharpred):
            # if current sequences is less than 128
            if len(text_generated) < 128:
                currlen = len(text_generated)
                rem = 128 - currlen
                input_eval = text_generated.copy()
                # prepadding
                for i in range(rem):
                    input_eval.insert(0, 0)

            # if greater than 128, take only last 128
            else:
                input_eval = text_generated.copy()
                input_eval = input_eval[-128:]

            # convert input to tensors
            input_eval = torch.tensor(input_eval).to(device)
            input_eval = input_eval.unsqueeze(0)

            # predict from inputs
            predictions = model(input_img, input_eval)

            # apply softmax to make probabilities
            softmax_layer = nn.Softmax(dim=1)
            predictions = softmax_layer(predictions)

            # Greedy mode 
            if prediction_mode == "greedy":
                best_char = torch.max(predictions, dim=1).indices
                text_generated.append(best_char.item())
                # if predict <end> break loop
                if best_char.item() == char2idx['<end>']:
                    break

            # Sampling mode (based on the probabilities generated)
            elif prediction_mode == "sampling":
                # convert to numpy
                predictions = predictions.detach().cpu().numpy()
                # randomly choose based on probability
                predicted_id = np.random.choice(len(char2idx), size = 1, replace = False, p = predictions[predictions.shape[0] - 1])
                # save generated chars
                text_generated.append(predicted_id[0])
                # if predict end token
                if predicted_id[0] == char2idx['<end>']:
                    break
        
        del test_string_proc
        del input_img
        del input_eval
        del predictions
        del softmax_layer
        
        final = ''
        final += ''.join([idx2char[elem] for elem in text_generated])
        return final
        
    
    # Beam Search
    elif prediction_mode == "beam":
        # Beam Search Prediction
        def predict_beam_search(input_seq):
            # if current sequences is less than 128
            if len(input_seq) < 128:
                currlen = len(input_seq)
                rem = 128 - currlen
                input_eval = input_seq.copy()
                # prepadding
                for i in range(rem):
                    input_eval.insert(0, 0)

            # if greater than 128, take only last 128
            else:
                input_eval = input_seq.copy()
                input_eval = input_eval[-128:]

            # convert input to tensors
            input_eval = torch.tensor(input_eval).to(device)
            input_eval = input_eval.unsqueeze(0)

            # predict from inputs
            predictions = model(input_img, input_eval)

            # apply softmax to make probabilities
            softmax_layer = nn.Softmax(dim=1)
            predictions = softmax_layer(predictions)

            # convert to numpy
            predictions = list(predictions.detach().cpu().numpy().ravel())
            # Take top k scores
            indexes = sorted(range(len(predictions)), key=lambda i: predictions[i])[-beam_width:]
            scores = [predictions[elem] for elem in indexes]
            return indexes, scores


        #### First Iteration of Beam Search
        indexes, scores = predict_beam_search(text_generated)
        # add to df for bookeeping
        col_a = []
        col_b= []      
        for idx, elem in enumerate(indexes):
            beam_temp = text_generated.copy()
            beam_temp.append(elem)
            col_a.append(beam_temp)
            col_b.append(scores[idx])
            beam_candidates.append(beam_temp)

        temp_df = pd.DataFrame()
        temp_df['candidates'] = col_a
        temp_df['score'] = col_b
        beam_table = beam_table.append(temp_df)
        #######

        # Method for bookeeping
        def bookkeep(indexes, scores, curr):
          col_a = []
          col_b= []      
          for idx, elem in enumerate(indexes):
              beam_temp = curr.copy()
              beam_temp.append(elem)
              col_a.append(beam_temp)
              col_b.append(scores[idx])

          temp_df = pd.DataFrame()
          temp_df['candidates'] = col_a
          temp_df['score'] = col_b
          return temp_df

        count_done = 0
        ### Beam Search
        while count_done != beam_width:
            sec_df = pd.DataFrame()
            while len(beam_candidates) != 0:
                curr = beam_candidates.pop()
                indexes, scores = predict_beam_search(curr)
                temp_df = bookkeep(indexes, scores, curr)
                sec_df = sec_df.append(temp_df, ignore_index=True)

            # take top k in pandas dataframe 
            sec_df = sec_df.sort_values(by=['score'], ascending = False).reset_index(drop = True)
            sec_df = sec_df.loc[0:(beam_width - count_done - 1),:]

            # Multiply Scores and place candidates in beam candidates array
            for i in range((beam_width-count_done)):
                curr_score = sec_df.loc[i, 'score']
                curr_arr = sec_df.loc[i, 'candidates']
                # get past score and multiply since it is conditional probability
                m = beam_table['candidates'].apply(lambda x: x == curr_arr[:-1])
                prev_score = beam_table.loc[m, :].loc[:, 'score'].values
                sec_df.loc[i, 'score'] = curr_score * prev_score[0]

                # last elem check -> If <end> add to count_done to reduce candidates
                if curr_arr[-1] == char2idx["<end>"]:
                    count_done += 1
                else:
                    # add to candidates
                    beam_candidates.append(curr_arr)

            # append all candidates to master table
            beam_table = beam_table.append(sec_df, ignore_index = True)

        # filter only those with end token
        def has_endtoken(x):
            if char2idx["<end>"] in x:
                return True
            else:
                return False
        beam_table['contains_end'] = beam_table['candidates'].apply(lambda x: has_endtoken(x))
        beam_table = beam_table[beam_table['contains_end'] == True]

        # sort by scores
        beam_table = beam_table.sort_values(by=['score'], ascending = False)
        
        del test_string_proc
        del input_img
        del input_eval
        del sec_df
        del curr_score
        del curr_arr
         
        final = '' 
        final += ''.join([idx2char[elem] for elem in beam_table.iloc[0, 0] ])
        return final


############ Prediction Method for LSTM
def do_prediction_LSTM(device, model, img_num, prediction_mode, beam_width = 3, maxcharpred = 250, test_string = ""):
    # device = device to predict on "cpu" or "cuda:0" for GPU
    # model = character model
    # img_num = encoded image number (0-98)
    # prediction_mode = "greedy", "sampling", "beam"
    # beam_width = parameter of beam search
    # maxcharpred = maximum character length of predictions
    # test_string = starter string for meme


    ### Method to preprocess input for padding
    def preprocess_input(test_string):
        # convert string to number array
        test_string_proc = [char2idx2["<start>"]]
        # add any test string
        for char in test_string:
            test_string_proc.append(char2idx2[char])
        # add the prepadding
        currlen = len(test_string_proc)
        rem = 199 - currlen  #199 here because length of sequence originally was 199 for input
        test_string_proc.extend([0] * rem)
        return test_string_proc

    # apply method
    # test_string_proc (list of length number of input string)
    test_string_proc = preprocess_input(test_string)

    # keep text generated
    text_generated = [char2idx2["<start>"]]
    # add any other from test string
    for elem in test_string:
        text_generated.append(char2idx2[elem])

    # Starter string
    input_eval = torch.tensor(test_string_proc).to(device)
    
    # create input image
    # input_img size ([1])
    input_img = torch.zeros(1).long() 
    input_img[0] = img_num # Pick Images
    input_img = input_img.to(device)

    # just for beam search scores
    beam_table = pd.DataFrame()
    beam_candidates = []

    if prediction_mode == "greedy" or prediction_mode == "sampling":
        # predict from inputs
        # predictions size (batch_size, 199) where 199 is seqlen
        input_eval = input_eval.unsqueeze(0)
        predictions, _ = model(input_img, input_eval, None, device, prediction_mode = True)
        predictions = torch.squeeze(predictions, dim = 0)

        # apply softmax to make probabilities
        softmax_layer = nn.Softmax(dim=1)
        predictions = softmax_layer(predictions)

        # Greedy mode 
        if prediction_mode == "greedy":
            best_char = torch.max(predictions, dim=1).indices
            best_char = best_char.detach().cpu().numpy()
            best_char = list(best_char)
            # cut off until <end> token
            if char2idx2["<end>"] in best_char:
                endidx = best_char.index(char2idx2["<end>"])
                best_char = best_char[:endidx + 1]
            text_generated.extend(best_char)

        # Sampling mode (based on the probabilities generated)
        elif prediction_mode == "sampling":
            # convert to numpy
            predictions = predictions.detach().cpu().numpy()
            # iterate thorugh the sequence length
            for i in range(predictions.shape[0]):
                # randomly choose based on probability
                predicted_id = np.random.choice(len(char2idx2), size = 1, replace = False, p = predictions[i])
                # save generated chars
                text_generated.append(predicted_id[0])
                # break the loop if <end> token is generated
                if char2idx2["<end>"] == predicted_id[0]:
                    break
        
        del test_string_proc
        del input_img
        del input_eval
        del predictions
        del softmax_layer
        
        final = ''
        final += ''.join([idx2char2[elem] for elem in text_generated])
        return final

    elif prediction_mode == "beam":
        # Beam Search Prediction
        def predict_beam_search(input_seq, pos):
            # convert input to tensors
            input_seq = torch.tensor(input_seq).to(device)
            input_seq = input_seq.unsqueeze(0)

            # predict from inputs
            predictions, _ = model(input_img, input_seq, None, device, prediction_mode = True)

            # apply softmax to make probabilities
            softmax_layer = nn.Softmax(dim=1)
            predictions = softmax_layer(predictions)
            predictions = torch.squeeze(predictions, dim=0)

            # convert to numpy shape (199, 72)
            predictions = predictions.detach().cpu().numpy()
            predictions = predictions[first_nonzero]
            #print(predictions[first_nonzero])
            # Take top k scores
            indexes = sorted(range(len(predictions)), key=lambda i: predictions[i])[-beam_width:]
            scores = [predictions[elem] for elem in indexes]
            return indexes, scores

        #### First Iteration of Beam Search
        first_nonzero = len(text_generated) + 1
        indexes, scores = predict_beam_search(input_eval, first_nonzero)
        # add to df for bookeeping
        col_a = []
        col_b= []      
        for idx, elem in enumerate(indexes):
            beam_temp = text_generated.copy()
            beam_temp.append(elem)
            col_a.append(beam_temp)
            col_b.append(scores[idx])
            beam_candidates.append(beam_temp)

        temp_df = pd.DataFrame()
        temp_df['candidates'] = col_a
        temp_df['score'] = col_b
        beam_table = beam_table.append(temp_df)
        #######

        # Method for bookeeping
        def bookkeep(indexes, scores, curr):
          col_a = []
          col_b= []      
          for idx, elem in enumerate(indexes):
              beam_temp = curr.copy()
              beam_temp.append(elem)
              col_a.append(beam_temp)
              col_b.append(scores[idx])

          temp_df = pd.DataFrame()
          temp_df['candidates'] = col_a
          temp_df['score'] = col_b
          return temp_df
      
        def pad_sequences(curr_seq):
          currlen = len(curr_seq)
          rem = 199 - currlen
          curr_seq.extend([0] * rem)
          return curr_seq


        count_done = 0
        first_nonzero = first_nonzero + 1 # need to keep increasing length
        ### Beam Search
        while count_done != beam_width:
            sec_df = pd.DataFrame()
            while len(beam_candidates) != 0:
                curr = beam_candidates.pop()
                curr_pad = pad_sequences(curr.copy())
                indexes, scores = predict_beam_search(curr_pad, first_nonzero)
                temp_df = bookkeep(indexes, scores, curr)
                sec_df = sec_df.append(temp_df, ignore_index=True)

            # take top k in pandas dataframe 
            sec_df = sec_df.sort_values(by=['score'], ascending = False).reset_index(drop = True)
            sec_df = sec_df.loc[0:(beam_width - count_done - 1),:]


            # Multiply Scores and place candidates in beam candidates array
            for i in range((beam_width-count_done)):
                curr_score = sec_df.loc[i, 'score']
                curr_arr = sec_df.loc[i, 'candidates']
                # get past score and multiply since it is conditional probability
                m = beam_table['candidates'].apply(lambda x: x == curr_arr[:-1])
                prev_score = beam_table.loc[m, :].loc[:, 'score'].values
                sec_df.loc[i, 'score'] = curr_score * prev_score[0]
              
                # last elem check -> If <end> add to count_done to reduce candidates
                if curr_arr[-1] == char2idx2["<end>"]:
                    count_done += 1
                elif curr_arr[-1] == char2idx2["<start>"]:
                    count_done += 1
                else:
                    # add to candidates
                    beam_candidates.append(curr_arr)
     
            # append all candidates to master table
            beam_table = beam_table.append(sec_df, ignore_index = True)

            # increase position for beam search
            first_nonzero += 1

        # filter only those with end token
        def has_endtoken(x):
            if char2idx2["<end>"] in x:
                return True
            else:
                return False
        beam_table['contains_end'] = beam_table['candidates'].apply(lambda x: has_endtoken(x))
        beam_table = beam_table[beam_table['contains_end'] == True]

        # sort by scores
        beam_table = beam_table.sort_values(by=['score'], ascending = False)
        
        final = '' 
        final += ''.join([idx2char2[elem] for elem in beam_table.iloc[0, 0] ])
        return final


############ Post processing to display meme in image format
def display_image_format(base_url, img_num, final_str):
    try:
        ## Define Webdriver options
        # selenium preferences to prevent load images
        # define webdriver options
        chrome_options = webdriver.ChromeOptions()
        chrome_options.binary_location = os.environ.get("GOOGLE_CHROME_BIN")
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--no-sandbox")
                
        # for deployment to heroku use this
        driver = webdriver.Chrome(executable_path=os.environ.get("CHROMEDRIVER_PATH"), chrome_options=chrome_options)
        
        # to run locally use this below with the chromedriver.exe in current path
        # driver = webdriver.Chrome(options=chrome_options)
        
        # get base url
        driver.get(base_url)
                    
        # type in search bar desired memes
        search_bar = driver.find_element_by_id("mm-search")
        if img_num == 27: # manual fix for search
            search_bar.send_keys("don't you squidward")
        else:
            search_bar.send_keys(idx2img[img_num])
        time.sleep(1.5)
        
        
        if img_num == 74: # some exception
            # do second result
            first_search_result = driver.find_element_by_xpath('//*[@id="mm-search-dropdown"]/table/tbody/tr[3]/td[2]')
            first_search_result.click()
        else:
            try:
                # click first result
                first_search_result = driver.find_element_by_xpath('//*[@id="mm-search-dropdown"]/table/tbody/tr[2]/td[2]')
                first_search_result.click()
            except:
                # not found from featured template
                first_search_result = driver.find_element_by_xpath('//*[@id="mm-search-dropdown"]/table/tbody/tr[3]/td[2]')
                first_search_result.click()            
               
        # select private
        private_button = driver.find_element_by_xpath('//*[@id="mm-settings"]/div[9]/div[2]/div/div')
        private_button.click()
            
        # final string preprocessing
        final_str = final_str.replace("<start>", "")
        final_str = final_str.replace("<end>", "")
        final_str = final_str.split("<sep>")
        
        # write text
        try:
            # write text
            for i in range(len(final_str)):
                text_box = driver.find_element_by_xpath('//*[@id="mm-settings"]/div[5]/div[' + str(i+1) +']/div[1]/textarea')
                text_box.send_keys(final_str[i])
        except:
            print("too much text")
                
        # click generate
        generate_button = driver.find_element_by_class_name('mm-generate')
        generate_button.click()
        time.sleep(2)
        
        # curr image + download to currmeme.jpg
        finished_img = driver.find_element_by_id('done-img')
        img_src = finished_img.get_attribute('src')
        urllib.request.urlretrieve(img_src, "currmeme.jpg")
        
        # close driver
        driver.close()
        
        del driver
        del search_bar
        del first_search_result
        del private_button
        del generate_button
        del finished_img
        del img_src
        
        return None
    except:
        return "Error"


# define constants
base_url = 'https://imgflip.com/memegenerator'


############################################### API Methods
# check if server is running
@app.route('/',methods=['GET'])
def get():
    return jsonify({'msg': 'Server running'})
    

# get some sample data memes
@app.route('/getSampleMemes', methods=['GET'])
def getSampleMemes():
    df = pd.read_csv('sample.csv')
    
    # generate index for train test split
    #random.seed(224)

    # generate random indexes for training and testing
    sample_idx = random.sample(range(df.shape[0]), 12) # sampling without replacement
    
    # make image base64 encoding
    def get_encoded_img(image_path):
        img = Image.open(image_path, mode='r')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        my_encoded_img = base64.encodebytes(img_byte_arr.getvalue()).decode('ascii')
        return my_encoded_img
        
    # initialize response
    response_body = {}
    all_images = []
    
    # retreive image and save locally
    def retrieve_image(orig_idx, filename):
        try:
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(df.loc[orig_idx, 'meme_link'], str(filename) + ".jpg")
        except: #try the next one
            if filename == df.shape[0]:
                orig_idx = orig_idx - 1
            elif filename == 0:
                orig_idx = orig_idx + 1
            else:
                orig_idx = orig_idx + 1
                
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(df.loc[orig_idx, 'meme_link'], str(filename) + ".jpg")    
    
    # iterate through all random indexes and get images
    Parallel(n_jobs=-1, require='sharedmem')(delayed(retrieve_image)(elem, idx) for idx, elem in enumerate(sample_idx))
    
    # encode to base64
    for idx, elem in enumerate(sample_idx):
        #retrieve_image(elem, idx)
        temp = get_encoded_img(str(idx) + ".jpg")
        all_images.append(temp)
    
    # delete variable
    del df
    gc.collect()
        
    response_body["all_images"] = all_images
    return make_response(jsonify(response_body), 200)


############################# ML method for prediction
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        if request.is_json:
            method = request.json['method']  #CNN/LSTM
            prediction_mode = request.json['prediction_mode'].lower() #greedy/sampling/beam
            image_num = int(request.json['image_num']) #integer from 0-98 denoting image_id
            test_string = request.json['test_string'] # test string
            
            # initialize response
            response_body = {}
            
            ### Sets up model
            #modelCNN, device = setup_model("CNN")
            #modelLSTM, device = setup_model("LSTM")
            
            # do prediction method
            if method == "CNN":
                model, device = setup_model("CNN")
                start = time.time()
                final = do_prediction_CNN(device, model, image_num, prediction_mode, beam_width = 3, maxcharpred = 250, test_string = test_string)
                end = time.time()
            elif method == "LSTM":
                model, device = setup_model("LSTM")
                start = time.time()
                final = do_prediction_LSTM(device, model, image_num, prediction_mode, beam_width = 3, maxcharpred = 250, test_string = test_string)
                end = time.time()
            
            del model
            del device
            gc.collect()
            
            # do some post processing
            final = re.sub(r'([a-z])\1+', r'\1\1\1', final)
                        
            # make image format from imgflip.com
            _ = display_image_format(base_url, image_num, final)
            
            gc.collect()
            
            # if display does not return error
            if _ == "Error":
                return make_response(jsonify({"message": "Error in Displaying Meme in Image Format"}), 400)
            else:
                # make image base64 encoding
                def get_encoded_img(image_path):
                    img = Image.open(image_path, mode='r')
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='PNG')
                    my_encoded_img = base64.encodebytes(img_byte_arr.getvalue()).decode('ascii')
                    return my_encoded_img

                # make response_body
                encoded_img = get_encoded_img('currmeme.jpg')
                response_body = {'final': final, 'image': encoded_img, 'prediction_time': round(end-start,2)}
                return make_response(jsonify(response_body), 200)
        else:
            return make_response(jsonify({"message": "Request body must be JSON"}), 400)
    else:
        return make_response(jsonify({"message": "Method not implemented POST is allowed"}), 400)

    
if __name__ == '__main__':
    # to run locally use this
    app.run()
    # to deploy to google cloud platform use this
    # app.run(port=8080)