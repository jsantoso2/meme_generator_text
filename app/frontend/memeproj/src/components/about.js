import React from 'react'

// components
import Navbar from './navbar';

function about() {
    return (
        <div>
            <Navbar />
            <div style={{margin: "20px"}}>
                <h1>About This Project: </h1>
                <p style={{margin: "20px"}}>This project was done for COMS 4995 Applied Deep Learning Course at Columbia. </p>

                <h2 style={{marginBottom: "10px"}}>Dataset: </h2>
                <li>575k Scraped Memes from Imgflip.com (Retrieved ~ Apr 2020) <a href="https://github.com/schesa/ImgFlip575K_Dataset">https://github.com/schesa/ImgFlip575K_Dataset</a></li>
                <li>100 Most Popular Memes <a href="https://imgflip.com/popular_meme_ids">https://imgflip.com/popular_meme_ids</a></li>
                
                <br/>

                <h2 style={{marginBottom: "10px"}}>References/Inspirations: </h2>
                <li>CNN Architecture: <a href="https://towardsdatascience.com/meme-text-generation-with-a-deep-convolutional-network-in-keras-tensorflow-a57c6f218e85">https://towardsdatascience.com/meme-text-generation-with-a-deep-convolutional-network-in-keras-tensorflow-a57c6f218e85</a></li>
                <li>LSTM Architecture: <a href="https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html">https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html</a></li>
                <li>Meme Text Generation: <a href="https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1184/reports/6909159.pdf">https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1184/reports/6909159.pdf</a></li>
                
                <br/>

                <h2 style={{marginBottom: "10px"}}>For More Details, Please Visit: </h2>
                <p><a href="https://github.com/jsantoso2/meme_generator_text">https://github.com/jsantoso2/meme_generator_text</a></p>
            </div>
        </div>
    )
}

export default about
