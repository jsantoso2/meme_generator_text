import React, {useState, useEffect} from 'react';

// components
import Navbar from './navbar';


// Material UI Imports
import Alert from '@material-ui/lab/Alert';
import Skeleton from '@material-ui/lab/Skeleton';



function Meme() {
    const [allImages, setAllImages] = useState([]);
    const [isLoading, setLoading] = useState(true);

    // use effect to fetch meme images
    useEffect(() => {
        fetch('https://meme-generator-flask.herokuapp.com/getSampleMemes', {mode: "cors"}).then(res => res.json()).then(data => {
            setAllImages(data['all_images']);
            setLoading(false);
        }).catch(() => {
            console.log("error in getOneBusinessData");
            setLoading(false);
        })
    }, []); 
        


    return (
        <div>
            <Navbar />

            {/*############################### Display Meme Images Skeleton ##################### */}
            <div style={{margin: "20px 20px 20px 20px"}}>
                <h1>Sample Memes from Dataset: </h1>
                <Alert severity="warning">Warning! No profanity filtering used on this dataset!</Alert>
                <Alert severity="info">Please Refresh to get more sample memes!</Alert>
                {allImages.length > 0 ?
                    <div style={{flex: "1", display: "flex", flexWrap: "wrap", alignItems: "center", margin: "20px 20px 20px 0px"}}>
                        {allImages.map((elem,i) => (
                            <div style={{height: "220px", width: "220px"}} key={"div" + i}>
                                <img src={`data:image/png;base64,${elem}`} alt="finalImage" key={i}
                                style={{borderStyle:"solid", borderWidth: "5px", borderColor: "black", maxWidth: "200px", maxHeight: "200px"}}/>
                            </div>
                        ))} 
                    </div>
                : <div></div>
                }
            </div>

            {/*############################### Loading Skeleton ##################### */}
            {isLoading ? 
                <div style={{marginLeft: "20px"}}>
                    <div style={{flex: "1", display: "flex", flexWrap: "wrap", alignItems: "center", margin: "0px 20px 20px 20px"}}>
                        <Skeleton animation="wave" style={{height: "220px", width: "220px", marginRight: "20px"}}/>
                        <Skeleton animation="wave" style={{height: "220px", width: "220px", marginRight: "20px"}}/>
                        <Skeleton animation="wave" style={{height: "220px", width: "220px", marginRight: "20px"}}/>
                        <Skeleton animation="wave" style={{height: "220px", width: "220px", marginRight: "20px"}}/>
                        <Skeleton animation="wave" style={{height: "220px", width: "220px", marginRight: "20px"}}/>
                        <Skeleton animation="wave" style={{height: "220px", width: "220px", marginRight: "20px"}}/>
                        <Skeleton animation="wave" style={{height: "220px", width: "220px", marginRight: "20px"}}/>
                        <Skeleton animation="wave" style={{height: "220px", width: "220px", marginRight: "20px"}}/>
                        <Skeleton animation="wave" style={{height: "220px", width: "220px", marginRight: "20px"}}/>
                        <Skeleton animation="wave" style={{height: "220px", width: "220px", marginRight: "20px"}}/>
                        <Skeleton animation="wave" style={{height: "220px", width: "220px", marginRight: "20px"}}/>
                        <Skeleton animation="wave" style={{height: "220px", width: "220px", marginRight: "20px"}}/>
                    </div>
                </div>
            : <div></div>}

            {/*################################ References ########################### */}
            <div style={{margin: "20px 20px 20px 20px"}}>
                <h3>Dataset: </h3>
                <li><a href="https://imgflip.com/">https://imgflip.com/</a></li>
                <li><a href="https://github.com/schesa/ImgFlip575K_Dataset">https://github.com/schesa/ImgFlip575K_Dataset</a></li>
            </div>



        </div>
    )
}

export default Meme;
