import React, {useState} from 'react'

// components
import Navbar from './navbar';

// Styling
import './home.css';

// Import all meme images
import tenguy from '../images/img/10-Guy.jpg';
import anditsgone from '../images/img/Aaaaand-Its-Gone.jpg';
import aintnobodygottimeforthat from '../images/img/Aint-Nobody-Got-Time-For-That.jpg';
import amitheonlyonearoundhere from '../images/img/Am-I-The-Only-One-Around-Here.jpg';
import americanchopperargument from '../images/img/American-Chopper-Argument.jpg';
import ancientaliens from '../images/img/Ancient-Aliens.jpg';
import andeverybodylosestheirminds from '../images/img/And-everybody-loses-their-minds.jpg';
import archer from '../images/img/Archer.jpg';
import awkwardmomentsealion from '../images/img/Awkward-Moment-Sealion.jpg';
import backinmyday from '../images/img/Back-In-My-Day.jpg';
import badluckbrian from '../images/img/Bad-Luck-Brian.jpg';
import badpundog from '../images/img/Bad-Pun-Dog.jpg';
import batmanslappingrobin from '../images/img/Batman-Slapping-Robin.jpg';
import belikebill from '../images/img/Be-Like-Bill.jpg';
import berniesupport from '../images/img/Bernie-I-Am-Once-Again-Asking-For-Your-Support.jpg';
import blackgirlwat from '../images/img/Black-Girl-Wat.jpg';
import blacknutbutton from '../images/img/Blank-Nut-Button.jpg';
import boardroommeeting from '../images/img/Boardroom-Meeting-Suggestion.jpg';
import braceyourselves from '../images/img/Brace-Yourselves-X-is-Coming.jpg';
import noneofmybusiness from '../images/img/But-Thats-None-Of-My-Business.jpg';
import captainpicard from '../images/img/Captain-Picard-Facepalm.jpg';
import changemymind from '../images/img/Change-My-Mind.jpg';
import confessionbear from '../images/img/Confession-Bear.jpg';
import creepywonka from '../images/img/Creepy-Condescending-Wonka.jpg';
import disastergirl from '../images/img/Disaster-Girl.jpg';
import distractedboyfriend from '../images/img/Distracted-Boyfriend.jpg';
import doge from '../images/img/Doge.jpg';
import dontyousquidward from '../images/img/Dont-You-Squidward.jpg';
import drevillaser from '../images/img/Dr-Evil-Laser.jpg';
import drakehotline from '../images/img/Drake-Hotline-Bling.jpg';
import epichandshake from '../images/img/Epic-Handshake.jpg';
import evilkermit from '../images/img/Evil-Kermit.jpg';
import eviltoddler from '../images/img/Evil-Toddler.jpg';
import expandingbrain from '../images/img/Expanding-Brain.jpg';
import faceyoumake from '../images/img/Face-You-Make-Robert-Downey-Jr.jpg';
import findingneverland from '../images/img/Finding-Neverland.jpg';
import firstworldproblems from '../images/img/First-World-Problems.jpg';
import futuramfry from '../images/img/Futurama-Fry.jpg';
import grandmafindsinternet from '../images/img/Grandma-Finds-The-Internet.jpg';
import grumpycat from '../images/img/Grumpy-Cat.jpg';
import swallowpills from '../images/img/Hard-To-Swallow-Pills.jpg';
import hidepain from '../images/img/Hide-the-Pain-Harold.jpg';
import boatcat from '../images/img/I-Should-Buy-A-Boat-Cat.jpg';
import waithere from '../images/img/Ill-Just-Wait-Here.jpg';
import imaginespongebob from '../images/img/Imagination-Spongebob.jpg';
import inhaleseagull from '../images/img/Inhaling-Seagull.jpg';
import pigeon from '../images/img/Is-This-A-Pigeon.jpg';
import jacksparrow from '../images/img/Jack-Sparrow-Being-Chased.jpg';
import laughingmen from '../images/img/Laughing-Men-In-Suits.jpg';
import exit12 from '../images/img/Left-Exit-12-Off-Ramp.jpg';
import leonardocheers from '../images/img/Leonardo-Dicaprio-Cheers.jpg';
import lookatme from '../images/img/Look-At-Me.jpg';
import marksafe from '../images/img/Marked-Safe-From.jpg';
import matrix from '../images/img/Matrix-Morpheus.jpg';
import liedetector from '../images/img/Maury-Lie-Detector.jpg';
import mockingsb from '../images/img/Mocking-Spongebob.jpg';
import monkeypuppet from '../images/img/Monkey-Puppet.jpg';
import mugatusohot from '../images/img/Mugatu-So-Hot-Right-Now.jpg';
import onedoesnotsimply from '../images/img/One-Does-Not-Simply.jpg';
import oprah from '../images/img/Oprah-You-Get-A.jpg';
import philosoraptor from '../images/img/Philosoraptor.jpg';
import picardwtf from '../images/img/Picard-Wtf.jpg';
import putitsomewhere from '../images/img/Put-It-Somewhere-Else-Patrick.jpg';
import rollsafethink from '../images/img/Roll-Safe-Think-About-It.jpg';
import runningaway from '../images/img/Running-Away-Balloon.jpg';
import sadpablo from '../images/img/Sad-Pablo-Escobar.jpg';
import saythatagain from '../images/img/Say-That-Again-I-Dare-You.jpg';
import scumbagsteve from '../images/img/Scumbag-Steve.jpg';
import seenobodycares from '../images/img/See-Nobody-Cares.jpg';
import skepticalbaby from '../images/img/Skeptical-Baby.jpg';
import spartaleonidas from '../images/img/Sparta-Leonidas.jpg';
import sbimmahead from '../images/img/Spongebob-Ight-Imma-Head-Out.jpg';
import starwarsyoda from '../images/img/Star-Wars-Yoda.jpg';
import steveharvey from '../images/img/Steve-Harvey.jpg';
import successkid from '../images/img/Success-Kid.jpg';
import surprisedpikachu from '../images/img/Surprised-Pikachu.jpg';
import thatwouldbe from '../images/img/That-Would-Be-Great.jpg';
import interestingman from '../images/img/The-Most-Interesting-Man-In-The-World.jpg';
import rockdiving from '../images/img/The-Rock-Driving.jpg';
import scrolltruth from '../images/img/The-Scroll-Of-Truth.jpg';
import skepticalkid from '../images/img/Third-World-Skeptical-Kid.jpg';
import successkidtw from '../images/img/Third-World-Success-Kid.jpg';
import trophy from '../images/img/This-Is-Where-Id-Put-My-Trophy-If-I-Had-One.jpg';
import toodamn from '../images/img/Too-Damn-High.jpg';
import trumpbill from '../images/img/Trump-Bill-Signing.jpg';
import tuxedopooh from '../images/img/Tuxedo-Winnie-The-Pooh.jpg';
import twobuttons from '../images/img/Two-Buttons.jpg';
import unclesam from '../images/img/Uncle-Sam.jpg';
import draw25 from '../images/img/UNO-Draw-25-Cards.jpg';
import unsettledtom from '../images/img/Unsettled-Tom.jpg';
import waitingskeleton from '../images/img/Waiting-Skeleton.jpg';
import hannibal from '../images/img/Who-Killed-Hannibal.jpg';
import whowouldwin from '../images/img/Who-Would-Win.jpg';
import womanyelling from '../images/img/Woman-Yelling-At-Cat.jpg';
import xallthey from '../images/img/X-All-The-Y.jpg';
import xxeverywhere from '../images/img/X-X-Everywhere.jpg';
import yuno from '../images/img/Y-U-No.jpg';
import gotmore from '../images/img/Yall-Got-Any-More-Of-That.jpg';
import heardyou from '../images/img/Yo-Dawg-Heard-You.jpg';

// import image indexing
import img2idx from './img2idx.json';

// Material UI stuff
import { Grid, MenuItem, Select, TextField, Button, CircularProgress } from '@material-ui/core';
import Dialog from '@material-ui/core/Dialog';
import DialogActions from '@material-ui/core/DialogActions';
import DialogContent from '@material-ui/core/DialogContent';
import DialogContentText from '@material-ui/core/DialogContentText';
import DialogTitle from '@material-ui/core/DialogTitle';
import Alert from '@material-ui/lab/Alert';


function Home() {
    // swap key and value
    function swap(json){
        var ret = {};
        for(var key in json){
          ret[json[key]] = key;
        }
        return ret;
    }
    // load json for img2idx mapping
    const idx2img = swap(img2idx);    

    // all images in a list
    const all_images = [tenguy, anditsgone, aintnobodygottimeforthat, amitheonlyonearoundhere, americanchopperargument, ancientaliens,
        andeverybodylosestheirminds, archer, awkwardmomentsealion, backinmyday, badluckbrian, badpundog, batmanslappingrobin, belikebill,
        berniesupport, blackgirlwat, blacknutbutton, boardroommeeting, braceyourselves, noneofmybusiness, captainpicard, changemymind,
        confessionbear, creepywonka, disastergirl, distractedboyfriend, doge, dontyousquidward, drevillaser, drakehotline, epichandshake,
        evilkermit, eviltoddler, expandingbrain, faceyoumake, findingneverland, firstworldproblems, futuramfry, grandmafindsinternet,
        grumpycat, swallowpills, hidepain, boatcat, waithere, imaginespongebob, inhaleseagull, pigeon, jacksparrow, laughingmen,
        exit12, leonardocheers, lookatme, marksafe, matrix, liedetector, mockingsb, monkeypuppet, mugatusohot, onedoesnotsimply, oprah,
        philosoraptor, picardwtf, putitsomewhere, rollsafethink, runningaway, sadpablo, saythatagain, scumbagsteve, seenobodycares,
        skepticalbaby, spartaleonidas, sbimmahead, starwarsyoda, steveharvey, successkid, surprisedpikachu, thatwouldbe, interestingman,
        rockdiving, scrolltruth, skepticalkid, successkidtw, trophy, toodamn, trumpbill, tuxedopooh, twobuttons, draw25, unclesam,
        unsettledtom, waitingskeleton, hannibal, whowouldwin, womanyelling, xallthey, xxeverywhere, yuno, gotmore, heardyou];
        
    const all_color = Array(all_images.length).fill("white");
    const len_arr = all_images.length;
    const all_prediction_method = ["Greedy", "Sampling", "Beam"];

    // React hooks for current starter term
    const [starterString, setStarterString] = useState('');
    const [selectedImage, setSelectedImage] = useState(-1);
    const [selectedModel, setSelectedModel] = useState('CNN');
    const [selectedPrediction, setSelectedPrediction] = useState('Greedy');
    const [predictionMethodArr, setPredictionMethodArr] = useState(all_prediction_method);
    const [cardColor, setCardColor] = useState(all_color);
    const [predMemeStr, setMemeStr] = useState('');
    const [predImg, setPredImg] = useState();
    const [predTime, setPredTime] = useState(0.0);
    const [loading, setLoading] = useState(false);
    const [isopenDialog, setIsOpenDialog] = useState(false);
    const [serverError, setServerError] = useState(false);
    const handleClose = () => { setIsOpenDialog(false); };

    
    // edit the current starter string
    const editStarterString = (e) => {
        // limit to only 10 characters
        if (e.target.value.length > 10) {
            console.log(e.target.value.substring(0,10));
            setStarterString(e.target.value.substring(0,10));
        } else {
            setStarterString(e.target.value);
        }
    }

    // edit the current selected image and set the border to red
    const editSelectedImage = (e) => {
        setSelectedImage(e.target.alt);
        const temp = Array(len_arr).fill("white");
        temp[e.target.alt] = "red";
        setCardColor(temp);
    }

    // edit selected model
    const editSelectedModel = (e) => {
        setSelectedModel(e.target.value);
        if (e.target.value === "LSTM"){
            setPredictionMethodArr(["Greedy"]);
        } else {
            setPredictionMethodArr(all_prediction_method);
        }
    }

    // edit selected prediction
    const editSelectedPrediction = (e) => {
        setSelectedPrediction(e.target.value);
    }


    // fetch predicted data
    const getPredictionClick = () => {
        if (selectedImage === -1){
            setIsOpenDialog(true);
        } else {
            setLoading(true);
            fetch('https://meme-generator-flask.herokuapp.com/predict', 
                    {mode: 'cors', method: "POST", body: JSON.stringify({"method": selectedModel, "prediction_mode": selectedPrediction, "image_num": selectedImage, "test_string": starterString}), headers: { "Content-Type": "application/json"} })
                .then(res => res.json()).then(data => {
                    setMemeStr(data["final"]);
                    setPredImg(data["image"]);
                    setPredTime(data["prediction_time"]);
                    setLoading(false);
            }).catch(() => {
                setServerError(true);
                setLoading(false);
                console.log("error in Predict");
            });
        }
    } 


    return (
        <div>
            <Navbar />
            {/* ################## Displays all the Meme in tiles ###############################*/}
            <div className="home_title">
                <h2>Pick a Meme Image:</h2>
                    <div style={{display: "flex", flexWrap: "wrap"}}>
                        {all_images.map( (img_name, idx) => (
                            <img className = "home_card_images" onClick = {editSelectedImage} 
                                src = {img_name} alt = {idx} key={"img" + idx} 
                                style={{borderStyle:"solid", borderWidth: "8px", borderColor: cardColor[idx]}}
                            />
                        ) ) }
                    </div>
            </div>
            
            {/* ################## Labels for Choices ###############################*/}
            <Grid container spacing={0}>
                <Grid item xs={12} sm={4} md={4}>
                    <h3 style={{margin: "0px 20px 10px 20px"}}>Prefix Text: </h3>
                </Grid>
                <Grid item xs={12} sm={4} md={4}>
                    <h3 style={{margin: "0px 20px 10px 20px"}}>Model Architecture:</h3>
                </Grid>
                <Grid item xs={8} sm={4} md={4}>
                    <h3 style={{margin: "0px 20px 10px 20px"}}>Prediction Method:</h3>
                </Grid>
            </Grid>
            
            {/* ################## Text Area and Select Box ###############################*/}
            <Grid container spacing={0}>
                <Grid item xs={12} sm={4} md={4}>
                    <TextField value = {starterString} onChange = {editStarterString} 
                                variant = "outlined" placeholder="Enter Prefix Text (Optional)"
                                size="small"
                                style={{width: "16rem", margin: "0px 0px 0px 20px"}}
                    />
                    <Alert severity="warning" style={{marginLeft: "20px", width: "14rem"}}>Limited to only 10 characters!</Alert>
                </Grid>
                <Grid item xs={12} sm={4} md={4}>
                    <Select value={selectedModel} onChange={editSelectedModel} variant = "outlined"
                        style={{height: "40px", margin: "0px 20px 0px 20px"}}>
                        <MenuItem value={'CNN'} name="CNN"> CNN </MenuItem>
                        <MenuItem value={'LSTM'} name="LSTM"> LSTM </MenuItem>
                    </Select>
                </Grid>
                <Grid item xs={12} sm={4} md={4}>
                    <Select value={selectedPrediction} onChange={editSelectedPrediction} variant = "outlined"
                        style={{height: "40px", margin: "0px 20px 0px 20px"}}>
                        {predictionMethodArr.map(elem => (
                            <MenuItem value={elem} key={elem}>{elem}</MenuItem>
                        ))}
                    </Select>
                </Grid>
            </Grid>
            
            {/* ################## General info about selected Image ###############################*/}
            <div style={{margin: "20px 20px 0px 20px",display: "flex", alignItems: "center"}}>
                <h3 style={{marginRight: "50px"}}>Selected Image: </h3>
                <p>{selectedImage === -1 ? "No Image Selected" : idx2img[selectedImage]}</p>
            </div>
            
            {/* ################## Predict Row ###############################*/}
            <div style={{margin: "0px 20px 0px 20px"}}>
                <div style={{display: "flex", alignItems: "center"}}>
                    <h3 style={{marginRight: "40px"}}>Predict Meme: </h3>
                    <Button style={{width: "20rem", backgroundColor:"black", margin: "20px 20px 20px 20px", color: "white"}}
                            onClick={getPredictionClick}>
                            Predict
                    </Button>
                    {loading ? <CircularProgress color="primary" /> : <div></div>}
                </div>
                
                <div style={{margin: "0px 0px 20px 0px"}}>
                    <Alert severity="warning">Warning! 1) Prediction might take up to several minutes! 2) No profanity filtering used on this dataset! 3) Server Error can be encountered!</Alert>
                    <Alert severity="info">Info Alert! If a blank image is rendered, Please click predict again! This is an error getting the image from imgflip.com</Alert>
                </div>

                <div style={{margin: "0px 0px 20px 0px"}}>
                    {serverError ? 
                        <Alert severity="error" onClose={() => {setServerError(false)}} >Error! Server is not responding correctly!</Alert>
                    : ''}
                </div>
                
                {/* ################## Display Prediction Results ###############################*/}
                {predImg ?
                    <div>
                        <h3>Predicted String:</h3>
                        <p style={{margin: "10px 0px 20px 0px"}}>{predMemeStr}</p> 
                        <img className="finalImage" src={`data:image/png;base64,${predImg}`} alt="finalImage"
                             style={{borderStyle:"solid", borderWidth: "8px", borderColor: "black"}}/>
                        <div style={{display: "flex", alignItems: "center", margin: "0px 0px 20px 0px"}}>
                            <h3>Prediction Time: </h3>
                            <p style={{margin: "20px 0px 20px 20px"}}>{predTime + "s"}</p> 
                        </div>
                    </div>
                : ''}

            </div>


            {/*########################## Dialog to generate meme without selecting image ##################### */}
            <Dialog open={isopenDialog} onClose={handleClose}>
                <DialogTitle id="alert-dialog-title">{"Error, Please check one of the following!"}</DialogTitle>
                <DialogContent>
                    <DialogContentText>
                        1) Select a Meme Image!
                    </DialogContentText>
                </DialogContent>
                <DialogActions>
                    <Button onClick={handleClose} color="primary" autoFocus> Close </Button>
                </DialogActions>
            </Dialog>
            
        </div>
    )
}

export default Home;
