import React from 'react';
import {Link} from 'react-router-dom';

// Styling and Images
import './navbar.css';
import meme_generator_logo from '../images/logo.JPG';

// Material UI Imports
import { Grid } from '@material-ui/core';


function Navbar() {
    return (
        <div>
            <div className="navbar_bg">
                <Grid container spacing={0}>
                    {/* ############################ Meme Generator Logo ########################### */}
                    <Grid item xs={5} sm={3} md={2}>
                        <Link to='/'>
                            <img className = "navbar_logo" src = {meme_generator_logo} alt = "logo" />
                        </Link>
                    </Grid>
                    {/* ############################ Navbar Menu Icons ########################### */}
                    <Grid item xs={7} sm={4} md={4}>
                        <div className = "navbar_menuitems">
                            <Link to='/'>
                                <p>Home</p>
                            </Link>
                            <Link to='/samplememe'>
                                <p>Memes</p>
                            </Link>
                            <Link to='/about'>
                                <p>About</p>
                            </Link>
                        </div>
                    </Grid>
                </Grid>
            </div>
        </div>
    )
}

export default Navbar;
