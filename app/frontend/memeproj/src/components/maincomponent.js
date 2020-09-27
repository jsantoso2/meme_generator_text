import React from 'react';
import {BrowserRouter as Router, Switch, Route, Redirect} from 'react-router-dom';

import Home from './home';
import Meme from './meme';
import About from './about';


function Main() {
    return (
        <Router>
            <Switch>
                <Route exact path="/" component={() => <Home />} />
                <Route path="/samplememe" component={() => <Meme />} />
                <Route path="/about" component={() => <About />} />
                <Redirect to ='/' />
            </Switch>
        </Router>
    )
}

export default Main;
