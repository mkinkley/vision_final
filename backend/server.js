const express = require('express');
const app = express();

// set up an arbitrary port to access on localhost
const port = process.env.PORT || 8081;

// make a call to handle any possible direction
//const route = require('./src/routers/router.js');

app.use(express.static('shat.html'));
app.use('/', );

app.listen(port, () => {
    console.log(`Server is up on port ${port}`);
});

module.exports = app;

