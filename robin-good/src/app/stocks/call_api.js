//api call scripts

//constants
require('dotenv').config();

const url =  'https://www.alphavantage.co/query?';
const apikey = process.env.API_KEY;

export default async function get_stock_info(ticker){
    
    //fetch data and convert to json
    const response = await fetch(url+"function=TIME_SERIES_WEEKLY&symbol=${ticker}&apikey=${apikey}")
    .then(response => response.json())
    .then(data => {
        console.log(data);
    })
    .catch(error => {
        console.error("get_stock:ERROR: could not fetch stock",error);
    });

    return response;
}