import React, {useEffect, useState} from "react";
import Route from "./Route"


const Display = () => {

    const [routes, setRoutes] = useState([]);
    const [currentRoutes, setCurrentRoutes] = useState([]);
    const [display, setDisplay] = useState([])

    // POPULATE ROUTES
    useEffect(() => { 
        const getRoute = async() => {
            await fetch('http://127.0.0.1:5001/Routes', {
            method:'GET'
        })
        .then(response => response.json())
        .then(data => {setRoutes(data); setCurrentRoutes(data); setDisplay(data.slice(0, 20))})
        .catch(error => console.log(error));
        }

        getRoute();

    }, [])

    // FILTER ROUTES BY ABREV FOR NOW
    const searchRoutes = (input) => {
        const sub = input.value.toUpperCase();
        const tempRoute = [];
        for(let i = 0; i<routes.length; i++){
            if (routes[i].Abreviation.includes(sub) || routes[i]['Stop Name'].toUpperCase().includes(sub)) tempRoute.push(routes[i]);

        }
        setCurrentRoutes(tempRoute);    
    }

    window.onscroll = () => {
        console.log('1 ' + window.innerHeight + document.documentElement.scrollTop)
        console.log('2 ' + document.documentElement.offsetHeight)
        if(window.innerHeight + document.documentElement.scrollTop === document.documentElement.offsetHeight){
            console.log('hi');
        }
    }
    return (
        <div>
            <input className="border-2 border-black" id="input"
                onChange={()=> searchRoutes(document.getElementById("input"))}
                placeholder="Enter Route"
                />

            {display.map((item) => (
                <Route  Abreviation = {item.Abreviation }
                 Routes= {item.Routes}
                 Stop_Name= {item['Stop Name']}/>
            ))}
        </div>
    )
}


export default Display;
