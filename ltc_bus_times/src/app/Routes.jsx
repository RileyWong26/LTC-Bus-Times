import React, {useEffect, useState} from "react";
import Route from "./Route"

const Display = () => {

    const [routes, setRoutes] = useState([]);
    const [currentRoutes, setCurrentRoutes] = useState([]);

    // POPULATE ROUTES
    useEffect(() => { 
        const getRoute = async() => {
            await fetch('http://127.0.0.1:5001/Routes', {
            method:'GET'
        })
        .then(response => response.json())
        .then(data => {setRoutes(data); setCurrentRoutes(data);})
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

    return (
        <div>
            <input className="border-2 border-black" id="input"
                onChange={()=> searchRoutes(document.getElementById("input"))}
                placeholder="Enter Route"
                />

            {currentRoutes.map((item) => (
                <Route  Abreviation = {item.Abreviation }
                 Routes= {item.Routes}
                 stopName= {item['Stop Name']}/>
            ))}
        </div>
    )
}


export default Display;
