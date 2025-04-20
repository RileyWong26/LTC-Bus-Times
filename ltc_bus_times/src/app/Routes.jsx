import React, {useEffect, useState} from "react";

const Display = () => {

    const [routes, setRoutes] = useState([]);

    // POPULATE ROUTES
    useEffect(() => { 
        const getRoute = async() => {
            await fetch('http://127.0.0.1:5001/Routes', {
            method:'GET'
        })
        .then(response => response.json())
        .then(data => setRoutes(data))
        .catch(error => console.log(error));
        }

        getRoute();

    }, [])

    return (
        <div>
            {routes.map((item) => (
        <p key={item['Route ID']}>{item.Abreviation}</p>
      ))}
            <button onClick={() => console.log(routes)}>yoyo</button>
        </div>
    )
}


export default Display;
