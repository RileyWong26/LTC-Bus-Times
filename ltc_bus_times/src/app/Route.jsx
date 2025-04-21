import React from "react";

export default function Route (props) {

    const abrev = props.Abreviation
    const routes = props.Routes
    const stopName = props.Stop_Name

    return (
        <div>
            <h1>
                {stopName}
            </h1>
            
            <h1>
                {abrev}
            </h1>
            <h1> 
                {routes}
            </h1>
            <button className={'border-2 border-black rounded-2xl'}>
                Find next time
            </button>
        </div>
    )
}
