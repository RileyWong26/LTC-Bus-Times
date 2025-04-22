import React from "react";

export default function Route (props) {

    const abrev = props.Abreviation
    const routes = props.Routes
    const stopName = props.Stop_Name

    return (
        <div className="m-auto w-1/2 text-center">
            <h1 className="w-1/4 m-auto">
                {stopName}
            </h1>
            
            <h1 className="w-1/4">
                {abrev}
            </h1>
            <h1 className="w-1/4"> 
                {routes}
            </h1>
            <button className='border-2 border-black rounded-2xl w-1/4'>
                Find next time
            </button>
        </div>
    )
}
