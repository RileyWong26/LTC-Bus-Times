import React from "react";

export default function Route (props) {

    const abrev = props.Abreviation
    const routes = props.Routes
    const stopName = props.Stop_Name

    return (
        <div className="m-auto w-1/2 text-center bg-gray-400 text-xl flex flex-row">
            <h1 className="w-1/4 m-auto">
                {stopName}
            </h1>
            
            <h1 className="w-1/4">
                {abrev}
            </h1>
            <h1 className="w-1/4"> 
                {routes}
            </h1>
            <button className='border-2 border-black rounded-2xl w-1/4 cursor-pointer 
                bg-white hover:scale-105 hover:bg-green-200 transition delay-150 ease-in-out'>
                Find next time
            </button>
        </div>
    )
}
