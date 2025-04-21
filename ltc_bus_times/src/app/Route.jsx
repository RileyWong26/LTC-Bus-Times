import React from "react";

export default function Route (props) {

    const abrev = props.Abreviation
    const routes = props.Routes
    const stopName = props.Stop_Name

    return (
        <div>
            {abrev} 
            {routes}
            {stopName}
        </div>
    )
}
