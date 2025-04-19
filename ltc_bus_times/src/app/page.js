'use client'
import Image from "next/image";
import React, {useEffect, useState} from "react";

export default function Home() {
  
  const [routes, setRoutes] = useState([])

  const fetchRoutes = async() => {
    fetch('http://127.0.0.1:5001/Routes', {
      method:'GET'
    })
    .then(response => response.json())
    .then(data => setRoutes(data))
    .catch(error => console.log(error));
  }

 
  return (
    <div className=" items-center content-center">
      <input className=" border-2 border-black" onClick={() => console.log(routes)}/>
      <button onClick={() => fetchRoutes()}>fetch</button>
      <button onClick={() => console.log(routes)}>display</button>

      {routes.map((item) => (
        <p key={item['Route ID']}>{item.Abreviation}</p>
      ))}
    </div>
  );
}
