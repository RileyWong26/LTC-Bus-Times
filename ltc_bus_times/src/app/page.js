'use client'
import Image from "next/image";
import React, {useEffect, useState} from "react";

export default function Home() {
  
  const [routes, setRoutes] = useState()

  const fetchRoutes = async() => {
    fetch('http://172.22.10.99:5001/Routes', {
      method:'GET'
    })
    .then(response => response.json())
    .then(data => setRoutes(data))
    .catch(error => console.log(error));
  }


  return (
    <div className="">
      <input />
      <button onClick={() => fetchRoutes()}>fetch</button>
      <button onClick={() => console.log(routes)}>display</button>
    </div>
  );
}
