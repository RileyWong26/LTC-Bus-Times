'use client'
import Image from "next/image";
import React, {useEffect, useState} from "react";
import Routes from "./Routes";

export default function Home() {
 
   return (
    <div className=" items-center content-center m-auto text-center">
      <Routes />
    </div>
  );
}
