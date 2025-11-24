import { useState } from 'react'
import '../App.css'
import '../index.css'
import { useLocation } from "react-router-dom";
import { useNavigate } from 'react-router-dom';
import { PlayerType } from '../PlayerType';




function MainPage() {

    const navigate = useNavigate();

    function playGame(player1, player2) {
        navigate("/play", { state: { player1: player1, player2: player2 } });// 1 = user, 2 = minimax , 3 = neural network
    }

    return (
        <>
            <div className="flex-col flex justify-center gap-40 items-center">
                <h1 className="text-9xl font-bold bg-gradient-to-t from-red-500 to-yellow-500 bg-clip-text text-transparent">Connect 4</h1>
                <div className="flex-col flex justify-center gap-10 items-center">
                    <button onClick={() => playGame(PlayerType.USER, PlayerType.CNN)} class="btn w-40 h-10 text-xl">Play</button>
                    <button onClick={() => playGame(PlayerType.MINIMAX, PlayerType.CNN)} class="btn w-60 h-15 text-xl ">Neural network vs Minimax </button>
                </div>


            </div>

        </>
    )
}

export default MainPage
