import { useState } from 'react'
import '../App.css'
import '../index.css'
import { useNavigate } from 'react-router-dom';
import Column from '../ui/Column'
import { useLocation } from "react-router-dom";
import { PlayerType } from '../PlayerType';
import StatusPlayer from '../ui/StatusPlayer';
import DisplayTurn from '../ui/DisplayTurn';
import { EndGame } from '../PlayerType';


export function convertRole(player) {
    switch (player) {
        case PlayerType.USER:
            return "User";
        case PlayerType.CNN:
            return "AI Player";
        case PlayerType.MINIMAX:
            return "Minimax Player";
        default:
            return "Unknown Player";
    }
}
const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
const player1C = prefersDark ? "#9ceb78" : "red"
const player2C = prefersDark ? "deeppink" : "seagreen"//#383d4a
const boardBackground = prefersDark ? "#31384a" : "#f3f4f6"
const emptytoken = prefersDark ? "#d1daed" : "white"

function BoardPage() {

    const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

    const location = useLocation();

    let player1 = PlayerType.USER;
    let player2 = PlayerType.CNN;

    if (location.state) {
        player1 = location.state.player1 || PlayerType.USER;
        player2 = location.state.player2 || PlayerType.CNN;
    }

    let sp1 = convertRole(player1)
    let sp2 = convertRole(player2)
    if (player1 === PlayerType.USER && player2 === PlayerType.USER) {
        sp1 = "player1"
        sp2 = "player2"
    }

    const navigate = useNavigate();

    const both_users = (player1 === PlayerType.USER && player2 === PlayerType.USER)
    const both_not_users = (player1 != PlayerType.USER && player2 != PlayerType.USER)

    const [boardState, setBoardState] = useState([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]); // first axis are columns

    const [currentPlayer, setCurrentPlayer] = useState(1)
    const [currentType, setCurrentType] = useState(player1)
    const [isFetching, setIsFetching] = useState(false);
    const [isOver, setIsOver] = useState(EndGame.NOTOVER);
    console.log(EndGame[2])

    // ---------------------------
    // Helper functions
    // ---------------------------

    function getLowestSquare(arr, column_index) {
        let index = arr.length - 1
        while (index >= 0) {
            if (arr[column_index][index] == 0) return index
            index -= 1
        }
        return -1; // no empty square
    }

    function handleMove(column_index, row_index) {

        setBoardState(prev => {
            const newState = prev.map(col => [...col]); // deep copy
            newState[column_index][row_index] = currentPlayer; // change the specific square
            return newState;
        })
        setCurrentPlayer(p => -p);
        setCurrentType(prev => (prev === player1 ? player2 : player1));
    }

    // ---------------------------
    // User interaction handlers
    // ---------------------------

    async function handleClick(column_index) {
        if (currentType !== PlayerType.USER || isOver !== EndGame.NOTOVER) return;

        const row = getLowestSquare(boardState, column_index);
        if (row === -1) return;

        // user move
        handleMove(column_index, row);

        if (both_users) return;

        // schedule AI move with the correct player info
        const nextPlayerNumber = currentPlayer === 1 ? -1 : 1;
        const nextPlayerType = currentType === player1 ? player2 : player1;

        const nextBoard = boardState.map(col => [...col]);
        nextBoard[column_index][row] = currentPlayer; // recreating the move

        setIsFetching(true);
        try {
            await fetch_and_apply(nextBoard, nextPlayerType, nextPlayerNumber, false);
        } finally {
            setIsFetching(false);
        }
    }

    async function handleNextMove() {
        if (isFetching || isOver !== EndGame.NOTOVER) return; // block multiple calls


        const nextPlayerNumber = currentPlayer === 1 ? 1 : -1;
        const nextPlayerType = currentType === player1 ? player1 : player2;

        const nextBoard = boardState.map(col => [...col]);
        nextBoard[0][0] = nextBoard[0][0]


        setIsFetching(true);
        try {
            await fetch_and_apply(nextBoard, nextPlayerType, nextPlayerNumber, false);
        } finally {
            setIsFetching(false);
        }
    }

    // ---------------------------
    // AI / fetch functions
    // ---------------------------



    async function fetch_and_apply(board, playerType, playerNumber, fake = true) {


        let column;

        if (fake) {
            await new Promise(r => setTimeout(r, 500));
            column = 0
            while (getLowestSquare(board, column) == -1) {
                column++;
            }
        }
        else {
            const res = await fetch("/api/getMove", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ board: board, model: playerType })
            });

            if (!res.ok) return;

            let result = await res.json();
            column = result.column
            const flag = result.flag

            if (flag != 0) {
                console.log("end of game: " + flag) //3 for draw , 1 for player(1) win, or 2 for player(-1) win, 0 for nothing
                console.log("the flag is: " + EndGame[flag])
                setIsOver(flag)
                if (column < 0) return
            }

        }

        console.log("this column is valid" + column)
        setBoardState(prev => {
            const row = getLowestSquare(prev, column);
            if (row === -1) return prev;

            const newBoard = prev.map(col => [...col]);
            newBoard[column][row] = playerNumber;

            return newBoard;
        });

        setCurrentPlayer(p => -p);
        setCurrentType(t => (t === player1 ? player2 : player1));
    }

    // ---------------------------
    // UI helpers
    // ---------------------------
    function resetBoard() {
        setBoardState([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],])
        setCurrentPlayer(1)
        setCurrentType(player1)
        setIsOver(EndGame.NOTOVER)
    }

    function goMainPage() {
        navigate('/', { replace: true });
    }

    // ---------------------------
    // Render
    // ---------------------------

    return (
        <>
            <DisplayTurn
                current_player_type={currentType}
                currentPlayer={currentPlayer}
                p1c={player1C}
                p2c={player2C}
                isFetching={isFetching}
                both_not_users={both_not_users}
                isOver={isOver}
            />

            <div className="flex flex-row items-center justify-center w-fit mx-auto p-5 rounded-xl" style={{ backgroundColor: boardBackground }}>
                {boardState.map((value, index) => (
                    <Column
                        p1c={player1C}
                        p2c={player2C}
                        value={value}
                        index={index}
                        handleClick={handleClick}
                        emptytoken={emptytoken}
                    />
                ))}
            </div>

            <div className="flex flex-row gap-10 justify-center mt-5">
                <StatusPlayer color={player1C} text={sp1} />
                <StatusPlayer color={player2C} text={sp2} />
            </div>

            <div className="pt-5 gap-5 flex justify-center  items-center">
                <button onClick={goMainPage} className="btn btn-lg btn-outline btn-primary">Main Menu</button>
                <button onClick={resetBoard} class="btn btn-lg btn-outline btn-error">Reset </button>
                {both_not_users && (
                    <button onClick={handleNextMove} className="btn btn-lg btn-outline" disabled={isFetching || isOver !== EndGame.NOTOVER} >get next move </button>
                )}
            </div>
        </>
    )
}

export default BoardPage
