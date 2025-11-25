import { PlayerType } from "../PlayerType";
import { convertRole } from "../Pages/BoardPage";
import { EndGame } from "../PlayerType";

function DisplayTurn({ current_player_type, currentPlayer, p1c, p2c, isFetching, both_not_users, isOver }) {
    const text = convertRole(current_player_type);
    let color = currentPlayer === 1 ? p1c : p2c;

    let message;
    if (isOver !== EndGame.NOTOVER) {

        if (isOver === EndGame.DRAW) {
            color = "black"
            message = "Game ended in a draw."
        }
        else if (isOver === EndGame.P1W) {
            color = p1c
            message = "Player 1 won."
        }
        else if (isOver === EndGame.P2W) {
            color = p2c
            message = "Player 2 won."
        }

    }
    else {
        if (current_player_type === PlayerType.USER) {
            message = "It's the user turn";
        } else {
            if (both_not_users && !isFetching) {
                message = both_not_users ? "Press Button to get move" : "";

            } else {
                message = `${text} is thinking`;
            }
        }
    }


    const showSpinner = isFetching && isOver === EndGame.NOTOVER;

    return (
        <div className="flex flex-row justify-center items-center mb-5 gap-3 h-10">
            <h1 className="text-4xl font-bold" style={{ color }}>
                {message}
            </h1>

            {showSpinner && (
                <span
                    style={{ backgroundColor: color }}
                    className="loading loading-dots w-15"
                ></span>
            )}
        </div>
    );
}


export default DisplayTurn;
