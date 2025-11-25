import { useState } from 'react'
import '../App.css'
import '../index.css'
import Square from './Square'



function Column({ value, index, handleClick, p1c, p2c, emptytoken }) {



    return (
        <div onClick={() => handleClick(index)} className="flex flex-col">
            {value.map((v, i) => (
                <Square
                    key={i}
                    value={v}
                    p1c={p1c}
                    p2c={p2c}
                    emptytoken={emptytoken}
                />
            ))}
        </div>
    );
}

export default Column


