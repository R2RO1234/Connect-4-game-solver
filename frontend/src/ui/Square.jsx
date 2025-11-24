import '../App.css'
import '../index.css'



function Square({ value, index, p1c, p2c }) {
    const color = value === 0 ? "white" : value === 1 ? p1c : p2c;
    return (
        <>

            <div className="w-18 h-18 flex items-center justify-center bg-gray-100">
                <div aria-label="status" className="status w-16 h-16 rounded-full " style={{ backgroundColor: color }}></div>

            </div>

        </>
    )

}

export default Square
