



function StatusPlayer({ color, text }) {

    return (
        <>

            <div className="flex flex-row gap-2 items-center">
                <div aria-label="status" className="status w-10 h-10 rounded-full " style={{ backgroundColor: color }}></div>
                <p style={{ color: color }} className={"text-xl"}>{text}</p>
            </div>


        </>
    )

}

export default StatusPlayer
