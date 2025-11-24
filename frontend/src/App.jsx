import './App.css'
import './index.css'
import MainPage from './Pages/MainPage'
import BoardPage from './Pages/BoardPage'
import { Routes, Route } from 'react-router-dom';


function App() {

  return (
    <div className="App">
      <Routes>
        <Route path="/" element={<MainPage />} />
        <Route path="play" element={<BoardPage />} />
      </Routes>
    </div>
  );
}

export default App
