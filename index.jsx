import React from 'react';

const images = ["image1.jpg", "image2.jpg"]; // Replace with your image data

const App = () => {
  return (
    <div>
      <h1>Rover Images</h1>
      {images.map((image, index) => (
        <div key={index}>
          <img src={`static/uploads/${image}`} alt="Rover Image" width="300" />
        </div>
      ))}
    </div>
  );
};

export default App;