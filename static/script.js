document.addEventListener("DOMContentLoaded", function() {
    const images = [
        "../static/img/pexels-cottonbro-studio-6332434.jpg",
        "../static/img/pexels-cottonbro-studio-6332432.jpg.",
        "../static/img/pexels-cottonbro-studio-6322489.jpg",
        "../static/img/pexels-cottonbro-studio-6321941.jpg",
        "../static/img/pexels-cottonbro-studio-6321925.jpg"
    ];

    let currentIndex = 0;
    const jumbotron = document.getElementById('jumbotron');

    function changeBackground() {
        jumbotron.style.backgroundImage = `url('${images[currentIndex]}')`;
        currentIndex = (currentIndex + 1) % images.length;
    }

    // Cambiar la imagen de fondo cada 5 segundos (5000 milisegundos)
    setInterval(changeBackground, 5000);
});