
app.use(express.static('public'));

io.on('connection', function(socket) {
    console.log('a user connected');
    
    socket.on('start_camera', function() {
        console.log('camera started');
    });
    
    socket.on('predict', function(data) {
        console.log('received image data:', data);
        // perform prediction here
        let predictionResult = 'example prediction';
        socket.emit('prediction_result', {prediction: predictionResult});
    });
    
    socket.on('disconnect', function() {
        console.log('user disconnected');
    });
});

server.listen(5000, function() {
    console.log('Server listening on port 5000');
});