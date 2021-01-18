var SGD_LR = 0.002;
var NUM_STEPS = 5;

self.onmessage = function(msg) {
    var classes = msg.data.classes;
    var data = msg.data.data;

    var tensorSize = data.length / (classes + 1);
    var size = Math.sqrt(tensorSize);
    var trainData = data.slice(0, tensorSize * classes);

    var parameters = trainedInit();
    var images = new jsnet.Tensor([classes, size, size, 1], trainData);
    var losses = trainNetwork(parameters, images, NUM_STEPS);

    images = new jsnet.Tensor([classes+1, size, size, 1], data);
    var output = applyNetwork(parameters, images).value;

    var probs = [];
    for (var i = 0; i < classes; ++i) {
        probs.push(Math.exp(output.data[i + classes * classes]));
    }
    self.postMessage({losses: losses, probs: probs});
};

function trainNetwork(parameters, images, steps) {
    var losses = [];
    for (var i = 0; i < steps; ++i) {
        var outputs = applyNetwork(parameters, images);
        var loss = computeLoss(outputs);
        losses.push(loss.value.data[0]);
        loss.backward(new jsnet.Tensor([], [1]));
        for (var j = 0; j < parameters.length; ++j) {
            var param = parameters[j];
            param.gradient.scale(SGD_LR);
            param.value.add(param.gradient);
            param.clearGrad();
        }
    }
    return losses;
}

function computeLoss(outputs) {
    var ways = outputs.value.shape[0];
    var rawMask = [];
    for (var i = 0; i < ways; ++i) {
        for (var j = 0; j < ways; ++j) {
            if (i == j) {
                rawMask.push(1);
            } else {
                rawMask.push(0);
            }
        }
    }
    var mask = new jsnet.Tensor([ways, ways], rawMask);
    var masked = jsnet.mul(outputs, new jsnet.Variable(mask));
    return jsnet.sumOuter(jsnet.sumOuter(masked));
}

function applyNetwork(parameters, images) {
    var output = new jsnet.Variable(images);
    for (var i = 0; i < 4; ++i) {
        output = applyConv(output, parameters.slice(i*4, (i+1)*4));
    }
    output = jsnet.reshape(output, [
        output.value.shape[0],
        output.value.shape[1] * output.value.shape[2] * output.value.shape[3]
    ]);
    output = applyDense(output, parameters.slice(16, 18));
    return jsnet.logSoftmax(output);
}

function applyConv(inputs, parameters, i) {
    var kernel = parameters[0];
	var bias = parameters[1];
    var gamma = parameters[2];
    var beta = parameters[3];
    var output = inputs;
    if (output.value.shape[1] % 2 === 1) {
        output = jsnet.padImages(inputs, 1, 1, 1, 1);
    } else {
        output = jsnet.padImages(inputs, 0, 1, 1, 0);
    }
    output = jsnet.conv2d(output, kernel, 2, 2);
	output = jsnet.add(output, jsnet.broadcast(bias, output.value.shape));
    output = jsnet.normalizeChannels(output);
    output = jsnet.mul(output, jsnet.broadcast(gamma, output.value.shape));
    output = jsnet.add(output, jsnet.broadcast(beta, output.value.shape));
    return jsnet.relu(output);
}

function applyDense(inputs, parameters) {
    var kernel = parameters[0];
    var bias = parameters[1];
    var output = jsnet.matmul(inputs, kernel);
    return jsnet.add(output, jsnet.broadcast(bias, output.value.shape));
}

function trainedInit() {
    var result = [];

    for (var i = 0; i < trainedParameters.length; i += 1) {
        var param = new jsnet.Variable(trainedParameters[i].copy());
        result.push(param);
    }

    return result;
}
