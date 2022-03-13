
let model = null;

async function getImage(id) {
    let img = document.getElementById(id)
    
    img = tf.cast(tf.browser.fromPixels(img), 'float32');

    const offset = tf.scalar(255.0);
    // Normalize the image from [0, 255] to [0, 1].
    img = img.div(offset)
    return img
}


async function loadModel() {
    if (model === null) {
        model = await tf.loadGraphModel('static/model/model.json');
        // Model warmup.
        await model.executeAsync(
            {
                'content':tf.randomUniform([256,256,3], dtype ='float32'),
                'style':tf.randomUniform([256,256,3], dtype ='float32'),
                'iters':tf.scalar(1,'int32'),
                'max_resolution':tf.scalar(128,'int32'),
            },
            ["result"]
        ); 
    }
    return model
}


async function stylize() {
    try {
        console.log(tf.getBackend());
        const content = await getImage('content')
        const style = await getImage('style')
        const iters = tf.scalar(
            document.getElementById('iters').value, 
            dtype='int32')
        const max_resolution = tf.scalar(
            document.getElementById('max_resolution').value,
            dtype='int32')
        
        console.log(content, style);
        console.log('Images loaded successfully!');

        const model = await loadModel();

        console.log('Model loaded successfully!');
        
        console.log(model);
        console.log('Doing stylization...');
        const result = await model.executeAsync(
            {content,style,iters,max_resolution},
            ["result"]
        ); 
        console.log(result);
        await tf.browser.toPixels(result, document.getElementById('result'));
    } catch (e) {
        console.error(e);
    }


}

styleInp.onchange = evt => {
    console.log(evt);
    const [file] = styleInp.files
    if (file) {
      document.getElementById('style').src = URL.createObjectURL(file)
    }
}

contentInp.onchange = evt => {
    const [file] = contentInp.files
    if (file) {
      content.src = URL.createObjectURL(file)
    }
}

function beginStylization() {
    let btn = document.getElementById("stylizeBtn")
    let btnText = btn.value
    btn.value = "Wait a moment. Stylizing..."
    document.getElementById("loadingScreen").style.display = "block";
    btn.disabled = true;
    stylize().then(() => {
        document.getElementById("loadingScreen").style.display = "none";
        btn.disabled = false;
        btn.value = btnText
        }    
    )
}