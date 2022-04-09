

// styleInp.onchange = evt => {
//     console.log(evt);
//     const [file] = styleInp.files
//     if (file) {
//       document.getElementById('style').src = URL.createObjectURL(file)
//     }
// }

// contentInp.onchange = evt => {
//     const [file] = contentInp.files
//     if (file) {
//       content.src = URL.createObjectURL(file)
//     }
// }

// function beginStylization() {
//     let btn = document.getElementById("stylizeBtn")
//     let btnText = btn.value
//     btn.value = "Wait a moment. Stylizing..."
//     document.getElementById("loadingScreen").style.display = "block";
//     btn.disabled = true;
//     stylize().then(() => {
//         document.getElementById("loadingScreen").style.display = "none";
//         btn.disabled = false;
//         btn.value = btnText
//         }    
//     )
// }

// Animations and transitions
$(".imgAdd").click(function () {
  $(this)
    .closest(".row")
    .find(".imgAdd")
    .before(
      '<div class="col-sm-2 imgUp"><div class="imagePreview"></div><label class="btn btn-primary">Upload<input type="file" class="uploadFile img" value="Upload Photo" style="width:0px;height:0px;overflow:hidden;"></label><i class="fa fa-times del"></i></div>'
    );
});
$(document).on("click", "i.del", function () {
  $(this).parent().remove();
});

function isFileImage(file) {
  const acceptedImageTypes = ["image/jpeg","image/jpg","image/png"];
  return file && $.inArray(file["type"], acceptedImageTypes);
}

var content_loaded = false;
var style_loaded = false;

$(function () {
  $(document).on("change", ".uploadFile", function (oEvent) {
    var uploadFile = $(this);
    var files = !!this.files ? this.files : [];
    if (!files.length || !window.FileReader) return; // no file selected, or no FileReader support
      if (/^image/.test( files[0].type)){ // only image file
    // if (isFileImage(files[0])) {
      var reader = new FileReader(); // instance of the FileReader
      reader.readAsDataURL(files[0]); // read the local file
      reader.onloadend = function (e) {
        // set image data as background of div
        //alert(uploadFile.closest(".upimage").find('.imagePreview').length);
        uploadFile
          .closest(".imgUp")
          .find(".imagePreview")
          .attr('src', e.target.result);
          // .attr("src",""+this.result+"");
          // .css("background-image", "url(" + this.result + ")");
          // .attr("src","url(" + this.result + ")");
          // .src = this.result;
        // Save the image
        // console.log(oEvent.target);

        // const input_name = $(oEvent.target).attr('id').toString();
        // if(input_name.includes('Content')){
        //   console.log('Content is in');
        //   content_loaded = true;
        // } else if (input_name.includes('Style')){
        //   console.log('Style is in');
        //   style_loaded = true;
        // };
        // // If images are loaded, unlock the stylize button
        // if(content_loaded && style_loaded){
        //   $('#Stylize').removeClass('disabled');
        // }
      };
    }
  });
});

$(document).ready(function() {
  $("#btnFetch").click(function() {
    // disable button
    $(this).prop("disabled", true);
    // add spinner to button
    $(this).html(
      `<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Loading...`
    );
  });
});

// From this tutorial: https://getbootstrap.com/docs/5.0/components/tooltips/
// Need to add popper js properly
var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
  return new bootstrap.Tooltip(tooltipTriggerEl)
})


// Stylizing the image
const delay = millis => new Promise((resolve, reject) => {
  setTimeout(_ => resolve(), millis)
});

const loading_state = `<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Stylizing...`

$('#Stylize').click(async function(){
  const prev_state = $('#Stylize').html()

  $('#Stylize').toggleClass('disabled');
  
  $('#Stylize').empty().append(loading_state);

  // await delay(1000);
  // console.log(content_img, style_img, $('#Result'));
  await stylize($('#ContentImage')[0], $('#StyleImage')[0], $('#Result')[0]);
  
  $('#Stylize').toggleClass('disabled');

  $('#Stylize').empty().append(prev_state);
});

// Model stuff

let model = null;

async function getImage(img) {
    // let img = document.getElementById(id)
    
    img = tf.cast(tf.browser.fromPixels(img), 'float32');

    const offset = tf.scalar(255.0);
    // Normalize the image from [0, 255] to [0, 1].
    img = img.div(offset)
    return img
}

$(document).ready(async function() {
  // Load the model.
  console.log('Hello');
  model = await loadModel();
  console.log(model);
  console.log(model.summary());
  console.log(model.inputs);
  console.log(model.outputs);
});

async function loadModel() {
    if (model === null) {
        console.log('Loading model...');
        model = await tf.loadGraphModel('assets/js/tfmodel/model.json');
        // Model warmup.
        // console.log('Warming up model...');
        // await model.executeAsync(
        //     {
        //         'content':tf.randomUniform([256,256,3], dtype ='float32'),
        //         'style':tf.randomUniform([256,256,3], dtype ='float32'),
        //         'iters':tf.scalar(1,'int32'),
        //         'max_resolution':tf.scalar(128,'int32'),
        //     },
        //     ["result"]
        // ); 
    }
    return model
}

async function stylize(content_elem, style_elem, result_elem) {
    try {
        // console.log(tf.getBackend());
        const content = await getImage(content_elem)
        const style = await getImage(style_elem)
        const iters = tf.scalar(
            // document.getElementById('iters').value, 
            1,
            dtype='int32')
        const max_resolution = tf.scalar(
            document.getElementById('max_resolution').value,
            // 256,
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
        await tf.browser.toPixels(result, result_elem);
    } catch (e) {
        console.error(e);
    }
}