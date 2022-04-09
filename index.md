---
layout: default
---

Written by: [@akassab](https://github.com/kassab902) and [@copilot](https://copilot.github.com/). Code available [here](https://github.com/kassab902/artstyletransfer).

So, [attention](https://arxiv.org/abs/1706.03762) is dominating the AI field now. Not only for NLP, but also for pretty much [any *computable* task out there](https://arxiv.org/abs/2103.05247). So, why not use this approach for image stylization as well? After all, if a human were to stylize an image by hand, it's very much expected that they would pay attention to different parts of the style image when deciding how to stylize, say the Sun on the content image. 

So, that's what [these guys did](https://arxiv.org/abs/1812.02342v5). They created a network that takes an *arbitrary* content and style image pair, and then tries to find the best way to blend the two -- using attention. 

Let's see how it works. Embedded below is a TensorflowJS application that will use the network described above two artistically render the content image. Click `Stylize!` to begin the process. Note that the warmup will take a few seconds depending on your hardware since it's using [chunky VGG-19](https://arxiv.org/abs/1409.1556) for inference). 

<section>
{% include stylizer.html %}
</section>

## So how does it work?

Well, the real question is how's it work *in* TensorflowJS, given that the original source came from [PyTorch repo](https://github.com/GlebSBrykin/SANET)? 

We (the copilot & me) will also explain the model architecture in detail. 

### The model architecture

An image is worth a ~~1000 words~~ [16x16 words](https://arxiv.org/abs/2010.11929). So, let's see the image of the architecture. 

![Model architecture]({{ "/assets/img/model-arch.png" | prepend: site.baseurl }})

That's one messy-looking model, but fear not. The copilot and me are going to explain it all. Let's read the image image from left to right. The two inputs are the content ($$I_c$$) and style ($$I_s$$)images. The content image is the image that we want to stylize. The style image is the image that we want to use as a style reference. 

These two inputs are first fed into the frozen `Encoder VGG` network, but only up to the `block4_conv1` and `block5_conv1` layers. This is because we want to use the encoder to extract features of different complexity of from the content and style images. We'll use these features to create a new image that is a blend of the content and style images.
 
The four short horizontal lines, $$F_c^{r_{41}}$$, $$F_c^{r_{51}}$$ and $$F_s^{r_{41}}$$, $$F_s^{r_{51}}$$ are the content and style representations of the content and style images. 

Now, we apply main ingredient to the mix, the attention! The two pairs of similar sized $$r_{41}$$ and $$r_{51}$$ images are pushed through the following module:

![Attention module]({{ "/assets/img/attn.png" | prepend: site.baseurl }})

The purpose of this segment is to embed local style patterns Fs within content feature maps Fc, by using learned $$1\times1$$ convolutions and attention (shown as $$3\times3$$ black-white matrix). This block implements the following mathematical relationship between inputs $$F_c$$ and $$F_s$$ to outputs $$F_cs$$:

<!-- ![Equation]({{ "/assets/img/eqn.png" | prepend: site.baseurl }}) -->

$$F^i_{cs} = \frac{1}{C(F)} \sum_{\forall j} \exp(f(\bar{F^i_c})^T g(F^j_s) )h(F^j_s)$$

This might look familiar to someone who knows what the $$softmax$$ function is. In fact, this *is* a softmax function multiplied with the $$h(Fs)$$. Conceptually, the purpose of the above equation is to create a way of re-constructing the content using the style feature map. In particular, the equation first calculates attention between the two $$g(\bar{F_s}) = W_g\bar{F_s}$$ and $$f(\bar{F_c}) = W_f\bar{F_c}$$ feature maps. This is done in the $$F^i_{cs} = \frac{1}{C(F)} \sum_{\forall j} \exp(f(\bar{F^i_c})^T g(F^j_s) )$$ part, where $$C(F) = \sum_{\forall{j}}\exp(f(\bar{F^i_c})^T g(F^j_s) )$$ is the normalization factor (makes softmax results fall between 0 and one, like probabilities). The softmax operation yields a $$32^2\times 32^2$$ matrix of probabilities. This matrix is then multiplied with This matrix is then multiplied with $$h(F^j_s)$$ producing a weighted average of the pixels in the style feature map. Thus, the SANet block learns such a relationship between the content and style (using $$W_f$$ and $$W_g$$ weights) that can be used to recombine style feature maps into the final image. This is the core idea of SANet that allows it to create much more detailed stylized outputs.

### Implementing SANet

#### First Deployment Pains
SANet authors provide the implementation of their paper in the following [github repository](https://github.com/GlebSBrykin/SANET). Initial challenges ranged from setting up the environment correctly in order to run inference, to fixing various bugs caused by outdated dependencies. The project is written in PyTorch. However, right from the start of this project, the end-goal was to create a user-friendly web-application. We began our implementation of this webapp by deploying the code directly on DigitalOcean. This deployment, however, would prove to be quite problematic, since our web app was extremely slow to load and prone to crashes due to the limited RAM the server had.  

Soon it became apparent that deploying SANet meant renting a costly, GPU-backed server.  Otherwise it would likely get overwhelmed by computational requirements. This, coupled with the internet latency issues and likely a private nature of the images uploaded there, prompted us to seek other solutions.  

As of Jan 2022, TensorflowJS is likely the only javascript based ML framework that is both fast and easy to use. The javascript-based nature of TFJS means that any model that is implemented within TFJS can run without a server. A server provides around 80MB of network weights just once, after which the client runs the network on their own device. This eliminates the latency, privacy and cost issues associated with the original PyTorch implementation. However, we will have to transfer the PyTorch network weights to TensorflowJS to do that.

#### Transferring the Network to TensorflowJS


We quickly learned that there is no direct path to converting the PyTorch model to Tensorflow, let alone to TensorflowJS. There is, however, an intermediate framework, called Open neural network exchange (ONNX), which aims to act as a glue between various ML frameworks, such as MXNet, Jax, Pytorch and Tensorflow. It is possible to export specially constructed PyTorch models into ONNX format. The original implementation does not satisfy the ONNX conversion rules. To fix this, we first re-implemented the SANet within the PyTorch framework, namely, by writing a new implementation of the SANet block which avoids the unsupported dynamic reshape operation by a supported `flatten(A,2)` operation. Additionally, in order to support varying image input sizes, we disable constant folding and specify dynamic axes in onnx.export function (dynamic axes, like height and width are not fixed). After these modifications and fixing other minor issues, we export the PyTorch model into three separate parts: `encoder.onnx`, `decoder.onnx` and `transform.onnx`, corresponding to the same three parts in the architecture. This split into three parts is necessary since the full SANet algorithm requires a for-loop, which is not possible to export within ONNX. 


Next, we used the [onnx-tensorflow](https://github.com/onnx/onnx-tensorflow) library to transform the three ONNX model files into three Tensorflow saved-models. We then loaded in the three Tensorflow saved-models separately and created a new `tf.Module` subclass, called `TFStyle`. This class would perform all the necessary calculations on the two input images, including the required for-loop operation. We then validated that the `TFStyle` class outputs match those of the PyTorch implementation to a floating point precision. At this point, we only need to set-up the `Tensorflow_Serving` signature for this model in order to be able to safely export it to a TensorflowJS environment. We designed the serving signature to:



* Accept any two arbitrarily-shaped input images
* Accept two additional parameters (num_iterations, max_resolution)
* Rescale and normalize the images (using max_resolution)
* Perform stylization (including the num_iterations for style strength)
* Reshape the output to the shape of the original content image
* Return results

Note that, as it currently stands, the `num_iterations` argument is fixed to be `2` in the website. This means that the content image is stylized once and then the result is stylized once more with the same style. Higher `num_iterations` will strengthen the style influence at the cost of more computation. ~~We also lack a quality control lever~~ We have added a quality control lever, but some bugs still persist (images are generated with the same shape, despite the quality changes).


Finally, the TFStyle model, along with the Serving signature is exported to a JSON format using the TensorflowJS converter tool. This generates an 80MB model folder that can be executed using the TensorflowJS framework, within a web app. 


#### Implementing the website

The website was implemented in two stages. First, a draft implementation was made, in pure HTML and Javascript, in order to validate that the exported TFJS model was not corrupted. 

During development of the first iteration, several problems were encountered and fixed. These include problems with javascript package versions, image data loading from the site and various TensorflowJS issues, mainly due to lack of knowledge of Javascript. 

Finally, after troubleshooting the errors in the first iteration of the website, we implemented an interactive blog-like website, shown in Figure 5. (right). This website utilizes a popular and simple blog-creation tool, Jekyll. Jekyll works by converting a plain markup language into a blog-like website. This conversion happens only once and the resulting HTML and CSS files are saved on disk. However, in order to make a website available to the general public, we needed to use a service called GitHub-Pages, which allows for free placement of a website under a unique domain name. We used GitHub-Pages, alongside with GitHub Actions (a continuous Integration service) in order to automatically recompile and update the website when GitHub detects any change with the repository. All this happens automatically and requires no human supervision. Lastly, we added a blog around the interactive stylization app that describes the inner workings and details of the SANet to serve as a future portfolio/showcase website.


### Limitations

With the current implementation of the style transfer web application, there are several outstanding problems. One of the most noticeable ones is the loading delay. The first run of the model can take up to 30s, while the subsequent calls only need up to a second (depending on the image resolution). We think this could be fixed by using knowledge distillation on encoder and decoder VGG networks, as these take up more than 95% of the network weights. Quantization is also a promising candidate and can be applied relatively easily using the TFLite library.  Another solution would be to forego the VGG architecture altogether and instead use MobileNet architecture, as these are much more performant and optimized for mobile use. This would likely require re-designing some key components of the SANet. 

Finally, the num_iterations and max_resolution variables are not available to the users on the published blog. This could be easily fixed by adding the appropriate input sliders to the form.


### Acknowledgements 

We thank Dr. Jacob Levman for offering guidance and insightful comments during the development of this website. We also thank Jessica Levman for providing us with sample art for testing our early iterations of the style transfer application. 

### Conclusion and Future Work

In this report we discussed our achievements while working on the subject of neural style transfer. In doing so, we learned the basics of PyTorch and ONNX, mastered Tensorflow and TensorflowJS, and worked with Jekyll and various development tools, such as Github actions and Github pages. 

In the future, we would like to make the SANet faster, smaller and more accessible. We would also like to implement a video stylization application. Last but not least, this application could be used as a basis for an iOS or an Android application. 

{% include footer.html %}