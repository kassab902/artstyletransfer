---
layout: default
---


# Achtung! Style Attentional Network!

Written by: [@akassab](https://github.com/kassab902) and [@copilot](https://copilot.github.com/)

So, [attention](https://arxiv.org/abs/1706.03762) is dominating the AI field now. Not only for NLP, but also for pretty much [any *computable* task out there](https://arxiv.org/abs/2103.05247). So, why not use this approach for image stylization as well? After all, if a human were to stylize an image by hand, it's very much expected that they would pay attention to different parts of the style image when deciding how to stylize, say the Sun on the content image. 

So, that's what [these guys did](https://arxiv.org/abs/1812.02342v5). They created a network that takes an *arbitrary* content and style image pair, and then tries to find the best way to blend the two -- using attention. 

Let's see how it works. Embedded below is a TensorflowJS application that will use the network described above two artistically render the content image. Click `Stylize!` to begin the process. Note that the warmup will take a few seconds depending on your hardware since it's using [chunky VGG-19](https://arxiv.org/abs/1409.1556) for inference). 

<section>
{% include stylizer.html %}
</section>

## So how's it work?

Well, the real question is how's it work *in* TensorflowJS, given that the original source came from [PyTorch repo](https://github.com/GlebSBrykin/SANET)? 

We (the copilot & me) will also explain the model architecture in detail. 

### The model architecture

An image is worth a ~~1000 words~~ [16x16 words](https://arxiv.org/abs/2010.11929). So, let's see the image of the architecture. 

![Model architecture](/assets/img/model-arch.png)

That's one messy-looking model, but fear not. The copilot and me are going to explain it all. Let's read the image image from left to right. The two inputs are the content ($$I_c$$) and style ($$I_s$$)images. The content image is the image that we want to stylize. The style image is the image that we want to use as a style reference. 

These two inputs are first fed into the frozen `Encoder VGG` network, but only up to the `block4_conv1` and `block5_conv1` layers. This is because we want to use the encoder to extract features of different complexity of from the content and style images. We'll use these features to create a new image that is a blend of the content and style images.
 
The four short horizontal lines, $$F_c^{r_{41}}$$, $$F_c^{r_{51}}$$ and $$F_s^{r_{41}}$$, $$F_s^{r_{51}}$$ are the content and style representations of the content and style images. 

Now, we apply main ingredient to the mix, the attention! The two pairs of similar sized $$r_{41}$$ and $$r_{51}$$ images are pushed through the following equation:

## The problem with VGG

It's too big. [Unecessarily](https://machinethink.net/blog/compressing-deep-neural-nets/#:~:text=Since%20MobileNet%20is%2032%20times,paper%20by%20Han%20et%20al.) big. So, to make this app runnable on the mobile, we need to somehow make it smaller, without changing its outputs too much. Also, we don't want to train it anew. So, how do we do it?


To be continued...

{% include footer.html %}