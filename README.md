# Objects Count Optimization for Text-to-image Diffusion Models

This repository contains the code related to our paper *Objects Count Optimization for Text-to-image Diffusion Models*.

> Oz Zafar\*<sup>1</sup>, Idan Schwartz\*<sup>1</sup>, Lior Wolf<sup>1</sup>
> <sup>1</sup>Tel Aviv University
> \* Denotes equal contribution
>
> We address a persistent challenge in text-to-image models: accurately generating a
specified number of objects. Current models, which learn from image-text pairs,
inherently struggle with this task due to the impossibility of finding an image for
every number. We propose a novel technique that iteratively modifies the text
conditioning and generates images, adjusting the number of objects via a counting
loss, which is derived from the aggregation of attention map peaks. Our method
offers three key advantages: (i) it is a zero-shot method requiring no additional
training; (ii) it is a plug-and-play solution facilitating rapid changes to the counting
and SD method; and (iii) it provides fine-grained user control. Through assessments
of the generation of various objects, we demonstrate that our approach significantly
enhances accuracy.
>
<a href="https://arxiv.org/abs/2408.11721"><img src="https://img.shields.io/badge/arXiv-2408.11721-b31b1b.svg" height=30.5></a> <a href="https://ozzafar.github.io/count_token/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=red" height=30.5></a> <a href="https://colab.research.google.com/drive/1ILVrX288kAmbfRXjz5jtVSHbTSDtyZlX?usp=sharing"><img src="https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252" height=30.5></a> 


<p align="center">
<img src="https://github.com/ozzafar/count_token_optimization/blob/main/docs/results.png" width="800px"/>
<br>
We propose a plug-and-play optimization of object counting accuracy of a text-to-image model based on detection models.
</p>

## Installations:

#### Hugging face

Run this command to log in with your HF Hub token if you haven't before:

`huggingface-cli login`

#### Create conda environment

`conda env create -f requirements.yml`

`conda activate objects_count_optimization`

#### Dependencies

###### Object Counting model

Our optimization is based on CLIP-COUNT, a vision-language model for class-agnostic object counting.

The code can be easily adapted to other models, if you will to utilize CLIP-COUNT download the pre-trained weights from their [repository](https://github.com/songrise/CLIP-Count) and locate it under the local clip_count folder.

###### Evaluation

Our evaluation is based both on CLIP-COUNT and YOLO.

For CLIP-COUNT setup, refer to previous section.

For YOLO setup, please refer to  [YOLOv9 docs](https://docs.ultralytics.com/models/yolov9/).

## Run and Evaluate:
<p align="center">
<img src="https://github.com/ozzafar/count_token_optimization/blob/main/docs/method_diagram.png" width="450px"/>
<br>
An overview of our method for optimizing a new discriminative token representation ($v_c$) using a pre-trained object detection model. For the prompt `A photo of a $S_c$ 6 beads,' we expect the output generated with the count $c$ to be 6. The object detection model, however, indicates that the amount of beads in the generated image is a lot bigger. We generate images iteratively and optimize the token representation using MSE loss. Once $v_c$ has been trained, more images of the target amount can be generated by including it in the context of the input text.
</p>


To train and evaluate use:
```
python run.py --clazz beads --amount 6 --train True  --evaluate True
```

#### Hyperparameters:
The hyperparameters can be changed in the `config.py` script. Note that the paper results are based on [SDXL-turbo](https://huggingface.co/stabilityai/sdxl-turbo).

#### Outputs
The script will create folders and store tokens representation in `token` and the images in `img`.


## Citation

If you make use of our work, please cite our paper:

```
@misc{zafar2024iterativeobjectcountoptimization,
      title={Iterative Object Count Optimization for Text-to-image Diffusion Models}, 
      author={Oz Zafar and Lior Wolf and Idan Schwartz},
      year={2024},
      eprint={2408.11721},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.11721}, 
}
```