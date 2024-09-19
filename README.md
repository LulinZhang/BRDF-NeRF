# BRDF-NeRF

### [BRDF-NeRF: Neural Radiance Fields with Optical Satellite Images and BRDF Modelling](https://arxiv.org/abs/2409.12014)
*[Lulin Zhang](https://scholar.google.com/citations?user=tUebgRIAAAAJ&hl=fr&oi=ao),
[Ewelina Rupnik](https://erupnik.github.io/),
[Tri Dung Nguyen](https://fr.linkedin.com/in/tri-dung-nguyen-738a57262),
[Stéphane Jacquemoud](https://www.ipgp.fr/~jacquemoud/),
[Yann Klinger](https://www.ipgp.fr/~klinger/)*


![](documents/teaser_v2.png)


## Setup
### Compulsory
The following steps are compulsory for running this repository:
1. Clone the git repository 
```
git clone https://github.com/LulinZhang/BRDF-NeRF.git
```

2. Create virtualenv `spsnerf`
```
conda init
bash -i setup_spsnerf_env.sh
```

### Optional
If you want to prepare the dataset yourself, you'll need to create virtualenv `ba`:
```
conda init
bash -i setup_ba_env.sh
```

## 1. Prepare dataset
You can skip this step and directly download the [Djibouti dataset](https://drive.google.com/file/d/1UFuEiG-_fNTTl22ZHzxiTy0wJi66qfHz/view?usp=sharing).

## Citation
If you find this code or work helpful, please cite:
```
@misc{zhang2024brdfnerfneuralradiancefields,
      title={BRDF-NeRF: Neural Radiance Fields with Optical Satellite Images and BRDF Modelling}, 
      author={Lulin Zhang and Ewelina Rupnik and Tri Dung Nguyen and Stéphane Jacquemoud and Yann Klinger},
      year={2024},
      eprint={2409.12014},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.12014}, 
}
```
